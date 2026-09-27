"""
BioText Transformer — Version LÉGÈRE & RAPIDE pour PC Gamer (v2 corrigée)
=========================================================================
Optimisé pour :
  • RTX 3060 (12 Go VRAM) — AMP fp16 + TF32 + attention fusionnée (SDPA)
  • Intel Core Ultra 5 245KF — tous les cœurs utilisés en mode CPU
  • Ratio vitesse/plasticité équilibré

Mécanismes bio :
  ✓ Plasticité des poids (Hebbian + STDP avec traces temporelles)
  ✓ Plasticité structurelle (pruning/growing d'arêtes, budget fixe)
  ✓ Sparse coding (top-k activations)
  ✓ Phase sommeil (replay + consolidation + élagage)

Usage :
    python biotext_light.py

Corrections v2 (par rapport à la v1) :
  1. Loss initial ≈ log(256) ≈ 5.5 : embeddings initialisés en N(0, 0.02²)
     (la v1 utilisait N(0,1) avec poids liés → logits énormes → loss ≈ 40).
  2. Masque causal ajouté : la v1 laissait le modèle « voir » le token futur
     qu'il devait prédire (fuite de données → génération dégénérée).
  3. Hebbian/STDP : activations standardisées (corrélations bornées),
     composante hebbienne séparée `H` avec décroissance et plafond de norme
     → impossible de faire exploser les poids. Appliqué à Q, K, V et à la FFN.
  4. Connectivité dynamique : scores d'importance normalisés par le max,
     élagage plafonné (max 25 % des arêtes par sommeil), plancher d'arêtes,
     repousse systématique. Tenseurs de taille fixe (`max_edges`) + masque
     d'activation → l'optimiseur suit toujours les bons paramètres.
  5. Top-k sparse : plancher `topk_min` (3 % de 64 = 1 seule feature en v1).
  6. API AMP modernes : torch.amp.autocast('cuda') / torch.amp.GradScaler('cuda').
  7. Sauvegarde relative : ./biotext_light_weights.pt
  8. Génération : sampling top-k + température + pénalité de répétition.
"""

import os
import math
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
from pathlib import Path


# ─────────────────────────────────────────────
# 0. CONFIG — AJUSTÉE POUR PC GAMER
# ─────────────────────────────────────────────
class Config:
    # ── DÉTECTION HARDWARE ──────────────────────────────────
    cuda_avail    = torch.cuda.is_available()
    gpu_name      = torch.cuda.get_device_name(0) if cuda_avail else "CPU"

    # Budget mémoire GPU (RTX 3060 = 12 Go, on vise 60-70 %)
    if cuda_avail:
        props = torch.cuda.get_device_properties(0)
        gpu_mem_gb  = props.total_memory / 1e9
        max_gpu_mem = gpu_mem_gb * 0.65
    else:
        max_gpu_mem = 2.0  # fallback CPU (RAM)

    device = "cuda" if cuda_avail else "cpu"

    # Threads CPU : tous les cœurs disponibles (Core Ultra 5 245KF = 14 cœurs)
    cpu_threads = max(1, os.cpu_count() or 1)

    # ── ARCHITECTURE RÉDUITE ──────────────────────────────
    vocab_size    = 256          # ASCII / octets
    seq_len       = 64           # court = rapide

    d_model       = 64
    n_heads       = 2            # 64 / 2 têtes = 32 par tête
    n_layers      = 2
    d_ff          = 128
    dropout       = 0.05

    # ── PLASTICITÉ ────────────────────────────────────────
    # Structurelle (connectivité dynamique)
    max_edges     = 256          # budget maximal d'arêtes par couche (taille fixe)
    tau_alloc     = 0.6          # occupation initiale du budget (60 % → 153 arêtes)
    tau_prune     = 0.15         # seuil d'élagage sur le score NORMALISÉ (score / max)
    prune_frac_max= 0.25         # on n'élague jamais plus de 25 % des arêtes par sommeil
    min_edges     = 32           # plancher : jamais moins de 32 arêtes actives
    grow_step     = 8            # arêtes d'exploration créées à chaque sommeil
    edge_proj_init= 0.02         # écart-type initial du poids d'une arête neuve
    gate_lr_mult  = 5.0          # lr des logits de gates = 5 × lr (gates plus réactives)
    importance_ema= 0.9          # lissage EMA du score d'importance

    # Poids (Hebbian/STDP)
    use_hebbian   = True         # False = ablation
    eta_hebb      = 2e-4         # pas hebbien (sur corrélations standardisées ∈ [-1, 1])
    eta_stdp      = 1e-4         # pas STDP
    stdp_tau      = 15.0         # constante de temps des traces (en tokens)
    decay_w       = 0.99         # oubli synaptique de la composante hebbienne H
    hebb_max_ratio= 0.05         # ||H||_F ≤ 5 % de ||W||_F (garde-fou anti-explosion)

    # Sparse coding
    use_sparse    = True         # False = ablation
    topk_ratio    = 0.03         # 3 % d'activations conservées…
    topk_min      = 4            # …avec un plancher (3 % de 64 = 1 seule feature sinon)

    # ── PHASE SOMMEIL ──────────────────────────────────────
    use_sleep     = True         # False = ablation
    replay_cap    = 512          # taille du replay buffer (séquences)
    sleep_every   = 30           # consolidation toutes les 30 étapes
    sleep_steps   = 10           # étapes de replay par sommeil
    sleep_lr_mult = 0.5          # lr réduit pendant le sommeil (consolidation douce)

    # ── ENTRAÎNEMENT ───────────────────────────────────────
    lr            = 3e-3         # petit modèle → lr élevé pour converger en 100 étapes
    weight_decay  = 0.01
    batch_size    = 8
    grad_accum    = 2            # accumulation de gradients (batch effectif = 16)
    train_steps   = 300          # ≈ 20 s sur CPU, quelques secondes sur RTX 3060
    seed          = 42

    # ── GÉNÉRATION ─────────────────────────────────────────
    gen_len       = 120
    gen_temperature = 0.7
    gen_topk      = 8
    gen_rep_penalty = 1.3        # pénalise les caractères répétés récemment

    # ── OPTIMISATIONS GPU ──────────────────────────────────
    mixed_precision = cuda_avail  # AMP fp16 si GPU
    amp_dtype       = torch.float16
    pin_memory      = cuda_avail

    # ── SAUVEGARDE ─────────────────────────────────────────
    save_path     = "./biotext_light_weights.pt"

    @classmethod
    def print_config(cls):
        print("╔════════════════════════════════════════════════════════╗")
        print("║ BioText Transformer — CONFIG PC GAMER (v2)             ║")
        print("╠════════════════════════════════════════════════════════╣")
        print(f"║ Device         : {cls.device.upper():37} ║")
        print(f"║ GPU            : {cls.gpu_name[:37]:37} ║")
        if cls.cuda_avail:
            print(f"║ GPU Mem Budget : {cls.max_gpu_mem:.1f} Go (max){' ':24} ║")
        else:
            print(f"║ Threads CPU    : {cls.cpu_threads:<37} ║")
        arch = f"d_model={cls.d_model}, n_layers={cls.n_layers}, d_ff={cls.d_ff}"
        print(f"║ Modèle         : {arch:37} ║")
        bs = f"{cls.batch_size} (×{cls.grad_accum} accum = eff. {cls.batch_size * cls.grad_accum})"
        print(f"║ Batch size     : {bs:37} ║")
        print(f"║ Mixed Precision: {str(cls.mixed_precision):37} ║")
        print(f"║ Hebbian/STDP   : {str(cls.use_hebbian):37} ║")
        print(f"║ Sparse coding  : {str(cls.use_sparse) + f' (top-k {cls.topk_ratio:.0%}, min {cls.topk_min})':37} ║")
        print(f"║ Sommeil        : {str(cls.use_sleep) + f' (toutes les {cls.sleep_every} étapes)':37} ║")
        print("╚════════════════════════════════════════════════════════╝")


cfg = Config()


def setup_backend():
    """Réglages globaux PyTorch : threads CPU, TF32 sur Ampere (RTX 3060), seed."""
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    if cfg.cuda_avail:
        # TF32 : matmuls ~2× plus rapides sur RTX 30xx sans perte notable
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    else:
        torch.set_num_threads(cfg.cpu_threads)


# ─────────────────────────────────────────────
# 1. TOP-K SPARSE GATE
# ─────────────────────────────────────────────
class TopKGate(nn.Module):
    """Ne conserve que les k activations les plus fortes (en valeur absolue)
    par token — analogue du sparse coding cortical (1-4 % de neurones actifs)."""
    def __init__(self, k_ratio=0.03, k_min=4):
        super().__init__()
        self.k_ratio = k_ratio
        self.k_min   = k_min
        self.last_active_frac = 0.0   # mesure de sparsité (pour les stats)

    def forward(self, x):
        if not cfg.use_sparse:
            return x
        dim = x.shape[-1]
        k = min(dim, max(self.k_min, math.ceil(dim * self.k_ratio)))
        topk_vals, _ = torch.topk(x.abs(), k, dim=-1)
        threshold = topk_vals[..., -1:].detach()
        mask = (x.abs() >= threshold).to(x.dtype)
        self.last_active_frac = k / dim
        return x * mask


# ─────────────────────────────────────────────
# 2. PLASTICITÉ STRUCTURELLE (MD-DCM)
# ─────────────────────────────────────────────
class FastDynamicConnectivity(nn.Module):
    """Arêtes dynamiques entre features du flux résiduel.

    Conception v2 : tous les tenseurs ont une taille FIXE `max_edges` et un
    masque booléen `active`. Élaguer = désactiver un slot ; faire pousser =
    réactiver un slot avec une nouvelle paire (src, dst). Ainsi les
    nn.Parameter ne sont jamais recréés et l'optimiseur reste cohérent.

    Score d'importance : s_e = g_e × |corr(src, dst)|  (corrélation de Pearson
    sur les échantillons du batch), lissé par EMA puis NORMALISÉ par le max
    des arêtes actives → l'échelle ne dépend plus de l'amplitude des activations.
    """
    def __init__(self, d_model, max_edges=256):
        super().__init__()
        self.d_model   = d_model
        self.max_edges = max_edges

        n_init = max(cfg.min_edges, int(max_edges * cfg.tau_alloc))
        src, dst = self._sample_edges(max_edges)
        active = torch.zeros(max_edges, dtype=torch.bool)
        active[:n_init] = True

        self.register_buffer("edge_src", src)
        self.register_buffer("edge_dst", dst)
        self.register_buffer("active", active)
        self.register_buffer("importance", torch.zeros(max_edges))
        self.register_buffer("age", torch.zeros(max_edges, dtype=torch.long))  # nb de mises à jour

        # logits > 0 → gate initiale ≈ 0.73 (une arête neuve est « ouverte »)
        self.logits    = nn.Parameter(torch.full((max_edges,), 1.0))
        self.edge_proj = nn.Parameter(torch.randn(max_edges) * cfg.edge_proj_init)

    # ---- utilitaires ------------------------------------------------------
    def _sample_edges(self, n, device="cpu"):
        """Tire n paires (src, dst) sans boucle sur soi-même."""
        src = torch.randint(0, self.d_model, (n,), device=device)
        dst = torch.randint(0, self.d_model, (n,), device=device)
        same = src == dst
        dst[same] = (dst[same] + 1) % self.d_model
        return src, dst

    @property
    def gates(self):
        return torch.sigmoid(self.logits)

    @property
    def n_active(self):
        return int(self.active.sum().item())

    def active_gates(self):
        return self.gates[self.active]

    # ---- forward ----------------------------------------------------------
    def forward(self, x):
        if not self.active.any():
            return x
        # Contribution de chaque arête : x[src] × gate × poids, masquée si inactive
        eff = self.gates * self.edge_proj * self.active.to(x.dtype)  # [E]
        contrib = x[..., self.edge_src] * eff                        # [B, T, E]
        # Accumulation vectorisée sur les destinations (remplace la boucle python v1)
        delta = torch.zeros_like(x).index_add(-1, self.edge_dst, contrib)
        return x + delta

    # ---- importance -------------------------------------------------------
    @torch.no_grad()
    def update_importance(self, x):
        """EMA de s_e = g_e × |corr_Pearson(x[src], x[dst])| sur le batch."""
        if not self.active.any():
            return
        flat = x.reshape(-1, x.shape[-1]).float()          # [N, D]
        flat = flat - flat.mean(dim=0, keepdim=True)
        std  = flat.std(dim=0, keepdim=True) + 1e-6
        z    = flat / std                                  # features standardisées
        corr = (z[:, self.edge_src] * z[:, self.edge_dst]).mean(dim=0).abs()  # ∈ [0, 1]
        score = self.gates * corr
        a = cfg.importance_ema
        self.importance = torch.where(
            self.active, a * self.importance + (1 - a) * score, self.importance
        )
        self.age += self.active.long()

    # ---- pruning / growing ------------------------------------------------
    @torch.no_grad()
    def prune_and_grow(self, optimizer=None):
        """Élague les arêtes faibles (score normalisé < tau_prune), puis fait
        pousser de nouvelles arêtes. Retourne (n_pruned, n_grown)."""
        n_act = self.n_active
        if n_act == 0:
            n_pruned = 0
        else:
            imp = self.importance.clone()
            imp[~self.active] = float("inf")            # les inactives ne comptent pas
            max_imp = self.importance[self.active].max().clamp_min(1e-8)
            norm_score = imp / max_imp                  # ∈ [0, 1] pour les actives

            # Candidates : score faible ET arête assez « vieille » pour être jugée
            mature = self.age >= 5
            cand = (norm_score < cfg.tau_prune) & self.active & mature

            # Garde-fous : max 25 % élaguées, jamais sous min_edges
            max_prune = min(int(cfg.prune_frac_max * n_act), max(0, n_act - cfg.min_edges))
            n_pruned = min(int(cand.sum().item()), max_prune)
            if n_pruned > 0:
                cand_idx = cand.nonzero(as_tuple=True)[0]
                # On retire d'abord les plus faibles
                order = norm_score[cand_idx].argsort()
                kill = cand_idx[order[:n_pruned]]
                self.active[kill] = False
                self.importance[kill] = 0.0
                self.age[kill] = 0

        # Croissance homéostatique : grow_step arêtes d'exploration + la moitié
        # des élaguées. Équilibre atteint quand n_pruned ≈ 2·grow_step par sommeil :
        # le nombre d'arêtes peut monter OU descendre (jamais sous min_edges).
        free = (~self.active).nonzero(as_tuple=True)[0]
        n_grow = min(len(free), cfg.grow_step + n_pruned // 2)
        if n_grow > 0:
            slots = free[torch.randperm(len(free), device=free.device)[:n_grow]]
            src, dst = self._sample_edges(n_grow, device=self.edge_src.device)
            self.edge_src[slots] = src
            self.edge_dst[slots] = dst
            self.active[slots] = True
            self.age[slots] = 0
            # Importance initiale = médiane des actives (évite l'élagage immédiat)
            if self.active.sum() > n_grow:
                med = self.importance[self.active].median()
            else:
                med = torch.tensor(0.0, device=self.importance.device)
            self.importance[slots] = med
            # Réinitialisation des paramètres du slot (nouvelle synapse)
            self.logits.data[slots] = 1.0
            self.edge_proj.data[slots] = torch.randn(n_grow, device=self.edge_proj.device) * cfg.edge_proj_init
            # Réinitialisation de l'état Adam de ces slots
            if optimizer is not None:
                for p in (self.logits, self.edge_proj):
                    st = optimizer.state.get(p)
                    if st:
                        for key in ("exp_avg", "exp_avg_sq"):
                            if key in st:
                                st[key][slots] = 0.0
        return n_pruned, n_grow


# ─────────────────────────────────────────────
# 3. HEBBIAN / STDP (STABILISÉ)
# ─────────────────────────────────────────────
class FastHebbianSTDP:
    """Plasticité locale des poids, appliquée aux projections Q, K, V et à la FFN.

    Hebbian : Δw = η_h · corr(post, pre)          (activations standardisées)
    STDP    : Δw = η_s · (post·trace_pre − trace_post·pre)
              traces exponentielles le long de l'axe temporel (positions T) :
              pré AVANT post → renforcement (LTP), post AVANT pré → affaiblissement (LTD)

    Stabilité (v2) : la contribution hebbienne est stockée séparément dans H
    (W_effectif = W_gradient + H). H décroît à chaque pas (oubli synaptique)
    et sa norme est plafonnée à `hebb_max_ratio` × ||W||. Les poids appris par
    gradient ne sont donc jamais érodés ni explosés.
    """
    def __init__(self, eta_hebb=2e-4, eta_stdp=1e-4, tau=15.0, decay=0.99, max_ratio=0.1):
        self.eta_hebb  = eta_hebb
        self.eta_stdp  = eta_stdp
        self.tau       = tau
        self.decay     = decay
        self.max_ratio = max_ratio
        self.enabled   = True
        self.H         = {}       # nom → composante hebbienne accumulée
        self._kernels  = {}       # cache des noyaux temporels par (T, device)
        self.last_update_norm = {}

    def _kernel(self, T, device):
        """Noyau causal K[t, s] = exp(-(t-s)/tau) pour s < t (trace du passé)."""
        key = (T, str(device))
        if key not in self._kernels:
            t = torch.arange(T, device=device, dtype=torch.float32)
            diff = t[:, None] - t[None, :]
            K = torch.exp(-diff / self.tau) * (diff > 0).float()
            self._kernels[key] = K
        return self._kernels[key]

    @staticmethod
    def _standardize(a):
        """Standardise chaque feature sur les échantillons (B·T)."""
        a = a.float()
        flat = a.reshape(-1, a.shape[-1])
        mu = flat.mean(dim=0)
        sd = flat.std(dim=0) + 1e-5
        return ((a - mu) / sd)

    @torch.no_grad()
    def step(self, name, weight, pre_act, post_act):
        if not (self.enabled and cfg.use_hebbian):
            return
        B, T, _ = pre_act.shape
        N = B * T
        pre  = self._standardize(pre_act.detach())     # [B, T, d_in]
        post = self._standardize(post_act.detach())    # [B, T, d_out]

        # Hebbian : matrice de corrélation post × pre  → [d_out, d_in]
        dw_hebb = self.eta_hebb * torch.einsum("btp,btq->pq", post, pre) / N

        # STDP : traces temporelles causales le long de la séquence
        K = self._kernel(T, pre.device)                # [T, T]
        pre_trace  = torch.einsum("ts,bsq->btq", K, pre)
        post_trace = torch.einsum("ts,bsp->btp", K, post)
        ltp = torch.einsum("btp,btq->pq", post, pre_trace)   # pré avant post
        ltd = torch.einsum("btp,btq->pq", post_trace, pre)   # post avant pré
        dw_stdp = self.eta_stdp * (ltp - ltd) / N

        dw = dw_hebb + dw_stdp
        # Clip par entrée (sécurité : les traces peuvent amplifier ~tau fois)
        lim = 3.0 * (self.eta_hebb + self.eta_stdp)
        dw = dw.clamp_(-lim, lim).to(weight.dtype)

        # Mise à jour de la composante hebbienne H (avec oubli synaptique)
        H_old = self.H.get(name)
        if H_old is None:
            H_old = torch.zeros_like(weight)
        H_new = self.decay * H_old + dw

        # Plafond : ||H|| ≤ max_ratio × ||W_grad||
        w_grad = weight.data - H_old
        max_norm = self.max_ratio * w_grad.norm()
        h_norm = H_new.norm()
        if h_norm > max_norm:
            H_new = H_new * (max_norm / (h_norm + 1e-12))

        weight.data.add_(H_new - H_old)     # W_eff = W_grad + H_new
        self.H[name] = H_new
        self.last_update_norm[name] = dw.norm().item()

    def state_dict(self):
        return {k: v.cpu() for k, v in self.H.items()}

    def load_state_dict(self, sd, device="cpu"):
        self.H = {k: v.to(device) for k, v in sd.items()}


# ─────────────────────────────────────────────
# 4. ATTENTION OPTIMISÉE (CAUSALE)
# ─────────────────────────────────────────────
class FastBioAttention(nn.Module):
    def __init__(self, d_model, n_heads, plasticity, layer_id):
        super().__init__()
        assert d_model % n_heads == 0, "d_model doit être divisible par n_heads"
        self.d_model    = d_model
        self.n_heads    = n_heads
        self.d_head     = d_model // n_heads
        self.layer_id   = layer_id
        self.plasticity = plasticity

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        self.gate = TopKGate(cfg.topk_ratio, cfg.topk_min)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        B, T, D = x.shape
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        # Plasticité locale (uniquement en entraînement)
        if self.training:
            pfx = f"layer{self.layer_id}.attn"
            self.plasticity.step(f"{pfx}.q", self.W_q.weight, x, Q)
            self.plasticity.step(f"{pfx}.k", self.W_k.weight, x, K)
            self.plasticity.step(f"{pfx}.v", self.W_v.weight, x, V)

        # Multi-têtes : [B, H, T, d_head]
        Q = Q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        K = K.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        V = V.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        # Attention causale fusionnée (Flash/mem-efficient sur GPU, rapide sur CPU).
        # is_causal=True : chaque position ne voit que le passé → pas de fuite.
        out = F.scaled_dot_product_attention(
            Q, K, V, is_causal=True,
            dropout_p=cfg.dropout if self.training else 0.0,
        )

        out = out.transpose(1, 2).contiguous().view(B, T, D)
        out = self.gate(out)              # sparse coding
        out = self.W_o(out)
        return self.drop(out)


# ─────────────────────────────────────────────
# 5. FFN LÉGER (avec plasticité)
# ─────────────────────────────────────────────
class FastBioFFN(nn.Module):
    def __init__(self, d_model, d_ff, plasticity, layer_id):
        super().__init__()
        self.fc1  = nn.Linear(d_model, d_ff, bias=False)
        self.fc2  = nn.Linear(d_ff, d_model, bias=False)
        self.gate = TopKGate(cfg.topk_ratio, cfg.topk_min)
        self.drop = nn.Dropout(cfg.dropout)
        self.plasticity = plasticity
        self.layer_id   = layer_id

    def forward(self, x):
        h = F.gelu(self.fc1(x))
        h = self.gate(h)                  # sparse coding sur la couche cachée
        out = self.fc2(h)
        if self.training:
            pfx = f"layer{self.layer_id}.ffn"
            self.plasticity.step(f"{pfx}.fc1", self.fc1.weight, x, h)
            self.plasticity.step(f"{pfx}.fc2", self.fc2.weight, h, out)
        return self.drop(out)


# ─────────────────────────────────────────────
# 6. COUCHE BIO COMPLÈTE
# ─────────────────────────────────────────────
class FastBioLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, plasticity, layer_id):
        super().__init__()
        self.attn = FastBioAttention(d_model, n_heads, plasticity, layer_id)
        self.ffn  = FastBioFFN(d_model, d_ff, plasticity, layer_id)
        self.dcm  = FastDynamicConnectivity(d_model, cfg.max_edges)
        self.ln1  = nn.LayerNorm(d_model)
        self.ln2  = nn.LayerNorm(d_model)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))    # pré-LayerNorm (plus stable)
        x = x + self.ffn(self.ln2(x))
        x = self.dcm(x)                   # arêtes dynamiques inter-features
        if self.training:
            self.dcm.update_importance(x.detach())
        return x


# ─────────────────────────────────────────────
# 7. MODÈLE COMPLET
# ─────────────────────────────────────────────
class FastBioTextTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.plasticity = FastHebbianSTDP(
            eta_hebb=cfg.eta_hebb, eta_stdp=cfg.eta_stdp,
            tau=cfg.stdp_tau, decay=cfg.decay_w, max_ratio=cfg.hebb_max_ratio,
        )
        self.embed   = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.seq_len,    cfg.d_model)
        self.layers  = nn.ModuleList([
            FastBioLayer(cfg.d_model, cfg.n_heads, cfg.d_ff, self.plasticity, i)
            for i in range(cfg.n_layers)
        ])
        self.ln_out = nn.LayerNorm(cfg.d_model)
        self.head   = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.head.weight = self.embed.weight       # poids liés

        # Init GPT-style : N(0, 0.02²). Avec les poids liés, les logits initiaux
        # ont un écart-type ≈ 0.16 → loss initial ≈ log(256) ≈ 5.55.
        nn.init.normal_(self.embed.weight,   std=0.02)
        nn.init.normal_(self.pos_emb.weight, std=0.02)

        self.replay = deque(maxlen=cfg.replay_cap)

    def forward(self, ids):
        B, T = ids.shape
        pos  = torch.arange(T, device=ids.device).unsqueeze(0)
        x    = self.embed(ids) + self.pos_emb(pos)
        for layer in self.layers:
            x = layer(x)
        x = self.ln_out(x)
        return self.head(x)

    # ---- pertes -----------------------------------------------------------
    @staticmethod
    def lm_loss(logits, ids):
        """Cross-entropy next-token : logits[t] prédit ids[t+1]."""
        return F.cross_entropy(
            logits[:, :-1].reshape(-1, logits.shape[-1]).float(),
            ids[:, 1:].reshape(-1),
        )

    # ---- replay -----------------------------------------------------------
    def push_to_replay(self, ids_batch):
        for seq in ids_batch:
            self.replay.append(seq.detach().cpu())

    @torch.no_grad()
    def eval_replay_loss(self, n=32):
        """Loss moyen sur un échantillon fixe du replay (mesure de consolidation)."""
        if len(self.replay) == 0:
            return float("nan")
        was_training = self.training
        self.eval()
        rng = random.Random(0)                       # même échantillon avant/après
        sample = rng.sample(list(self.replay), min(n, len(self.replay)))
        ids = torch.stack(sample).to(cfg.device)
        loss = self.lm_loss(self(ids), ids).item()
        if was_training:
            self.train()
        return loss

    # ---- sommeil ----------------------------------------------------------
    def sleep(self, optimizer, scaler=None):
        """Phase sommeil : replay + consolidation + élagage/croissance."""
        if not cfg.use_sleep or len(self.replay) < cfg.batch_size:
            return
        print("  ◌ Phase sommeil (consolidation)...")
        loss_before = self.eval_replay_loss()

        # lr réduit pendant le sommeil (consolidation douce)
        base_lrs = [g["lr"] for g in optimizer.param_groups]
        for g in optimizer.param_groups:
            g["lr"] = g["lr"] * cfg.sleep_lr_mult

        self.train()
        for _ in range(cfg.sleep_steps):
            batch = random.sample(list(self.replay), min(cfg.batch_size, len(self.replay)))
            ids   = torch.stack(batch).to(cfg.device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            if scaler is not None:
                with torch.amp.autocast("cuda", dtype=cfg.amp_dtype):
                    loss = self.lm_loss(self(ids), ids)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss = self.lm_loss(self(ids), ids)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        for g, lr0 in zip(optimizer.param_groups, base_lrs):
            g["lr"] = lr0

        # Plasticité structurelle
        for i, layer in enumerate(self.layers):
            n_before = layer.dcm.n_active
            n_pruned, n_grown = layer.dcm.prune_and_grow(optimizer)
            print(f"    Couche {i} : {n_before} arêtes → élaguées {n_pruned}, "
                  f"créées {n_grown} → {layer.dcm.n_active} actives")

        loss_after = self.eval_replay_loss()
        print(f"    Loss replay : {loss_before:.4f} → {loss_after:.4f} "
              f"({'↓' if loss_after < loss_before else '↑'})")

    # ---- génération -------------------------------------------------------
    @torch.no_grad()
    def generate(self, prompt, length=None, temperature=None, top_k=None,
                 rep_penalty=None, greedy=False):
        """Génère du texte caractère par caractère (fenêtre glissante seq_len)."""
        length      = length or cfg.gen_len
        temperature = cfg.gen_temperature if temperature is None else temperature
        top_k       = cfg.gen_topk if top_k is None else top_k
        rep_penalty = cfg.gen_rep_penalty if rep_penalty is None else rep_penalty

        self.eval()
        # Encodage UTF-8 (cohérent avec l'entraînement : accents = plusieurs octets)
        ids = torch.tensor([text_to_ids(prompt)], device=cfg.device)
        out_ids = ids[0].tolist()
        for _ in range(length):
            ctx = ids[:, -cfg.seq_len:]
            logits = self(ctx)[0, -1].float()
            # Pénalité de répétition sur les 8 derniers caractères (anti "uuuuu")
            if rep_penalty > 1.0:
                recent = torch.tensor(out_ids[-8:], device=logits.device)
                logits[recent] = torch.where(
                    logits[recent] > 0, logits[recent] / rep_penalty, logits[recent] * rep_penalty
                )
            if greedy or temperature <= 0:
                next_id = int(logits.argmax())
            else:
                logits = logits / temperature
                if top_k > 0:
                    v, _ = torch.topk(logits, min(top_k, logits.numel()))
                    logits[logits < v[-1]] = -float("inf")
                probs = F.softmax(logits, dim=-1)
                next_id = int(torch.multinomial(probs, 1))
            out_ids.append(next_id)
            ids = torch.cat([ids, torch.tensor([[next_id]], device=cfg.device)], dim=1)
        return ids_to_text(out_ids)

    # ---- sauvegarde / chargement -----------------------------------------
    def save(self, path=None):
        path = Path(path or cfg.save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model": self.state_dict(),
            "hebbian_H": self.plasticity.state_dict(),
            "config": {k: v for k, v in vars(Config).items()
                       if not k.startswith("_") and isinstance(v, (int, float, str, bool))},
        }, path)
        return path

    @classmethod
    def load(cls, path=None, device=None):
        device = device or cfg.device
        ckpt = torch.load(Path(path or cfg.save_path), map_location=device, weights_only=False)
        model = cls().to(device)
        model.load_state_dict(ckpt["model"])
        model.plasticity.load_state_dict(ckpt.get("hebbian_H", {}), device)
        return model


# ─────────────────────────────────────────────
# 8. DONNÉES — Texte français custom
# ─────────────────────────────────────────────
def load_sample_text():
    """Texte court en français (à remplacer par un corpus plus grand)."""
    text = (
        "Le cerveau contient environ quatre-vingt-six milliards de neurones. "
        "Chaque neurone est connecté à d'autres par des synapses dynamiques. "
        "La plasticité synaptique permet l'apprentissage continu. "
        "Les règles hebbienne et STDP gouvernent le changement synaptique. "
        "Durant le sommeil, la consolidation mémoire rejoue les expériences. "
        "L'architecture du cerveau est parcellisée en couches hiérarchiques. "
        "Les spikes sont les signaux événementiels du système nerveux. "
        "Le sparse coding économise l'énergie cérébrale et booste la performance. "
        "La plasticité structurelle crée et élimine les connexions synaptiques. "
        "Un PC gamer moderne simule ces phénomènes à petite échelle efficacement. "
    )
    return text * 15


def text_to_ids(text):
    """Encodage octet par octet (UTF-8) → vocab 256, réversible (accents OK)."""
    return list(text.encode("utf-8"))


def ids_to_text(ids):
    return bytes(ids).decode("utf-8", errors="replace")


def make_batch(ids_list, batch_size, seq_len, device):
    n = len(ids_list) - seq_len - 1
    starts = [random.randint(0, max(0, n)) for _ in range(batch_size)]
    batch  = torch.tensor([ids_list[s: s + seq_len] for s in starts], dtype=torch.long)
    if cfg.pin_memory:
        batch = batch.pin_memory()
    return batch.to(device, non_blocking=True)


# ─────────────────────────────────────────────
# 9. BOUCLE D'ENTRAÎNEMENT OPTIMISÉE
# ─────────────────────────────────────────────
def train(text=None, verbose=True, model=None):
    """Boucle d'entraînement. `model` permet de reprendre depuis des poids chargés."""
    setup_backend()
    if verbose:
        cfg.print_config()

    fresh = model is None
    if fresh:
        model = FastBioTextTransformer()
    model = model.to(cfg.device)
    # Deux groupes de paramètres : les logits de connectivité (gates) reçoivent
    # un lr plus élevé et pas de weight decay → dynamique structurelle visible.
    gate_params  = [layer.dcm.logits for layer in model.layers]
    gate_ids     = {id(p) for p in gate_params}
    other_params = [p for p in model.parameters() if id(p) not in gate_ids]
    optimizer = torch.optim.AdamW(
        [
            {"params": other_params, "lr": cfg.lr, "weight_decay": cfg.weight_decay},
            {"params": gate_params,  "lr": cfg.lr * cfg.gate_lr_mult, "weight_decay": 0.0},
        ],
        betas=(0.9, 0.95),
    )
    scaler = torch.amp.GradScaler("cuda") if cfg.mixed_precision else None

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    text = text or load_sample_text()
    ids_all = text_to_ids(text)
    if verbose:
        print(f"\n  Paramètres : {n_params:,}")
        print(f"  Texte d'entraînement : {len(text):,} caractères ({len(ids_all):,} octets)")
        n_edges = [layer.dcm.n_active for layer in model.layers]
        print(f"  Arêtes initiales par couche : {n_edges}")
        print(f"  Loss de référence (aléatoire) : log({cfg.vocab_size}) = {math.log(cfg.vocab_size):.3f}\n")

    history = {"step": [], "loss": [], "edges": []}
    running = []
    t0 = time.time()

    for step in range(1, cfg.train_steps + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        step_loss = 0.0

        # Accumulation de gradients : grad_accum micro-batches par étape
        for _ in range(cfg.grad_accum):
            ids = make_batch(ids_all, cfg.batch_size, cfg.seq_len, cfg.device)
            if scaler is not None:
                with torch.amp.autocast("cuda", dtype=cfg.amp_dtype):
                    loss = model.lm_loss(model(ids), ids) / cfg.grad_accum
                scaler.scale(loss).backward()
            else:
                loss = model.lm_loss(model(ids), ids) / cfg.grad_accum
                loss.backward()
            step_loss += loss.item()
            model.push_to_replay(ids)

        if scaler is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        running.append(step_loss)
        if step == 1 and verbose:
            hint = f" (attendu ≈ log({cfg.vocab_size}) = {math.log(cfg.vocab_size):.2f})" if fresh else " (reprise)"
            print(f"  Étape   1/{cfg.train_steps} | Loss={step_loss:.4f}{hint}")

        if step % 10 == 0:
            mean_loss = sum(running) / len(running)
            running = []
            edges = [layer.dcm.n_active for layer in model.layers]
            gates = [f"{layer.dcm.active_gates().mean().item():.2f}" for layer in model.layers]
            history["step"].append(step)
            history["loss"].append(mean_loss)
            history["edges"].append(edges)
            if verbose:
                print(
                    f"  Étape {step:3d}/{cfg.train_steps} | "
                    f"Loss={mean_loss:.4f} | "
                    f"Arêtes={edges} | gate moy.={gates} | "
                    f"{(time.time() - t0) / step * 1000:.0f} ms/étape"
                )

        if cfg.use_sleep and step % cfg.sleep_every == 0:
            if verbose:
                print(f"\n  ─────── Sommeil (étape {step}) ───────")
            model.sleep(optimizer, scaler)
            if verbose:
                print()

    elapsed = time.time() - t0
    if verbose:
        print("\n" + "=" * 60)
        print(f"  ✓ Entraînement terminé en {elapsed:.1f} s "
              f"({elapsed / cfg.train_steps * 1000:.0f} ms/étape)")
        print("=" * 60)

    # ── Génération ──────────────────────────────────────────
    if verbose:
        prompt = "Le cerveau"
        print("\n  Génération (greedy + pénalité de répétition) :")
        print("  " + model.generate(prompt, greedy=True).replace("\n", " "))
        print(f"\n  Génération (sampling T={cfg.gen_temperature}, top-k={cfg.gen_topk}) :")
        print("  " + model.generate(prompt).replace("\n", " "))
        print()

        print("  Statistiques finales :")
        for i, layer in enumerate(model.layers):
            dcm = layer.dcm
            g = dcm.active_gates()
            print(f"    Couche {i} | arêtes actives={dcm.n_active}/{cfg.max_edges} | "
                  f"gates>0.5={(g > 0.5).sum().item()} | "
                  f"gates min/moy/max={g.min().item():.2f}/{g.mean().item():.2f}/{g.max().item():.2f} | "
                  f"sparsité attn={layer.attn.gate.last_active_frac:.1%} "
                  f"ffn={layer.ffn.gate.last_active_frac:.1%}")
        if model.plasticity.H:
            hn = {k: f"{v.norm().item():.3f}" for k, v in model.plasticity.H.items()}
            print(f"    ||H|| hebbien par matrice : {hn}")

    model.history = history
    return model


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("\n")
    model = train()
    print("\n  ✓ Modèle prêt ! Tu peux le sauver ou l'affiner.\n")

    path = model.save(cfg.save_path)
    size_mb = path.stat().st_size / 1e6
    print(f"  Poids sauvegardés dans : {path.resolve()} ({size_mb:.2f} Mo)\n")
