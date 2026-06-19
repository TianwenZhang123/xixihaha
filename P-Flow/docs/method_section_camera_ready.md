# P-Flow: Reference-Guided Video Generation via Orthogonal Three-Layer Decomposition

## 3. Method

Given a reference video $V_{\text{ref}} \in \mathbb{R}^{T \times H \times W \times 3}$ and its textual description $c$, our goal is to guide a pretrained text-to-video (T2V) diffusion model to generate a video $V_{\text{gen}}$ that preserves the appearance and motion patterns of $V_{\text{ref}}$, without modifying any model parameters (zero-training). We formalize this as manipulating three orthogonal input interfaces of the T2V model: the text condition space (Sec. 3.2), the noise space (Sec. 3.3), and the intermediate feature space (Sec. 3.4). We first introduce the preliminary in Sec. 3.1, then detail each layer, and finally present the unified formulation in Sec. 3.5.

---

### 3.1 Preliminaries

#### 3.1.1 Flow Matching for Video Generation

Our backbone model, Wan2.1-T2V-1.3B, is a Diffusion Transformer (DiT) trained with the Flow Matching objective. Unlike DDPM-based models that define a discrete noise schedule, Flow Matching constructs a continuous probability path via linear interpolation between noise and data:

$$
x_t = (1 - t) \cdot \varepsilon + t \cdot x_1, \quad t \in [0, 1], \quad \varepsilon \sim \mathcal{N}(0, I)
\tag{1}
$$

where $x_1$ denotes a sample from the data distribution $p_{\text{data}}$ and $\varepsilon$ is standard Gaussian noise. The model $v_\theta$ is trained to predict the velocity field:

$$
v_\theta(x_t, t, c) \approx x_1 - \varepsilon = \frac{x_1 - x_t}{1 - t}
\tag{2}
$$

Generation proceeds by solving the ODE from $t = 0$ (pure noise) to $t = 1$ (data) using Euler integration:

$$
x_{t + \Delta t} = x_t + \Delta t \cdot v_\theta(x_t, t, c), \quad \Delta t = \frac{1}{N}
\tag{3}
$$

where $N = 30$ is the number of inference steps and $c$ denotes the text condition encoded by a frozen UMT5 text encoder.

#### 3.1.2 DiT Architecture and Input Interfaces

The Wan2.1 DiT consists of $L = 30$ transformer blocks, each containing self-attention, cross-attention (conditioned on UMT5 embeddings), and feed-forward layers. We identify three non-invasive interfaces through which reference information can be injected:

- **Text interface**: The prompt string $c$, encoded by UMT5 into token embeddings $E(c) \in \mathbb{R}^{S \times d}$ that serve as keys/values in cross-attention.
- **Noise interface**: The initial latent $z_T \sim \mathcal{N}(0, I)$ that serves as the ODE starting point.
- **Feature interface**: Intermediate activations $h^{(l,t)}$ at layer $l$ and timestep $t$, accessible via PyTorch `register_forward_hook`.

These three interfaces operate in distinct vector spaces—text embedding space, noise (latent) space, and feature (activation) space—ensuring **orthogonal** operation without direct interference.

#### 3.1.3 Latent Representation

The reference video $V_{\text{ref}} \in \mathbb{R}^{1 \times 3 \times T \times H \times W}$ (with $T = 81$, $H = 480$, $W = 832$) is first encoded into a compact latent representation by a pretrained 3D-VAE:

$$
z_1 = \mathcal{E}(V_{\text{ref}}) \in \mathbb{R}^{1 \times C \times F \times H' \times W'}
\tag{4}
$$

where $C = 16$ is the latent channel dimension, and the VAE applies $4\times$ temporal and $8\times$ spatial downsampling, yielding $F = 21$, $H' = 60$, $W' = 104$.

---

### 3.2 Layer 1: Minimal Prompt Rewriting

#### 3.2.1 Motivation: U-Shaped Positional Attention Distribution

The text condition enters the DiT via cross-attention layers. A natural question arises: **which token positions in the prompt contribute most to generation quality?** We answer this empirically by instrumenting the `WanAttnProcessor` with attention weight extraction hooks.

**Observation (U-Shaped Distribution).** Let $\mathbf{A} = \text{softmax}(QK^\top / \sqrt{d}) \in \mathbb{R}^{N_v \times S}$ denote the cross-attention weight matrix, where $N_v$ is the number of visual tokens and $S$ is the prompt sequence length. We compute the mean attention received by each text position $j$:

$$
\bar{a}_j = \frac{1}{N_v} \sum_{i=1}^{N_v} A_{ij}
\tag{5}
$$

Across all layers and timesteps, we observe a consistent **U-shaped distribution**:

$$
\bar{a}_0 \approx \bar{a}_{S-1} \approx 0.029, \quad \bar{a}_j \approx 0.001 \;\; \forall j \in \{1, \ldots, S-2\}
\tag{6}
$$

That is, the first and last token positions receive approximately $10$–$15\times$ more attention than interior positions, with an immediate $96\%$ drop from position 0 to position 1.

**Root Cause.** This phenomenon originates from the UMT5 encoder's relative position bias (`relative_attention_num_buckets=32`, `relative_attention_max_distance=128`), which induces boundary-favoring statistics in the output embeddings. The DiT's cross-attention inherits these positional biases directly.

**Design Implication.** Only the **head** (subject noun) and **tail** (vivid keyword) of the prompt exert significant influence on generation; interior tokens have marginal contribution.

#### 3.2.2 Head-Tail Keyword Replacement Strategy

Based on the U-shaped finding, we design a minimal editing strategy that modifies only the high-attention boundary positions while preserving all interior content:

**Step 1: LLM Head-Tail Replacement.** Given the original VLM-generated caption $c_{\text{raw}}$, we apply the following rules:

- If the head contains uninformative prefixes (e.g., "The video depicts/shows/features..."), replace with the subject noun phrase.
- If the tail contains generic summary sentences (e.g., "The overall atmosphere/mood..."), replace with 1–3 visual/motion keywords already present in the interior.
- Interior content is preserved verbatim (100% retention).

**Step 2: VLM Factual Correction.** A vision-language model (Qwen2.5-VL-7B) views the video frames and corrects at most 3 factual errors (incorrect color, object count, misidentified objects) via single-word substitutions.

The combined edit ratio is bounded by:

$$
r_{\text{edit}} = \frac{|\text{changed tokens}|}{|\text{total tokens}|} \leq 8\%
\tag{7}
$$

This constraint ensures that the UMT5 encoding of the rewritten prompt remains close to the original:

$$
\cos\big(E(c_{\text{rewrite}}),\; E(c_{\text{raw}})\big) \geq 0.95
\tag{8}
$$

which is critical for maintaining compatibility with Layer 3 (Sec. 3.4), where reference features are cached using the original prompt.

#### 3.2.3 Avoiding Semantic Conflict with Layer 2

A crucial design principle of Layer 1 is that **it must not modify motion-related descriptions**. The rationale is as follows: Layer 2 (SVD Noise Prior) provides a motion direction bias extracted from the reference video's temporal structure. If the prompt explicitly specifies a different motion direction (e.g., "dolly-in" when SVD encodes "pan-left"), the two signals produce contradictory constraints on the ODE trajectory, causing the model to generate confused motion patterns.

Empirically, our earlier v7e strategy (which added precise motion descriptions) combined with SVD resulted in X-CLIP dropping by $-4.1\%$ compared to baseline—confirming that explicit motion in the prompt conflicts with the SVD prior. Our head-tail replacement strategy deliberately operates only on boundary positions with non-motion semantics, delegating motion guidance entirely to Layer 2.

---

### 3.3 Layer 2: SVD-Based Motion Prior Injection

#### 3.3.1 Overview

The core idea of Layer 2 is to encode the reference video's temporal dynamics into a structured initial noise $z_T$ that biases the ODE trajectory toward generating similar motion patterns. This is achieved through three steps: (1) Flow Matching Inversion to map $V_{\text{ref}}$ back to noise space, (2) SVD filtering to isolate temporal (motion) components from spatial (appearance) components, and (3) controlled blending of the motion signal into the random initial noise.

#### 3.3.2 Flow Matching Inversion

To obtain the noise-space representation of the reference video, we solve the ODE in the reverse direction, from $t = 1$ (data) to $t = 0$ (noise):

$$
z_{t - \Delta t} = z_t - \Delta t \cdot v_\theta(z_t, t, c), \quad \Delta t = \frac{1}{N_{\text{inv}}}
\tag{9}
$$

with $z_1 = \mathcal{E}(V_{\text{ref}})$ as the initial condition and $N_{\text{inv}} = 50$ inversion steps. The guidance scale is set to $1.0$ (no classifier-free guidance) to avoid introducing CFG-induced bias into the inverted noise. The final output $\eta_{\text{inv}} = z_0$ represents the reference video's encoding in the noise space.

For higher reconstruction accuracy, we optionally employ a second-order midpoint method:

$$
k_1 = v_\theta(z_t, t, c)
\tag{10a}
$$
$$
z_{\text{mid}} = z_t - \frac{\Delta t}{2} \cdot k_1
\tag{10b}
$$
$$
k_2 = v_\theta\left(z_{\text{mid}},\; t - \frac{\Delta t}{2},\; c\right)
\tag{10c}
$$
$$
z_{t - \Delta t} = z_t - \Delta t \cdot k_2
\tag{10d}
$$

This reduces the truncation error from $\mathcal{O}(\Delta t^2)$ to $\mathcal{O}(\Delta t^3)$ at the cost of doubling the model evaluations per step.

#### 3.3.3 Two-Stage SVD Filtering

The inverted noise $\eta_{\text{inv}} \in \mathbb{R}^{C \times F \times H' \times W'}$ simultaneously encodes both appearance information (what objects look like) and motion information (how they move). We design a two-stage SVD decomposition to disentangle these components.

**Stage 1: Spatial Decontenting.** We remove the dominant spatial (appearance) components by performing SVD along the spatial dimensions:

$$
\eta_{\text{inv}} \xrightarrow{\text{reshape}} M_s \in \mathbb{R}^{(C \cdot F) \times (H' \cdot W')}
\tag{11}
$$
$$
M_s = U_s \Sigma_s V_s^\top
\tag{12}
$$

We determine the minimal rank $k_s$ such that removing the top-$k_s$ singular vectors retains at least $\rho_s$ fraction of total energy:

$$
k_s = \min\left\{k : \frac{\sum_{i=k+1}^{r} \sigma_{s,i}^2}{\sum_{i=1}^{r} \sigma_{s,i}^2} \geq \rho_s \right\}
\tag{13}
$$

The filtered noise and the spatial residual are:

$$
\eta_{\text{filtered}} = M_s - \sum_{i=1}^{k_s} \sigma_{s,i} \cdot \mathbf{u}_{s,i} \mathbf{v}_{s,i}^\top
\tag{14}
$$
$$
\eta_{\text{spatial}} = \sum_{i=1}^{k_s} \sigma_{s,i} \cdot \mathbf{u}_{s,i} \mathbf{v}_{s,i}^\top
\tag{15}
$$

**Intuition.** Appearance information (static scene content, object textures) is highly correlated across the temporal dimension, manifesting as the dominant spatial singular vectors. Removing these leaves primarily the inter-frame variations—the motion signal.

**Stage 2: Temporal Retention.** We further concentrate the remaining signal onto the dominant temporal modes by performing SVD along the temporal dimension:

$$
\eta_{\text{filtered}} \xrightarrow{\text{reshape}} M_m \in \mathbb{R}^{(C \cdot H' \cdot W') \times F}
\tag{16}
$$
$$
M_m = U_m \Sigma_m V_m^\top
\tag{17}
$$

We retain the top-$k_m$ temporal singular vectors capturing $\rho_m$ fraction of temporal energy:

$$
k_m = \min\left\{k : \frac{\sum_{i=1}^{k} \sigma_{m,i}^2}{\sum_{i=1}^{r} \sigma_{m,i}^2} \geq \rho_m \right\}
\tag{18}
$$
$$
\eta_{\text{temporal}} = \sum_{i=1}^{k_m} \sigma_{m,i} \cdot \mathbf{u}_{m,i} \mathbf{v}_{m,i}^\top
\tag{19}
$$

**Intuition.** The dominant temporal modes capture the principal directions of frame-to-frame variation—the global motion patterns. Minor temporal components correspond to noise or localized jitter. Retaining only the top modes yields a clean motion prior.

**Default parameters:** $\rho_s = 0.1$, $\rho_m = 0.9$. For computational efficiency, Stage 1 uses randomized SVD (`torch.svd_lowrank` with rank $q = \min(\min(CF, H'W'), \max(50, 0.3 \cdot \min(CF, H'W')))$) since the spatial dimensions are large; Stage 2 uses full SVD since $F = 21$ is small.

#### 3.3.4 Noise Blending

The extracted motion prior is blended with random noise to construct the structured initial latent:

**Two-component blending** (motion only):

$$
z_T = \sqrt{\alpha} \cdot \eta_{\text{temporal}} + \sqrt{1 - \alpha} \cdot \eta_{\text{random}}, \quad \eta_{\text{random}} \sim \mathcal{N}(0, I)
\tag{20}
$$

**Three-component blending** (motion + appearance):

$$
z_T = \sqrt{\alpha} \cdot \eta_{\text{temporal}} + \sqrt{\beta} \cdot \hat{\eta}_{\text{spatial}} + \sqrt{1 - \alpha - \beta} \cdot \eta_{\text{random}}
\tag{21}
$$

where the spatial component requires **magnitude matching** (renormalization) to ensure that $\alpha$ and $\beta$ have comparable semantic significance:

$$
\hat{\eta}_{\text{spatial}} = \eta_{\text{spatial}} \cdot \frac{\text{std}(\eta_{\text{temporal}})}{\text{std}(\eta_{\text{spatial}})}
\tag{22}
$$

**Motivation for renormalization.** Empirically, $\text{std}(\eta_{\text{spatial}}) \approx 0.9\text{--}1.2$ while $\text{std}(\eta_{\text{temporal}}) \approx 0.28\text{--}0.41$. Without renormalization, the spatial component at coefficient $\sqrt{\beta}$ would inject $\sqrt{\beta} \times 1.0 \approx 0.032$ energy while the temporal component injects only $\sqrt{\alpha} \times 0.35 \approx 0.022$ energy—the appearance signal would dominate despite having a smaller coefficient.

**Critical design constraint.** The blending coefficient must be extremely small: $\alpha = 0.003$, yielding $\sqrt{\alpha} \approx 0.055$. The actual energy injected is approximately $0.055 \times 0.35 \approx 0.019$, which is $<2\%$ of the unit-variance random noise. Extensive ablation ($>10$ configurations including frequency-domain reshaping, channel-concentrated injection, multi-scale decomposition, phase interpolation, and SGA adaptation) confirms that **$\alpha \approx 0.003$ is the absolute ceiling for black-box injection**. The reason is fundamental: Flow Matching models are trained under the strict assumption $z_T \sim \mathcal{N}(0, I)$; any structured deviation from this distribution causes the ODE trajectory to depart from the learned data manifold, degrading generation quality.

#### 3.3.5 Temporal Signal Reliability (TSR)

Different videos exhibit vastly different motion characteristics: object-motion videos (e.g., animals running) are insensitive to $\alpha$, while scene-motion videos (e.g., slow indoor camera pans) are extremely sensitive ($\alpha$ increasing from 0.001 to 0.002 causes X-CLIP to drop by 0.075). We propose TSR for per-sample adaptive $\alpha$ scaling.

TSR is composed of two complementary signals:

**Temporal Concentration Ratio (TCR).** Measures how much of the temporal energy is concentrated in the first singular mode:

$$
\text{TCR} = \frac{\sigma_{m,1}^2}{\sum_{i=1}^{r} \sigma_{m,i}^2}
\tag{23}
$$

High TCR indicates a single dominant motion direction (reliable signal); low TCR indicates distributed, potentially noisy temporal structure.

**Temporal Autocorrelation (TAC).** Measures the frame-to-frame consistency of the temporal signal:

$$
\text{TAC} = \frac{1}{F-1} \sum_{i=1}^{F-1} \frac{\langle \eta_{\text{temporal}}^{(i)}, \eta_{\text{temporal}}^{(i+1)} \rangle}{\|\eta_{\text{temporal}}^{(i)}\| \cdot \|\eta_{\text{temporal}}^{(i+1)}\|}
\tag{24}
$$

where $\eta_{\text{temporal}}^{(i)} \in \mathbb{R}^{C \cdot H' \cdot W'}$ is the $i$-th frame's flattened temporal component.

**TSR Computation.** We normalize both signals and combine them multiplicatively:

$$
\text{TCR}_{\text{norm}} = \sigma\big(s_{\text{tcr}} \cdot (\text{TCR} - \mu_{\text{tcr}})\big)
\tag{25}
$$
$$
\text{TAC}_{\text{norm}} = \max(0, \text{TAC})
\tag{26}
$$
$$
\text{TSR} = \text{TCR}_{\text{norm}} \times \text{TAC}_{\text{norm}}
\tag{27}
$$

where $\sigma(\cdot)$ is the sigmoid function, $s_{\text{tcr}} = 10.0$ (slope), and $\mu_{\text{tcr}} = 0.1$ (center).

**Adaptive $\alpha$.** The per-sample blending coefficient is:

$$
\alpha_{\text{adaptive}} = \alpha_{\min} + \text{TSR} \cdot (\alpha_{\max} - \alpha_{\min})
\tag{28}
$$

with $\alpha_{\min} = 0.0$ and $\alpha_{\max} = 0.003$, subject to a floor constraint $\alpha_{\text{eff}} \geq \alpha_{\text{floor}} = 0.001$.

**Semantics.** TSR is high when the temporal signal is both concentrated (clear dominant direction) and coherent (consistent across frames)—indicating a reliable motion prior that can tolerate stronger injection. TSR is low when motion is weak, distributed, or incoherent—signaling that injection should be minimized to avoid introducing noise.

#### 3.3.6 On the Harmfulness of Renormalization for $\eta_{\text{temporal}}$

A natural preprocessing step would be to renormalize $\eta_{\text{temporal}}$ to unit variance before blending. However, this destroys an important **implicit adaptivity**: different samples produce $\eta_{\text{temporal}}$ with naturally varying standard deviations ($0.28$–$0.41$). Samples with weak motion signals have smaller std (appropriately receiving less injection), while samples with strong motion have larger std. Renormalization forces equal injection energy across all samples, resulting in over-injection for weak-motion samples. Empirically, renormalization degrades X-CLIP by $-4.7\%$.

---

### 3.4 Layer 3: Adaptive Feature Injection

#### 3.4.1 Motivation

Layer 2 can only influence the ODE **starting point** $z_T$, and the injected information is progressively diluted over $N = 30$ integration steps as the model's generation prior takes over. Ablation confirms $\alpha = 0.003$ is the absolute ceiling—increasing it degrades quality rather than enhancing guidance. Feature Injection circumvents this limitation by operating **inside** the ODE trajectory, modifying the DiT's intermediate representations at every denoising step.

Conceptually, if Layer 2 determines "where the ODE starts," Layer 3 determines "how the ODE evolves"—providing per-step semantic constraints that steer the generation toward the reference video's appearance and structure, analogous to a zero-training ControlNet but without any auxiliary network.

#### 3.4.2 Reference Feature Caching

During Flow Matching Inversion (Sec. 3.3.2), we simultaneously cache the DiT's intermediate activations via forward hooks registered on target layers $\mathcal{L}_{\text{target}}$:

$$
h_{\text{ref}}^{(l, t)} = \text{CrossAttn}_l\big(z_t, t, E(c)\big), \quad l \in \mathcal{L}_{\text{target}},\; t \in \{t_0, \ldots, t_{N-1}\}
\tag{29}
$$

where $\mathcal{L}_{\text{target}} = \{10, 11, \ldots, 19\}$ (the middle 10 layers of the 30-layer DiT). We cache the **cross-attention output** specifically, as it represents the text-conditioned semantic features most relevant to content and motion guidance.

**Cache indexing.** Since inversion uses $N_{\text{inv}} = 50$ steps while generation uses $N_{\text{gen}} = 30$ steps, we map generation step $i$ to the nearest inversion timestep:

$$
t_{\text{gen}}^{(i)} = \frac{i + 1}{N_{\text{gen}}}, \quad h_{\text{ref}}^{(l, i)} = h_{\text{ref}}^{(l,\; \text{nearest}(t_{\text{gen}}^{(i)}))}
\tag{30}
$$

#### 3.4.3 Injection with Adaptive Gating

At each generation step $i$ and target layer $l$, the current generation activations $h_{\text{gen}}^{(l,i)}$ are blended with the cached reference features:

$$
h_{\text{out}}^{(l,i)} = (1 - \lambda_{\text{eff}}^{(l,i)}) \cdot h_{\text{gen}}^{(l,i)} + \lambda_{\text{eff}}^{(l,i)} \cdot \tilde{h}_{\text{ref}}^{(l,i)}
\tag{31}
$$

where $\tilde{h}_{\text{ref}}$ denotes EMA-smoothed reference features (Sec. 3.4.6) and $\lambda_{\text{eff}}$ is the effective injection strength computed through three modulation stages:

**Stage (a): Temporal Schedule.** The base injection strength varies over denoising steps following a sinusoidal (middle-peak) profile:

$$
\lambda_{\text{base}}(i) = \lambda_{\max} \cdot \sin\left(\frac{\pi \cdot i}{N - 1}\right)
\tag{32}
$$

**Rationale.** Early steps ($i \approx 0$) determine global structure; late steps ($i \approx N$) refine high-frequency details. Both extremes benefit from more model freedom. The middle steps, which determine semantic content and motion patterns, benefit most from reference guidance.

**Stage (b): Cosine Similarity Gate.** We suppress injection when generation features already align well with reference features:

$$
s^{(l,i)} = \frac{\langle h_{\text{gen}}^{(l,i)},\; \tilde{h}_{\text{ref}}^{(l,i)} \rangle}{\|h_{\text{gen}}^{(l,i)}\| \cdot \|\tilde{h}_{\text{ref}}^{(l,i)}\|}
\tag{33}
$$
$$
g^{(l,i)} = 1 - \sigma\big(\tau \cdot (s^{(l,i)} - 0.5)\big)
\tag{34}
$$

where $\tau = 5.0$ is the temperature. When similarity is high ($s > 0.5$), the gate closes ($g \to 0$), preventing unnecessary over-constraint. When similarity is low ($s < 0.5$), the gate opens ($g \to 1$), applying corrective injection.

**Stage (c): Quality Scale.** Not all reference videos provide reliable guidance. Videos with incoherent motion (e.g., noise-dominated scenes) produce unreliable cached features. We compute a per-sample quality score based on the temporal coherence of $\eta_{\text{temporal}}$:

$$
\bar{s}_{\text{frame}} = \frac{1}{F-1} \sum_{i=1}^{F-1} \cos\big(\eta_{\text{temporal}}^{(i)},\; \eta_{\text{temporal}}^{(i+1)}\big)
\tag{35}
$$
$$
\text{QS} = 0.1 + 0.9 \cdot \sigma\big(20 \cdot (\bar{s}_{\text{frame}} - \theta)\big)
\tag{36}
$$

where $\theta = 0.05$. When motion coherence is high, $\text{QS} \approx 1.0$; when motion is incoherent, $\text{QS} \approx 0.1$, significantly suppressing injection.

**Combined effective strength:**

$$
\lambda_{\text{eff}}^{(l,i)} = \lambda_{\text{base}}(i) \cdot g^{(l,i)} \cdot \text{QS}
\tag{37}
$$

#### 3.4.4 Layer Selection Strategy

The 30 DiT transformer blocks encode information at different levels of abstraction:

- **Early layers (0–9):** Low-level spatial features, edges, textures.
- **Middle layers (10–19):** Semantic structure, object identity, motion patterns.
- **Late layers (20–29):** High-frequency details, fine textures.

We inject into middle layers ($\mathcal{L}_{\text{target}} = \{10, \ldots, 19\}$) because they encode the semantic-level information most relevant to preserving the reference video's identity and motion, while allowing the model freedom in early/late layers to handle low-level reconstruction and fine details autonomously.

#### 3.4.5 Alternative Temporal Schedules

Beyond the default sinusoidal schedule, we support:

- **Constant:** $\lambda_{\text{base}}(i) = \lambda_{\max}$
- **Warmup-decay:** Linear warmup over the first 20% of steps ($\lambda_{\text{base}} = \lambda_{\max} \cdot (0.5 + 0.5 \cdot p)$ where $p = i / (0.2N)$), followed by cosine decay.
- **Cosine decay:** $\lambda_{\text{base}}(i) = \lambda_{\max} \cdot \cos\left(\frac{\pi}{2} \cdot \frac{i}{N-1}\right)$

Ablation shows that `middle_peak` performs best overall, as it concentrates injection on the semantically critical middle denoising steps.

#### 3.4.6 EMA Feature Smoothing

The cached reference features may exhibit temporal discontinuities due to numerical errors in the inversion process or the discrete timestep mismatch between inversion and generation. We apply exponential moving average (EMA) smoothing across denoising steps:

$$
\tilde{h}_{\text{ref}}^{(l, i)} = \gamma \cdot \tilde{h}_{\text{ref}}^{(l, i-1)} + (1 - \gamma) \cdot h_{\text{ref}}^{(l, i)}
\tag{38}
$$

with decay factor $\gamma = 0.7$. This suppresses abrupt feature jumps and produces smoother guidance signals.

#### 3.4.7 Relationship to ODE Dynamics

Feature Injection does not modify the ODE state variable $x_t$ directly. Instead, it alters the intermediate computation within the velocity field prediction $v_\theta$. Formally, the modified velocity at step $i$ can be written as:

$$
\tilde{v}_\theta(x_t, t, c) = v_\theta(x_t, t, c) + \Delta v^{(L3)}(x_t, t, c, \{h_{\text{ref}}\})
\tag{39}
$$

where $\Delta v^{(L3)}$ represents the implicit velocity perturbation induced by feature replacement in the middle layers. This formulation makes explicit that Layer 3 modifies the **direction** of the ODE trajectory at each step, while Layer 2 modifies the **starting point**—the two operate in complementary aspects of the ODE dynamics.

---

### 3.5 Unified Three-Layer Formulation

#### 3.5.1 Complete Generation Process

Combining all three layers, the complete P-Flow generation can be written as:

$$
V_{\text{gen}} = \mathcal{D}\left(\text{ODE-Solve}_{t=0}^{t=1}\Big(z_T^{(L2)},\; \tilde{v}_\theta(\cdot, \cdot, E(c^{(L1)}); \{h_{\text{ref}}\})\Big)\right)
\tag{40}
$$

where:

- $c^{(L1)}$ is the minimally rewritten prompt (Sec. 3.2),
- $z_T^{(L2)}$ is the SVD-structured initial noise (Sec. 3.3),
- $\tilde{v}_\theta$ is the feature-injected velocity field (Sec. 3.4),
- $\mathcal{D}$ is the VAE decoder.

#### 3.5.2 Per-Step Computation

At each denoising step $i \in \{0, 1, \ldots, N-1\}$:

1. **Compute generation features** at target layers:
$$
h_{\text{gen}}^{(l,i)} = \text{CrossAttn}_l\big(x_{t_i}, t_i, E(c^{(L1)})\big), \quad l \in \mathcal{L}_{\text{target}}
$$

2. **Compute adaptive gate**:
$$
s^{(l,i)} = \cos(h_{\text{gen}}^{(l,i)}, \tilde{h}_{\text{ref}}^{(l,i)}), \quad g^{(l,i)} = 1 - \sigma(\tau \cdot (s^{(l,i)} - 0.5))
$$

3. **Compute effective injection strength**:
$$
\lambda_{\text{eff}}^{(l,i)} = \lambda_{\max} \cdot \sin\left(\frac{\pi i}{N-1}\right) \cdot g^{(l,i)} \cdot \text{QS}
$$

4. **Inject reference features**:
$$
h_{\text{out}}^{(l,i)} = (1 - \lambda_{\text{eff}}^{(l,i)}) \cdot h_{\text{gen}}^{(l,i)} + \lambda_{\text{eff}}^{(l,i)} \cdot \tilde{h}_{\text{ref}}^{(l,i)}
$$

5. **ODE step** (using modified velocity from injected features):
$$
x_{t_{i+1}} = x_{t_i} + \Delta t \cdot \tilde{v}_\theta(x_{t_i}, t_i, c^{(L1)})
$$

6. **Initial condition** from Layer 2:
$$
x_{t_0} = z_T^{(L2)} = \sqrt{\alpha} \cdot \eta_{\text{temporal}} + \sqrt{\beta} \cdot \hat{\eta}_{\text{spatial}} + \sqrt{1-\alpha-\beta} \cdot \eta_{\text{random}}
$$

#### 3.5.3 Orthogonality and Synergy

The three layers exhibit orthogonal operation and synergistic enhancement:

**Orthogonality.** Each layer operates in a distinct mathematical space:

- L1: Text embedding space $\mathbb{R}^{S \times d}$ (modifies cross-attention keys/values)
- L2: Noise space $\mathbb{R}^{C \times F \times H' \times W'}$ (modifies ODE initial condition)
- L3: Feature space $\mathbb{R}^{B \times N_v \times d_{\text{model}}}$ (modifies intermediate activations)

There is no direct algebraic coupling between these spaces.

**Synergy.** Despite operating independently, the layers achieve super-additive performance. Individually, L2 achieves +5.1% X-CLIP and L3 achieves +7.0%. Combined, they achieve +8.6%—exceeding the sum of individual contributions, indicating positive interaction. The mechanism is:

- L2 biases the ODE starting point toward the reference motion manifold, ensuring that early denoising steps already produce features geometrically close to the reference.
- L3 leverages this proximity: when $h_{\text{gen}}$ starts closer to $h_{\text{ref}}$ (thanks to L2), the adaptive gate requires less aggressive intervention, producing smoother and more stable injection.

**Conflict avoidance.** The L1–L2 conflict (motion description vs. SVD prior) is resolved by the head-tail strategy's explicit constraint of not modifying interior motion descriptions. The L1–L3 compatibility is maintained by the $\leq 8\%$ edit ratio, ensuring the prompt embedding drift is small enough that features cached under $c_{\text{raw}}$ remain aligned with features generated under $c_{\text{rewrite}}$.

---

### 3.6 Algorithm Summary

We present the complete P-Flow algorithm in Algorithm 1.

---

**Algorithm 1:** P-Flow: Reference-Guided Video Generation

---

**Input:** Reference video $V_{\text{ref}}$, caption $c_{\text{raw}}$, model $v_\theta$, VAE encoder $\mathcal{E}$, decoder $\mathcal{D}$

**Output:** Generated video $V_{\text{gen}}$

**Hyperparameters:** $N=30$, $N_{\text{inv}}=50$, $\alpha_{\max}=0.003$, $\lambda_{\max}=0.1$, $\rho_s=0.1$, $\rho_m=0.9$, $\tau=5.0$, $\gamma=0.7$, $\theta=0.05$

---

**// Layer 1: Minimal Prompt Rewriting**

1: $c \leftarrow \text{HeadTailReplace}(c_{\text{raw}})$ $\quad$ // LLM head-tail keyword replacement

2: $c \leftarrow \text{VLMCorrect}(c, V_{\text{ref}})$ $\quad$ // VLM factual correction ($\leq 3$ edits)

**// Layer 2: SVD Noise Prior**

3: $z_1 \leftarrow \mathcal{E}(V_{\text{ref}})$ $\quad$ // VAE encode

4: $\eta_{\text{inv}} \leftarrow \text{Invert}(z_1, v_\theta, c, N_{\text{inv}})$ $\quad$ // Flow Matching Inversion (Eq. 9)

5: $\eta_{\text{filtered}}, \eta_{\text{spatial}} \leftarrow \text{SVD-Stage1}(\eta_{\text{inv}}, \rho_s)$ $\quad$ // Spatial decontenting (Eq. 11–15)

6: $\eta_{\text{temporal}} \leftarrow \text{SVD-Stage2}(\eta_{\text{filtered}}, \rho_m)$ $\quad$ // Temporal retention (Eq. 16–19)

7: $\text{TSR} \leftarrow \text{ComputeTSR}(\eta_{\text{temporal}})$ $\quad$ // Adaptive signal reliability (Eq. 23–27)

8: $\alpha \leftarrow \alpha_{\min} + \text{TSR} \cdot (\alpha_{\max} - \alpha_{\min})$ $\quad$ // Adaptive blending coefficient (Eq. 28)

9: $z_T \leftarrow \sqrt{\alpha} \cdot \eta_{\text{temporal}} + \sqrt{\beta} \cdot \hat{\eta}_{\text{spatial}} + \sqrt{1-\alpha-\beta} \cdot \eta_{\text{random}}$ $\quad$ // Noise blending (Eq. 21)

**// Layer 3: Feature Injection Preparation**

10: $\{h_{\text{ref}}^{(l,t)}\} \leftarrow \text{CacheFeatures}(z_1, v_\theta, c, \mathcal{L}_{\text{target}}, N_{\text{inv}})$ $\quad$ // Cache during inversion (Eq. 29)

11: $\text{QS} \leftarrow \text{QualityScale}(\eta_{\text{temporal}}, \theta)$ $\quad$ // Per-sample quality (Eq. 35–36)

**// Generation with Three-Layer Guidance**

12: $x_0 \leftarrow z_T$

13: **for** $i = 0$ **to** $N - 1$ **do**

14: $\quad$ $t_i \leftarrow i / N$

15: $\quad$ **for** $l \in \mathcal{L}_{\text{target}}$ **do**

16: $\quad\quad$ $h_{\text{gen}}^{(l,i)} \leftarrow \text{CrossAttn}_l(x_{t_i}, t_i, E(c))$

17: $\quad\quad$ $\tilde{h}_{\text{ref}}^{(l,i)} \leftarrow \gamma \cdot \tilde{h}_{\text{ref}}^{(l,i-1)} + (1-\gamma) \cdot h_{\text{ref}}^{(l,i)}$ $\quad$ // EMA smoothing

18: $\quad\quad$ $s \leftarrow \cos(h_{\text{gen}}^{(l,i)}, \tilde{h}_{\text{ref}}^{(l,i)})$

19: $\quad\quad$ $\lambda_{\text{eff}} \leftarrow \lambda_{\max} \cdot \sin(\pi i / (N{-}1)) \cdot [1 - \sigma(\tau(s{-}0.5))] \cdot \text{QS}$

20: $\quad\quad$ $h_{\text{out}}^{(l,i)} \leftarrow (1 - \lambda_{\text{eff}}) \cdot h_{\text{gen}}^{(l,i)} + \lambda_{\text{eff}} \cdot \tilde{h}_{\text{ref}}^{(l,i)}$

21: $\quad$ **end for**

22: $\quad$ $x_{t_{i+1}} \leftarrow x_{t_i} + \Delta t \cdot \tilde{v}_\theta(x_{t_i}, t_i, c)$

23: **end for**

24: $V_{\text{gen}} \leftarrow \mathcal{D}(x_1)$ $\quad$ // VAE decode

25: **return** $V_{\text{gen}}$

---

### 3.7 Discussion

#### 3.7.1 Comparison with Training-Based Approaches

Unlike IP-Adapter, ControlNet, or VideoComposer that require training adapter modules on large-scale paired datasets, P-Flow operates entirely at inference time by manipulating existing model interfaces. This zero-training property offers three advantages: (1) no additional training cost, (2) immediate applicability to any Flow Matching T2V model, and (3) no risk of catastrophic forgetting or overfitting to training data distributions.

#### 3.7.2 Sensitivity Analysis

The three layers exhibit markedly different sensitivity profiles:

- **L1 (Prompt):** Robust to implementation choices; the key constraint is maintaining $\leq 8\%$ edit ratio.
- **L2 (SVD Noise):** Extremely sensitive to $\alpha$; the effective range is $[0.001, 0.005]$ with severe degradation outside this interval.
- **L3 (Feature Injection):** Moderately sensitive to $\lambda_{\max}$ and layer selection; the adaptive gating mechanism provides self-regulation that reduces hyperparameter sensitivity.

The asymmetric sensitivity motivates the design of TSR (adaptive $\alpha$) and the multi-stage gating in Layer 3—both serve to make the system more robust across diverse video types.

#### 3.7.3 Computational Overhead

The additional computation over standard T2V generation consists of: (1) one inversion pass ($N_{\text{inv}} = 50$ model evaluations), (2) SVD decomposition ($<1$s on GPU), and (3) per-step feature blending (negligible FLOP overhead). The total generation time increases by approximately $1.7\times$ compared to standard generation (dominated by the inversion pass), with no increase in GPU memory beyond the feature cache ($\sim$2GB for 10 layers $\times$ 30 steps).

---

*End of Method Section*
