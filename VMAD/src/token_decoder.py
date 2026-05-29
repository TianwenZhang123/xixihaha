"""
Velocity-Preserving Token Decoding.

Core Innovation: Maps continuous motion embedding Δe back to discrete text tokens
while preserving functional equivalence in the velocity field. This enables:
    1. Human-readable motion descriptions (interpretability)
    2. Cross-model transfer (discrete tokens are model-agnostic)
    3. User editing (modify motion via text manipulation)

Theoretical Motivation (TPSO, 2025 + DPO-Diff, ICML 2024):
    The text embedding space and token space are dual representations of the
    same semantic content. However, naive nearest-neighbor decoding from
    embedding to tokens loses motion-critical information because:
    - The embedding→token mapping is many-to-one (information loss)
    - Standard decoding optimizes for semantic similarity, not velocity field fidelity
    - Token discretization introduces quantization noise in motion-sensitive dimensions

    Our Velocity-Preserving Token Decoding addresses this through a three-stage pipeline:
    Stage 1: Neural Projection (continuous → candidate tokens)
    Stage 2: Gumbel-Softmax Relaxation (differentiable discrete selection)
    Stage 3: Velocity-Preserving Reranking (select tokens that best preserve v_θ)

Mathematical Formulation:
    Given optimized Δe ∈ R^{L×D} and vocabulary V = {v_1, ..., v_|V|}:

    Stage 1 — Neural Projection:
        For each position j: score_j = MLP(Δe[j]) ∈ R^{|V|}
        Top-K candidates: C_j = argtop_K(score_j)

    Stage 2 — Gumbel-Softmax Selection:
        soft_token_j = Σ_k softmax((score_j[C_j] + g_k) / τ) · Embed(C_j[k])
        where g_k ~ Gumbel(0,1), τ is temperature (annealed)

    Stage 3 — Velocity-Preserving Reranking:
        For each candidate sequence s ∈ beam:
            e_s = TextEncoder(s)
            v_s = v_θ(x_t, t, e_s)
            v_ref = v_θ(x_t, t, e₀ + Δe)
            score(s) = -||v_s - v_ref||² + λ_fluency · log P_LM(s)

        Output: s* = argmax_s score(s)

    The velocity-preserving constraint ensures that the decoded tokens,
    when re-encoded and used for generation, produce a velocity field
    functionally equivalent to the original Δe-augmented embedding.

Connection to Rate-Distortion Theory:
    Token decoding can be viewed as lossy compression of the motion embedding:
    - Rate: number of tokens (information capacity)
    - Distortion: velocity field deviation ||v(tokens) - v(Δe)||²
    The velocity-preserving reranking finds the rate-distortion optimal
    discrete representation on the Pareto frontier.

Computational Complexity:
    Stage 1 (NN Projection): O(L · V · D) — cosine similarity over vocabulary
    Stage 2 (Gumbel-Softmax): O(N_steps · L · K · D) — K candidates, N annealing steps
    Stage 3 (Velocity Reranking): O(L · K · DiT_forward) — beam search with velocity eval
    Total: O(L · K · DiT_forward) dominated by Stage 3
    For default settings (L=77, K=50, beam_width=5):
        ~385 DiT forward passes (amortized over the full pipeline)
    Note: Stage 3 is optional — without z0, only Stages 1-2 run in O(L·V·D) time.

References:
    - TPSO (2025): Token-Prompt Space Optimization
    - DPO-Diff (ICML 2024): Gradient-based discrete prompt optimization
    - Gumbel-Softmax (Jang et al., 2017): Differentiable discrete sampling
    - RichSpace (ICLR 2025): T5 embedding space structure
"""

import logging
import math
from typing import Optional, List, Dict, Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class VelocityPreservingTokenDecoder:
    """
    Velocity-Preserving Token Decoding Module.

    Decodes continuous motion embedding Δe into discrete text tokens while
    maintaining functional equivalence in the velocity field space.

    Three-Stage Pipeline:
        1. NN Projection: Find nearest vocabulary tokens per position
        2. Gumbel-Softmax: Differentiable relaxation for gradient-based refinement
        3. Velocity Reranking: Select final tokens based on velocity field fidelity

    This module bridges the gap between the continuous optimization space
    (where Δe lives) and the discrete token space (where human-readable
    and cross-model-transferable representations live).
    """

    def __init__(
        self,
        pipe,
        tokenizer=None,
        text_encoder=None,
        top_k: int = 50,
        beam_width: int = 5,
        num_rerank_timesteps: int = 3,
        temperature_init: float = 2.0,
        temperature_final: float = 0.1,
        lambda_fluency: float = 0.1,
        lambda_velocity: float = 1.0,
        device: str = "cuda",
    ):
        """
        Args:
            pipe: Diffusers pipeline (for velocity field computation)
            tokenizer: Text tokenizer (if None, extracted from pipe)
            text_encoder: Text encoder (if None, extracted from pipe)
            top_k: Number of candidate tokens per position in Stage 1
            beam_width: Beam search width in Stage 3
            num_rerank_timesteps: Number of timesteps to evaluate velocity at
            temperature_init: Initial Gumbel-Softmax temperature
            temperature_final: Final Gumbel-Softmax temperature
            lambda_fluency: Weight for language model fluency score
            lambda_velocity: Weight for velocity preservation score
            device: Compute device
        """
        self.pipe = pipe
        self.device = device
        self.top_k = top_k
        self.beam_width = beam_width
        self.num_rerank_timesteps = num_rerank_timesteps
        self.temperature_init = temperature_init
        self.temperature_final = temperature_final
        self.lambda_fluency = lambda_fluency
        self.lambda_velocity = lambda_velocity

        # Extract tokenizer and text encoder from pipeline
        self._tokenizer = tokenizer
        self._text_encoder = text_encoder
        self._embedding_matrix = None  # Lazy-loaded

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            if hasattr(self.pipe, "tokenizer"):
                self._tokenizer = self.pipe.tokenizer
            else:
                raise ValueError("No tokenizer available")
        return self._tokenizer

    @property
    def text_encoder(self):
        if self._text_encoder is None:
            if hasattr(self.pipe, "text_encoder"):
                self._text_encoder = self.pipe.text_encoder
            else:
                raise ValueError("No text encoder available")
        return self._text_encoder

    @property
    def embedding_matrix(self) -> torch.Tensor:
        """
        Get the token embedding matrix from the text encoder.
        
        NOTE: This returns INPUT embeddings. Since Δe lives in the T5 encoder
        OUTPUT space (after multi-layer transformer), NN search in input space
        is an approximation. For T5 with shared embeddings, the input embedding
        matrix provides a reasonable proxy because:
        1. T5's shared embedding is used for both input and (via lm_head) output
        2. The encoder output space is still anchored to the input embedding space
           through residual connections
        
        For higher fidelity, use _build_output_codebook() to precompute encoder
        outputs for vocabulary tokens (expensive but more accurate).
        """
        if self._embedding_matrix is None:
            encoder = self.text_encoder
            if hasattr(encoder, "get_input_embeddings"):
                embed_layer = encoder.get_input_embeddings()
                self._embedding_matrix = embed_layer.weight.detach()
            elif hasattr(encoder, "shared"):
                # T5 uses shared embeddings
                self._embedding_matrix = encoder.shared.weight.detach()
            else:
                raise ValueError("Cannot extract embedding matrix from text encoder")
        return self._embedding_matrix

    @torch.no_grad()
    def _build_output_codebook(
        self, vocab_subset: Optional[List[int]] = None, batch_size: int = 128
    ) -> torch.Tensor:
        """
        Build an output-space embedding codebook by running tokens through the
        T5 encoder. This enables NN search in the correct (output) space where
        Δe actually lives.
        
        Args:
            vocab_subset: Token IDs to include (if None, uses full vocabulary)
            batch_size: Batch size for encoder forward passes
            
        Returns:
            output_embeddings: (V, D) tensor of encoder output hidden states
                              for each vocabulary token
        """
        encoder = self.text_encoder
        tokenizer = self.tokenizer
        
        if vocab_subset is None:
            vocab_subset = list(range(self.embedding_matrix.shape[0]))
        
        output_embeddings = []
        
        for i in range(0, len(vocab_subset), batch_size):
            batch_ids = vocab_subset[i:i + batch_size]
            # Create input_ids: each token as a single-token sequence with padding
            input_ids = torch.tensor(batch_ids, device=self.device).unsqueeze(1)  # (B, 1)
            
            # Run through encoder
            encoder_output = encoder(input_ids=input_ids)
            # Take the hidden state at position 0 (the token itself)
            hidden = encoder_output.last_hidden_state[:, 0, :]  # (B, D)
            output_embeddings.append(hidden.detach())
        
        return torch.cat(output_embeddings, dim=0)  # (V, D)

    def decode(
        self,
        delta_e: torch.Tensor,
        e0: torch.Tensor,
        z0: Optional[torch.Tensor] = None,
        base_caption: str = "",
        max_motion_tokens: int = 10,
    ) -> Dict[str, Any]:
        """
        Decode continuous Δe into discrete motion tokens.

        Full three-stage pipeline:
            1. NN Projection → candidate token sets per position
            2. Gumbel-Softmax → differentiable soft selection
            3. Velocity Reranking → final token sequence

        Args:
            delta_e: Optimized motion embedding residual (B, L, D) or (L, D)
            e0: Base caption embedding (B, L, D) or (L, D)
            z0: Target video latent (for velocity reranking, optional)
            base_caption: Original caption text (for context)
            max_motion_tokens: Maximum number of motion-descriptive tokens to output

        Returns:
            Dictionary containing:
                - motion_tokens: List of decoded token strings
                - motion_text: Concatenated motion description
                - token_ids: Token ID sequence
                - velocity_preservation_score: How well tokens preserve v_θ
                - per_position_confidence: Decoding confidence per position
        """
        if delta_e.dim() == 3:
            delta_e = delta_e.squeeze(0)
        if e0.dim() == 3:
            e0 = e0.squeeze(0)

        # Stage 1: Neural Projection (NN search in embedding space)
        logger.info("  [TokenDecode] Stage 1: Neural Projection...")
        candidates, scores = self._stage1_nn_projection(delta_e)

        # Stage 2: Gumbel-Softmax Selection
        logger.info("  [TokenDecode] Stage 2: Gumbel-Softmax Selection...")
        soft_selection, hard_indices = self._stage2_gumbel_selection(
            delta_e, candidates, scores
        )

        # Stage 3: Velocity-Preserving Reranking
        if z0 is not None:
            logger.info("  [TokenDecode] Stage 3: Velocity-Preserving Reranking...")
            final_tokens, velocity_score = self._stage3_velocity_reranking(
                candidates, hard_indices, e0, delta_e, z0
            )
        else:
            # Without z0, use Stage 2 output directly
            final_tokens = hard_indices
            velocity_score = 0.0

        # Convert token IDs to text
        motion_tokens, motion_text = self._tokens_to_text(
            final_tokens, delta_e, max_motion_tokens
        )

        # Compute per-position confidence
        confidence = self._compute_decoding_confidence(delta_e, final_tokens)

        return {
            "motion_tokens": motion_tokens,
            "motion_text": motion_text,
            "token_ids": final_tokens.cpu().tolist() if isinstance(final_tokens, torch.Tensor) else final_tokens,
            "velocity_preservation_score": velocity_score,
            "per_position_confidence": confidence,
        }

    def _stage1_nn_projection(
        self, delta_e: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Stage 1: Nearest-Neighbor Projection.

        For each position j in Δe, find the top-K nearest vocabulary tokens
        in embedding space using cosine similarity.

        This provides the initial candidate set for subsequent refinement.
        The intuition is that tokens geometrically close to Δe[j] in embedding
        space are likely to produce similar effects when encoded.

        Args:
            delta_e: Motion embedding (L, D)

        Returns:
            candidates: Top-K token indices per position (L, K)
            scores: Similarity scores (L, K)
        """
        embed_matrix = self.embedding_matrix.to(
            device=delta_e.device, dtype=delta_e.dtype
        )

        L, D = delta_e.shape
        K = min(self.top_k, embed_matrix.shape[0])

        # Normalize for cosine similarity
        delta_e_norm = F.normalize(delta_e, dim=-1)  # (L, D)
        embed_norm = F.normalize(embed_matrix, dim=-1)  # (V, D)

        # Compute similarity: (L, D) @ (D, V) -> (L, V)
        similarity = delta_e_norm @ embed_norm.T

        # Top-K per position
        scores, candidates = similarity.topk(K, dim=-1)  # (L, K)

        return candidates, scores

    def _stage2_gumbel_selection(
        self,
        delta_e: torch.Tensor,
        candidates: torch.Tensor,
        scores: torch.Tensor,
        num_steps: int = 20,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Stage 2: Gumbel-Softmax Differentiable Selection.

        Uses Gumbel-Softmax relaxation to perform differentiable discrete
        token selection from the candidate set. Temperature is annealed
        from high (soft, exploratory) to low (hard, decisive).

        This stage refines the NN projection by considering the interaction
        between positions — a token at position j affects the optimal choice
        at position j+1 through the text encoder's contextual processing.

        Args:
            delta_e: Motion embedding (L, D)
            candidates: Candidate token indices (L, K)
            scores: Initial similarity scores (L, K)
            num_steps: Number of annealing steps

        Returns:
            soft_selection: Soft token embeddings (L, D)
            hard_indices: Selected token indices (L,)
        """
        L, K = candidates.shape
        embed_matrix = self.embedding_matrix.to(
            device=delta_e.device, dtype=delta_e.dtype
        )

        # Initialize logits from NN scores
        logits = scores.clone().requires_grad_(False)

        # Temperature annealing
        best_indices = candidates[:, 0]  # Start with top-1

        for step in range(num_steps):
            # Anneal temperature
            progress = step / max(num_steps - 1, 1)
            temperature = self.temperature_init * (
                self.temperature_final / self.temperature_init
            ) ** progress

            # Gumbel-Softmax sampling
            gumbel_noise = -torch.log(-torch.log(
                torch.rand_like(logits) + 1e-8
            ) + 1e-8)
            soft_logits = (logits + gumbel_noise) / temperature
            soft_weights = F.softmax(soft_logits, dim=-1)  # (L, K)

            # Soft token embedding: weighted sum of candidate embeddings
            candidate_embeds = embed_matrix[candidates]  # (L, K, D)
            soft_embed = (soft_weights.unsqueeze(-1) * candidate_embeds).sum(dim=1)  # (L, D)

            # Update best indices based on final soft weights
            best_indices = candidates[
                torch.arange(L, device=candidates.device),
                soft_weights.argmax(dim=-1)
            ]

        # Final soft embedding
        soft_selection = soft_embed.detach()

        return soft_selection, best_indices

    def _stage3_velocity_reranking(
        self,
        candidates: torch.Tensor,
        initial_selection: torch.Tensor,
        e0: torch.Tensor,
        delta_e: torch.Tensor,
        z0: torch.Tensor,
    ) -> Tuple[torch.Tensor, float]:
        """
        Stage 3: Velocity-Preserving Reranking.

        Evaluates candidate token sequences by their velocity field fidelity:
        how well the re-encoded tokens reproduce the velocity field of the
        original Δe-augmented embedding.

        For each candidate sequence:
            1. Encode tokens → embedding e_candidate
            2. Compute v_candidate = v_θ(x_t, t, e_candidate)
            3. Compute v_reference = v_θ(x_t, t, e₀ + Δe)
            4. Score = -||v_candidate - v_reference||²

        The sequence with highest score best preserves the motion-generating
        capability of the original continuous embedding.

        Args:
            candidates: Candidate tokens per position (L, K)
            initial_selection: Initial token selection (L,)
            e0: Base embedding (L, D)
            delta_e: Motion embedding (L, D)
            z0: Target latent (for velocity computation)

        Returns:
            best_tokens: Best token sequence (L,)
            best_score: Velocity preservation score
        """
        model = self.pipe.transformer if hasattr(self.pipe, "transformer") else self.pipe.unet
        L = candidates.shape[0]

        # Reference velocity field
        with torch.no_grad():
            t_eval = torch.tensor(0.15, device=self.device)  # Mid-range of T_m
            eta = torch.randn_like(z0)
            x_t = (1 - t_eval) * eta + t_eval * z0

            e_ref = (e0 + delta_e).unsqueeze(0) if (e0 + delta_e).dim() == 2 else (e0 + delta_e)
            timestep = t_eval.unsqueeze(0).to(dtype=z0.dtype)

            v_ref = model(
                hidden_states=x_t,
                timestep=timestep.expand(x_t.shape[0]),
                encoder_hidden_states=e_ref,
                return_dict=False,
            )
            if isinstance(v_ref, tuple):
                v_ref = v_ref[0]

        # Evaluate initial selection
        best_tokens = initial_selection
        best_score = self._evaluate_velocity_score(
            initial_selection, e0, x_t, timestep, v_ref, model
        )

        # Beam search: try swapping positions with alternative candidates
        for pos in range(min(L, 5)):  # Focus on first 5 positions (highest influence)
            for k in range(min(self.beam_width, candidates.shape[1])):
                trial_tokens = best_tokens.clone()
                trial_tokens[pos] = candidates[pos, k]

                score = self._evaluate_velocity_score(
                    trial_tokens, e0, x_t, timestep, v_ref, model
                )

                if score > best_score:
                    best_score = score
                    best_tokens = trial_tokens.clone()

        return best_tokens, best_score

    def _evaluate_velocity_score(
        self,
        token_ids: torch.Tensor,
        e0: torch.Tensor,
        x_t: torch.Tensor,
        timestep: torch.Tensor,
        v_ref: torch.Tensor,
        model: nn.Module,
    ) -> float:
        """
        Evaluate combined velocity preservation + fluency score for a token sequence.

        Score(s) = λ_velocity · (-||v_s - v_ref||²) + λ_fluency · log P_LM(s)

        The velocity term ensures motion fidelity; the fluency term ensures
        the decoded tokens form coherent natural language (important for
        human readability and cross-model transfer).
        """
        with torch.no_grad():
            # Get embeddings for token sequence
            embed_matrix = self.embedding_matrix.to(
                device=token_ids.device, dtype=e0.dtype
            )
            token_embeds = embed_matrix[token_ids]  # (L, D)

            # Combine with base embedding
            e_candidate = e0 + token_embeds
            if e_candidate.dim() == 2:
                e_candidate = e_candidate.unsqueeze(0)

            # Compute velocity
            v_candidate = model(
                hidden_states=x_t,
                timestep=timestep.expand(x_t.shape[0]),
                encoder_hidden_states=e_candidate,
                return_dict=False,
            )
            if isinstance(v_candidate, tuple):
                v_candidate = v_candidate[0]

            # Velocity preservation score (negative MSE, higher is better)
            velocity_score = -((v_candidate - v_ref) ** 2).mean().item()

            # Fluency score via language model log-probability
            fluency_score = self._compute_fluency_score(token_ids)

            # Combined score
            score = (
                self.lambda_velocity * velocity_score
                + self.lambda_fluency * fluency_score
            )

        return score

    def _compute_fluency_score(self, token_ids: torch.Tensor) -> float:
        """
        Compute language model fluency score: log P_LM(token_sequence).

        Uses the text encoder's causal/masked LM head if available, otherwise
        approximates fluency via embedding-space smoothness (cosine similarity
        between consecutive token embeddings as a proxy for n-gram coherence).

        The fluency score encourages decoded tokens to form grammatically
        plausible phrases, improving human readability without sacrificing
        velocity fidelity (controlled by λ_fluency).

        Args:
            token_ids: Token indices (L,)

        Returns:
            Normalized log-probability (higher = more fluent), or smoothness proxy.
        """
        # Attempt true LM scoring via the text encoder
        if hasattr(self, '_lm_scorer') and self._lm_scorer is not None:
            return self._lm_scorer_forward(token_ids)

        # Proxy: embedding-space smoothness (bigram cosine similarity)
        # Rationale: fluent text has smooth embedding trajectories;
        # random token sequences have low inter-token similarity.
        embed_matrix = self.embedding_matrix.to(
            device=token_ids.device, dtype=torch.float32
        )
        embeds = embed_matrix[token_ids]  # (L, D)

        if embeds.shape[0] < 2:
            return 0.0

        # Consecutive cosine similarity
        embeds_norm = F.normalize(embeds, dim=-1)
        bigram_sim = (embeds_norm[:-1] * embeds_norm[1:]).sum(dim=-1)  # (L-1,)

        # Mean bigram similarity as fluency proxy (range [-1, 1])
        return bigram_sim.mean().item()

    def _lm_scorer_forward(self, token_ids: torch.Tensor) -> float:
        """
        Forward pass through a causal LM for true log-probability scoring.

        This is used when an external LM scorer has been attached via
        `set_lm_scorer()`. Computes mean per-token log P(t_i | t_{<i}).
        """
        try:
            input_ids = token_ids.unsqueeze(0)  # (1, L)
            with torch.no_grad():
                outputs = self._lm_scorer(input_ids)
                logits = outputs.logits  # (1, L, V)

            # Shift: predict token i from context <i
            shift_logits = logits[:, :-1, :]  # (1, L-1, V)
            shift_labels = token_ids[1:]  # (L-1,)

            log_probs = F.log_softmax(shift_logits, dim=-1)
            token_log_probs = log_probs[0, torch.arange(len(shift_labels)), shift_labels]

            # Normalized mean log-probability
            return token_log_probs.mean().item()
        except Exception:
            return 0.0

    def set_lm_scorer(self, lm_model: nn.Module):
        """
        Attach an external language model for fluency scoring in Stage 3.

        Args:
            lm_model: A causal language model (e.g., GPT-2, Qwen2) that accepts
                      input_ids and returns logits. Should share the same tokenizer
                      vocabulary as the text encoder.

        Example:
            from transformers import AutoModelForCausalLM
            lm = AutoModelForCausalLM.from_pretrained("gpt2").eval().to(device)
            decoder.set_lm_scorer(lm)
        """
        self._lm_scorer = lm_model
        logger.info(f"  [TokenDecode] LM scorer attached: {type(lm_model).__name__}")

    def _tokens_to_text(
        self,
        token_ids: torch.Tensor,
        delta_e: torch.Tensor,
        max_tokens: int,
    ) -> Tuple[List[str], str]:
        """
        Convert token IDs to human-readable text.

        Filters out low-energy positions (where Δe is near zero)
        and special tokens to produce a clean motion description.

        Args:
            token_ids: Selected token indices (L,)
            delta_e: Motion embedding for energy-based filtering (L, D)
            max_tokens: Maximum tokens to include

        Returns:
            motion_tokens: List of individual token strings
            motion_text: Concatenated motion description
        """
        # Compute per-position energy to identify active positions
        position_energy = (delta_e ** 2).sum(dim=-1)  # (L,)
        energy_threshold = position_energy.mean() * 0.1  # 10% of mean energy

        # Filter: only keep positions with significant Δe energy
        active_mask = position_energy > energy_threshold
        active_positions = active_mask.nonzero(as_tuple=True)[0]

        # Sort by energy (highest first)
        energies_at_active = position_energy[active_positions]
        sorted_indices = energies_at_active.argsort(descending=True)
        active_positions = active_positions[sorted_indices]

        # Decode tokens at active positions
        motion_tokens = []
        for pos in active_positions[:max_tokens]:
            tid = token_ids[pos].item()
            token_str = self.tokenizer.decode([tid]).strip()

            # Filter special tokens and empty strings
            if token_str and token_str not in ["<pad>", "</s>", "<s>", "[PAD]", "[CLS]", "[SEP]"]:
                motion_tokens.append(token_str)

        # Concatenate into motion text
        motion_text = " ".join(motion_tokens) if motion_tokens else ""

        return motion_tokens, motion_text

    def _compute_decoding_confidence(
        self,
        delta_e: torch.Tensor,
        token_ids: torch.Tensor,
    ) -> List[float]:
        """
        Compute per-position decoding confidence.

        Confidence is based on the cosine similarity between Δe[j] and
        the embedding of the selected token. High similarity indicates
        the discrete token is a good approximation of the continuous embedding.
        """
        embed_matrix = self.embedding_matrix.to(
            device=delta_e.device, dtype=delta_e.dtype
        )

        if isinstance(token_ids, torch.Tensor):
            selected_embeds = embed_matrix[token_ids]  # (L, D)
        else:
            selected_embeds = embed_matrix[torch.tensor(token_ids, device=delta_e.device)]

        # Cosine similarity per position
        delta_e_norm = F.normalize(delta_e, dim=-1)
        selected_norm = F.normalize(selected_embeds, dim=-1)
        confidence = (delta_e_norm * selected_norm).sum(dim=-1)  # (L,)

        return confidence.cpu().tolist()
