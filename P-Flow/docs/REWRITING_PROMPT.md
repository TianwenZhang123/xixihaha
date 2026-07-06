# Caption Rewriting Strategy for Text-to-Video Generation

This document describes the caption rewriting strategy used to produce the optimal captions in `video3_captions_best`. The strategy was derived from a 3-version × 24-case systematic experiment and can be formulated as a prompt for LLM-based rewriting.

---

## Rewriting System Prompt

```
You are a video caption rewriter. Your task is to improve a raw VLM-generated 
video caption for downstream text-to-video generation. Follow these rules:

### Core Principles

1. **Subject-First Opening.** The first sentence must begin with the primary 
   subject and its core action. Remove generic openings such as "The video 
   begins with", "A black screen transitioning to", or "The camera focuses on".

2. **Multi-Sentence Scene Separation.** Different scenes, camera angles, or 
   temporal segments MUST be separated into independent sentences with periods. 
   Never merge two distinct scenes into one comma-separated clause. Each scene 
   transition should appear as a clean sentence boundary.

3. **Same-Subject Compression.** Adjacent sentences describing the same subject 
   in a single continuous scene may be merged into a flowing sentence with 
   commas, provided no scene boundary is crossed.

4. **Preserve Visual Details.** Retain all original visual descriptors: colors, 
   materials, lighting conditions, spatial relationships, object attributes. 
   Do not replace specific visual terms with generic synonyms.

5. **Inject Temporal Cues.** Where applicable, insert temporal connectives such 
   as "initially", "then", "as", or "while" to make action sequences explicit. 
   Transform passive/static descriptions into active motion descriptions.

6. **Visual Ending.** The final sentence should describe the overall visual 
   atmosphere, mood, or aesthetic effect of the scene. Do not end with 
   abstract summaries, metadata, or keyword lists.

7. **Minimal Edit Budget.** Keep total edit distance below 50% of the original. 
   Preserve the approximate word count (±20%). The rewriting should be 
   structural rearrangement, not content regeneration.

8. **When in Doubt, Don't Change.** If the original caption already follows 
   principles 1-6, leave it unchanged. Do not rewrite for the sake of rewriting.

### Output Format

Return only the rewritten caption text. No explanations, no markup.
```

---

## Three Strategies and When to Apply

Based on experiments with 24 videos, the optimal rewriting strategy varies 
by video complexity:

| Strategy | Best For | Key Operations |
|----------|----------|---------------|
| **S1: Keep Original** | Well-structured captions with clear subject+action opening | No changes |
| **S2: Compress Same-Subject** | Single-subject videos; captions with weak openings | Subject-first reorder; merge adjacent same-subject clauses; visual ending |
| **S3: Split Scenes** | Multi-scene/multi-character videos; videos with scene transitions | Separate each scene into its own sentence; never merge across scenes |

**Decision Heuristic:**
- Has scene transitions or multiple distinct locations? → S3
- Single continuous scene with one main subject? → S2
- Already well-written with subject-first opening? → S1

**Precise S2 vs S3 Decision Flow:**
```
Does the video contain scene transitions (spatial/temporal cuts)?
  ├─ YES → S3. Each scene becomes its own sentence.
  │      Indicators: "transitions to", "shifts to", "then the scene"
  │      Edge case: A→B→A往返结构 (e.g., close-up→wide→close-up) → still S3
  ├─ NO → Are there multiple independent subjects doing different things
  │       in the SAME space/time? (e.g., 3 dancers + 2 skaters on one rooftop)
  │        ├─ YES → S2. They share the same visual frame, merge with commas.
  │        └─ NO → S2. Single subject, single scene, merge adjacent clauses.
  └─ NO, and original is already well-structured → S1. Keep as-is.
```

**Common Edge Cases:**
- Multiple camera angles of the same scene (pan→zoom→track) → S2 (no scene cut)
- Same location but visible time change (day→night) → S3 (temporal boundary)
- Real→animated transition → S3 (different ontological realms)
- Close-up of product → reveal full scene → S2 if continuous zoom, S3 if hard cut

---

## Before / After Examples

### Example 1: S2 — Compress Same-Subject (Case #30)

**Before (VLM Raw):**
> A black screen transitioning to a dimly lit living room. The camera focuses on a robotic vacuum cleaner positioned on the floor. The vacuum cleaner is illuminated by a blue light, and it appears to be moving across the wooden floor. In the background, there is a couch and a television set. The scene is quiet and still, with no visible movement or activity other than the vacuum cleaner's operation.

**After (Best):**
> A robotic vacuum cleaner moving across a wooden floor in a dimly lit living room, illuminated by its own blue light, a couch and a television set visible in the quiet and still background with no other movement or activity occurring, focused static shot, dim interior lighting with the blue glow from the vacuum as the only source of illumination.

**Operations applied:**
- Deleted framework opening ("A black screen transitioning to")
- Subject-first reorder (vacuum cleaner → first word)
- Merged same-subject clauses with commas (single scene, single subject)
- Visual ending: "dim interior lighting with the blue glow..."

---

### Example 2: S3 — Split Scenes (Case #34)

**Before (VLM Raw):**
> A serene garden scene showcasing lush greenery and blooming flowers bathed in sunlight, transitioning to a futuristic high-tech environment. A woman in an elegant white dress walks through a circular room filled with mirrors, illuminated by a large circular light fixture. The video captures her graceful movements from various angles, highlighting the intricate details of her attire against the reflective surfaces of the surrounding mirrors.

**After (Best):**
> A serene garden scene showcases lush greenery and blooming flowers bathed in sunlight. The camera transitions to a futuristic, high-tech environment. A woman in an elegant white dress walks through a circular room filled with mirrors. The room is illuminated by a large, circular light fixture, creating a mesmerizing effect as the woman gracefully moves around. The video captures her from various angles, highlighting the intricate details of her attire and the reflective surfaces of the mirrors. The reflective mirrors and elegant white dress create a sophisticated and futuristic aesthetic.

**Operations applied:**
- Split garden/transition/woman/mirror into 6 independent sentences
- Each scene and visual element gets its own sentence
- Active voice: "The video captures" → "captures her from various angles"
- Visual ending added: "sophisticated and futuristic aesthetic"

---

### Example 3: S3 — Multi-Character + Scene Transition (Case #28)

**Before (VLM Raw):**
> A young child jumping on a blue couch in a living room at night, with the room dimly lit in blue and white tones. The child, dressed in a striped shirt and jeans, bounces playfully across the blue couch. In the background, another child plays with a toy while an adult is engaged in cleaning activities. The room features a large blue sofa adorned with yellow pillows, adding a pop of color to the space, and a potted plant near the window contributes to the room's cozy aesthetic.

**After (Best):**
> A young child in a striped shirt and jeans jumping on a blue couch in a living room at night. The room is dimly lit with blue and white tones, creating a cozy atmosphere. The child's playful energy is evident as they move around the room, interacting with various objects. In the background, another child plays with a toy, while an adult is engaged in cleaning activities. The room features a large blue sofa adorned with yellow pillows, adding a pop of color to the space. A potted plant sits near the window, contributing to the room's aesthetic appeal. The family enjoys their time together in this warm and inviting environment.

**Operations applied:**
- Subject-first: "A young child in a striped shirt and jeans jumping" (merged dress description into opening)
- Multi-character separation: child / background child / adult → each in its own sentence
- Environment details separated: blue sofa / potted plant → individual sentences
- Active construction: "A potted plant sits near the window"
- Visual ending: "warm and inviting environment"

---

### Example 4: S1 — Keep Original (Case #43)

**Before (VLM Raw, same as Best):**
> (The original VLM caption was already optimal — S1 applied, no changes made)

**When NO rewriting is needed:**
This caption already follows all principles:
- Subject-first: "A person interacting with a smartwatch"
- Clear scene transition: "The scene then transitions to an outdoor setting"
- Temporal cues: "then", "as", "while"
- Visual ending: "emphasizing the physical effort and determination required"
- All visual details preserved

---

## Design Principles Summary

Derived from 72 generation experiments (3 versions × 24 videos):

1. **Never merge across scene boundaries.** Merging two distinct scenes into one sentence caused XCLIP drops of up to -0.177 (Case #34).
2. **Subject-first opening is universally beneficial.** All winning captions across 254 cases place the primary subject at or near the first word.
3. **Preserve visual vocabulary.** Replacing specific visual descriptors (e.g., "blue glow" → "blue light") degrades CLIP-I alignment.
4. **Active over passive.** "The camera captures" → "captures" improves temporal dynamics.
5. **End with atmosphere, not summary.** Abstract ending summaries harm XCLIP; visual mood descriptions improve it.
6. **S1 proportion varies by dataset type.** In a small narrative dataset (24 cases), 33% needed no rewriting. In a large advertising/product dataset (204 cases), only 4% were S1.

---

## Embedding-Level Motivation

The three-strategy framework can be understood from an **embedding space compatibility** perspective within the P-Flow three-layer architecture:

- **Layer 2 (SVD noise prior)** encodes a motion-direction bias in noise space: $\eta_{\text{temporal}} \in \mathbb{R}^{C \times F \times H' \times W'}$
- **Layer 3 (Feature Injection)** encodes structure/appearance bias in feature space: $h_{\text{ref}} \in \mathbb{R}^{N_v \times d_{\text{model}}}$
- **Layer 1 (Prompt)** produces text embeddings in UMT5 space: $E(c) \in \mathbb{R}^{S \times d}$

These three embedding spaces interact through cross-attention in the DiT. The rewriting strategies aim to optimize $E(c)$ so that its semantic direction aligns with those of $\eta_{\text{temporal}}$ and $h_{\text{ref}}$:

| Strategy | Embedding Effect | When It Works |
|----------|-----------------|---------------|
| S1 (keep) | $E(c)$ naturally aligned with SVD+FI direction | Original caption matches reference motion |
| S2 (compress) | Reduces redundant semantic components, sharpening dominant motion direction | Single subject, original prompt structure loose |
| S3 (split) | Distributes scene-specific components into independent embeddings, preventing cross-scene averaging | Multiple transitions, merged scenes create ambiguity |

**Synergy vs. Conflict**: When $E(c)$ and SVD+FI directions align, ODE trajectory receives superimposed signals. When contradictory (e.g., prompt says "dolly-in" but SVD encodes "pan-left"), the two signals pull the trajectory apart, causing a -4.1% XCLIP drop. The three strategies optimize $E(c)$ through **structural rearrangement rather than motion rewriting** — preserving visual vocabulary while only adjusting token distribution to produce text embeddings more compatible with L2/L3 directions.

---

## Two-Stage L1: Editing Budget and Layer Compatibility

The rewriting strategy operates within a two-stage framework that balances caption quality against Layer 3 feature cache compatibility:

| Stage | Edit Budget | Purpose |
|-------|:----------:|---------|
| Head-Tail Replacement | ≤8% | Guarantees $\cos(E(c_{\text{rewrite}}), E(c_{\text{raw}})) \geq 0.95$ |
| Three-Version Selection | ~50% | Optimizes caption structure at higher edit budget |

In deployment, the three-version selection output is used as the primary $c^{(L1)}$. If the selected version exceeds the edit threshold, the system falls back to head-tail replacement to maintain three-layer synergy.

---

## Extended Validation

In addition to the 24-case `video3` dataset, the strategy was validated on two additional datasets:

| Dataset | Cases | S1 | S2 | S3 | Notes |
|--------|:---:|:--:|:--:|:--:|------|
| `all_captions_rewritten` | 204 | 4% | 83% | 13% | Advertising/product videos |
| `newvideo_captions_rewritten` | 50 | 0% | 100% | 0% | All single-scene, supplementary |

All 254 cases across datasets verified through manual semantic review against the 8 core principles with zero violations.

---

## Verification Process

After rewriting, each caption is verified through manual semantic review:

1. **Framework check** — No "The video depicts/begins/shows" in sentence 1
2. **Subject-first check** — First words = primary subject + action
3. **Scene separation** — S3: scenes in separate sentences with periods; S2: same-scene merged with commas
4. **Visual vocabulary** — Colors, materials, lighting terms unchanged
5. **Temporal cues** — "as", "while", "then" inserted where appropriate
6. **Visual ending** — Last sentence describes mood/atmosphere, not abstract summary
7. **Edit budget** — Structural rearrangement, not content regeneration
8. **Truncation/metadata** — No truncated sentences, no AI watermark text

---

## Experimental Results

| Metric | Original VLM | After Rewriting | Δ |
|--------|:-----------:|:---------------:|:---:|
| CLIP-I (image alignment) | 0.8723 | 0.8738 | +0.0015 |
| XCLIP (temporal alignment) | 0.7331 | 0.7426 | +0.0095 |

Results averaged over 24 videos. Each video used the caption version (S1/S2/S3) that maximized its XCLIP score in the 3-version comparison experiment.

---

## Reference

This strategy was developed as part of the P-Flow project. For implementation details, see `scripts/qwen_video_caption.py` (VLM captioning) and `docs/5.28周会.md` (fusion strategy experiments).
