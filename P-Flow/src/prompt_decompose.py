"""
方向4+1: 结构化 Prompt 分解 + CLIPScore 择优 (非侵入式)

流程:
  1. LLM 将原始 caption 分解为 5 个组件 (Subject/Scene/Motion/Camera/Style)
  2. 每个组件 LLM 生成 2-3 个变体表述
  3. CLIPScore 评估每个变体与参考视频的视觉-文本对齐度
  4. 每个组件选最高分变体，组装最终 prompt

启动方式:
  --prompt_decompose           启用 (默认关闭)
  --llm_api_key KEY            API Key (或环境变量 LLM_API_KEY)
  --llm_api_base URL           默认 https://token-plan-cn.xiaomimimo.com/v1
  --llm_model NAME             默认 mimo-v2.5-pro

学术出处:
  - CLIPScore: "Pick-a-Pic" (Kirstain et al., 2023)
  - 结构化分解: "VideoComposer" (Wang et al., 2023)
"""

import os
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path

import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# LLM Prompt 模板
# ─────────────────────────────────────────────────────────────

DECOMPOSE_SYSTEM = """You are a professional video caption analyzer. Your task is to decompose a video caption into 5 structured components.

## Output Format (JSON only, no markdown):
{
  "subject": "The main subjects/objects in the video — their identity, appearance, quantity, clothing, features",
  "scene": "The background, environment, setting, location, weather, time of day, depth",
  "motion": "What movements happen — actions, gestures, speed, direction, trajectories, camera movement",
  "camera": "Camera specifications — shot type (close-up/wide/medium), angle, camera movement type",
  "style": "Visual style — color palette, contrast, saturation, lighting mood, artistic quality"
}

## Rules:
1. Each component must be a self-contained descriptive phrase (not just keywords)
2. DO NOT invent information not present in the original caption
3. If a component has no information in the caption, use "not specified"
4. Subject component: include appearance adjectives, NOT action verbs
5. Motion component: include action verbs and direction ONLY
6. Output ONLY valid JSON, no explanation
"""

DECOMPOSE_USER = """Decompose this video caption:
"{caption}"
"""

VARIANT_SYSTEM = """You are a professional prompt engineer for text-to-video generation.

Generate {n_variants} alternative phrasings for a specific component of a video caption.
Target the {component_name} aspect: {component_description}

## Rules:
1. Each variant must be a self-contained phrase (5-15 words)
2. Vary vocabulary, specificity level, and phrasing style
3. Maintain the SAME factual meaning as the original
4. More specific is better (prefer "golden retriever" over "dog")
5. Include ALL relevant visual details from the original

Output ONLY a JSON array of strings:
["variant 1", "variant 2", "variant 3"]
"""

VARIANT_USER = """Original caption: "{full_caption}"
Current {component_name} text: "{current_text}"

Generate {n_variants} alternative phrasings for the {component_name} component:"""

ASSEMBLE_SYSTEM = """You are a professional prompt engineer for text-to-video models. Assemble a final T2V prompt from structured components.

## Empirical T2V Attention Rule (CRITICAL):
The DiT model's cross-attention has a U-shaped position weight distribution:
- Token positions 0-3 receive 10-15x MORE attention than middle positions
- The LAST 1-2 tokens receive ~equal attention as position 0
- ALL middle tokens receive nearly uniform low attention

Therefore:
1. OPENING (positions 0-3): Place the 1-3 most distinctive visual descriptors — 
   subject noun + key visual adjective first. This is the most important decision.
   Example: "Golden retriever, snowy forest. Two golden retrievers..."
   NOT: "Two golden retrievers playing in a snowy forest..."

2. MIDDLE (positions 4-N-2): Natural fluent description of scene, motion, camera.
   Keep motion descriptions MINIMAL — motion is injected by another system component.
   Focus on APPEARANCE details: colors, textures, objects, spatial relationships.

3. ENDING (last 1-2 tokens): Concrete visual keywords, NOT abstract atmosphere summary.
   BAD:  "...creating a serene atmosphere."
   GOOD: "...warm sunlight, soft shadows."

## Output Rules:
1. 80-150 words single paragraph
2. NO preamble ("The video depicts", "This scene shows", "We see", etc.)
3. First 3-5 words MUST be the most distinctive visual descriptors
4. Ending MUST be concrete visual keywords (colors, textures, objects)
5. Minimize motion words — let the motion system handle movement
6. Output ONLY the assembled prompt, no markdown, no JSON, no explanation
"""

ASSEMBLE_USER = """Assemble a T2V prompt from these components:
- Subject: {subject}
- Scene: {scene}
- Motion: {motion}
- Camera: {camera}
- Style: {style}
"""


# ─────────────────────────────────────────────────────────────
# LLM Client
# ─────────────────────────────────────────────────────────────

class LLMClient:
    """OpenAI-compatible LLM client."""

    def __init__(self, api_key: str, api_base: str, model: str, temperature: float = 0.5):
        self.api_key = api_key
        self.api_base = api_base
        self.model = model
        self.temperature = temperature

    def _call(self, system: str, user: str, temperature: float = None,
              max_tokens: int = 1024, response_format: dict = None) -> str:
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai package required for prompt decomposition. pip install openai")

        client = OpenAI(
            api_key=self.api_key,
            base_url=self.api_base,
            timeout=60,
        )
        kwargs = dict(
            model=self.model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=temperature or self.temperature,
            max_tokens=max_tokens,
        )
        if response_format:
            kwargs["response_format"] = response_format

        resp = client.chat.completions.create(**kwargs)
        return resp.choices[0].message.content.strip()

    def decompose(self, caption: str) -> Dict[str, str]:
        """将 caption 分解为 5 个组件."""
        import json
        resp = self._call(DECOMPOSE_SYSTEM, DECOMPOSE_USER.format(caption=caption))
        try:
            return json.loads(resp)
        except json.JSONDecodeError:
            # 尝试提取 JSON
            import re
            match = re.search(r'\{.*\}', resp, re.DOTALL)
            if match:
                return json.loads(match.group())
            raise ValueError(f"LLM returned non-JSON: {resp[:200]}")

    def generate_variants(self, component_name: str, component_desc: str,
                          full_caption: str, current_text: str,
                          n_variants: int = 3) -> List[str]:
        """为单个组件生成变体."""
        import json
        system = VARIANT_SYSTEM.format(
            n_variants=n_variants,
            component_name=component_name,
            component_description=component_desc,
        )
        user = VARIANT_USER.format(
            full_caption=full_caption,
            component_name=component_name,
            current_text=current_text,
            n_variants=n_variants,
        )
        resp = self._call(system, user)
        try:
            return json.loads(resp)
        except json.JSONDecodeError:
            import re
            match = re.search(r'\[.*\]', resp, re.DOTALL)
            if match:
                return json.loads(match.group())
            # fallback: 按行分割
            lines = [l.strip().strip('"\'') for l in resp.split('\n') if l.strip()]
            return lines[:n_variants]

    def assemble(self, components: Dict[str, str]) -> str:
        """组装最优 prompt."""
        return self._call(
            ASSEMBLE_SYSTEM,
            ASSEMBLE_USER.format(
                subject=components.get("subject", ""),
                scene=components.get("scene", ""),
                motion=components.get("motion", ""),
                camera=components.get("camera", ""),
                style=components.get("style", ""),
            ),
            temperature=0.3,
            max_tokens=512,
        )


# ─────────────────────────────────────────────────────────────
# CLIP Scorer
# ─────────────────────────────────────────────────────────────

class CLIPScorer:
    """CLIP 视觉-文本对齐评分."""

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.model = None
        self.processor = None

    def _lazy_load(self):
        if self.model is None:
            # 可选: 缓存到本地避免重复下载
            model_name = "openai/clip-vit-base-patch32"
            self.model = CLIPModel.from_pretrained(model_name).to(self.device)
            self.processor = CLIPProcessor.from_pretrained(model_name)
            self.model.eval()
            logger.info(f"  [CLIPScorer] loaded {model_name}")

    def _extract_frames(self, video_path: str, num_frames: int = 8) -> List[Image.Image]:
        """从视频提取均匀分布的帧."""
        # 复用 vlm_client 的提取函数
        from .vlm_client import _extract_frames
        return _extract_frames(video_path, num_frames)

    @staticmethod
    def _load_video_tensor(video_path: str, num_frames: int = 8) -> torch.Tensor:
        """用 decord 加载视频帧为 tensor (C,F,H,W)."""
        try:
            from decord import VideoReader, cpu
            vr = VideoReader(video_path, ctx=cpu(0))
            total = len(vr)
            indices = np.linspace(0, total - 1, num_frames, dtype=int)
            frames = vr.get_batch(indices).asnumpy()  # (F, H, W, C)
            frames = torch.from_numpy(frames).permute(3, 0, 1, 2).float()  # (C, F, H, W)
            # 归一化到 [0, 1]
            frames = frames / 255.0
            return frames
        except ImportError:
            pass
        # fallback: PIL images
        from .vlm_client import _extract_frames
        pil_frames = _extract_frames(video_path, num_frames)
        tensors = []
        for img in pil_frames:
            arr = np.array(img.resize((224, 224))).transpose(2, 0, 1)
            tensors.append(torch.from_numpy(arr).float() / 255.0)
        return torch.stack(tensors, dim=1)  # (C, F, H, W)

    def score_text(self, prompt: str, video_path: str,
                   num_frames: int = 8) -> float:
        """
        计算 CLIP 文本-视频对齐分数。

        Returns:
            cosine_similarity ∈ [-1, 1], 越高越好
        """
        self._lazy_load()
        pil_frames = self._extract_frames(video_path, num_frames)
        if not pil_frames:
            logger.warning(f"  [CLIPScorer] 无法从 {video_path} 提取帧")
            return 0.0

        with torch.no_grad():
            # 文本特征
            text_inputs = self.processor(
                text=[prompt], return_tensors="pt", padding=True, truncation=True
            ).to(self.device)
            text_features = self.model.get_text_features(**text_inputs)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            # 图像特征
            frame_features = []
            for frame in pil_frames:
                img_inputs = self.processor(
                    images=frame, return_tensors="pt"
                ).to(self.device)
                feat = self.model.get_image_features(**img_inputs)
                frame_features.append(feat)

            # 平均池化所有帧 → 视频级特征
            video_features = torch.stack(frame_features).mean(dim=0, keepdim=True)
            video_features = video_features / video_features.norm(dim=-1, keepdim=True)

            score = (text_features @ video_features.T).item()

        return score

    def score_variants(self, variants: List[str], video_path: str,
                       num_frames: int = 8) -> List[Tuple[str, float]]:
        """批量评分多个变体, 返回排序后的 (text, score) 列表."""
        scored = []
        for variant in variants:
            score = self.score_text(variant, video_path, num_frames)
            scored.append((variant, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored


# ─────────────────────────────────────────────────────────────
# Main Pipeline
# ─────────────────────────────────────────────────────────────

class PromptDecomposer:
    """结构化 Prompt 分解 + CLIPScore 择优."""

    # 组件描述 (用于 LLM variant 生成)
    COMPONENT_DESCRIPTIONS = {
        "subject": "the main subjects/objects and their appearance",
        "scene": "the background environment, setting, and atmosphere",
        "motion": "actions, movements, gestures, speed, direction",
        "camera": "camera shot type, angle, and movement",
        "style": "visual style, color palette, lighting, mood",
    }

    # 组件的 CLIP 权重 (subject > scene > style > camera > motion)
    COMPONENT_WEIGHTS = {
        "subject": 2.0,
        "scene": 1.5,
        "style": 1.0,
        "camera": 0.8,
        "motion": 0.2,  # motion 由 SVD 处理，CLIP 难以评估
    }

    def __init__(self, llm_client: LLMClient, clip_scorer: CLIPScorer):
        self.llm = llm_client
        self.clip = clip_scorer

    def optimize(self, original_prompt: str, video_path: str,
                 n_variants: int = 3) -> str:
        """
        完整优化流程。

        Returns:
            优化后的最终 prompt
        """
        logger.info(f"  [PromptDecompose] 开始优化, 原文 {len(original_prompt)} chars")

        # Step 1: 分解
        try:
            components = self.llm.decompose(original_prompt)
        except Exception as e:
            logger.warning(f"  [PromptDecompose] LLM分解失败: {e}, 返回原文")
            return original_prompt

        logger.info(
            f"  [PromptDecompose] 分解为 {len(components)} 组件: "
            f"{', '.join(components.keys())}"
        )

        # Step 2: 每个组件生成变体 + CLIP 择优
        best_components = {}
        for comp_name, comp_desc in self.COMPONENT_DESCRIPTIONS.items():
            current_text = components.get(comp_name, "not specified")
            if not current_text or current_text.lower() in ("not specified", "none", ""):
                best_components[comp_name] = current_text
                continue

            # 获取变体
            try:
                variants = self.llm.generate_variants(
                    comp_name, comp_desc, original_prompt, current_text, n_variants
                )
            except Exception as e:
                logger.warning(f"  [PromptDecompose] {comp_name} 变体生成失败: {e}")
                best_components[comp_name] = current_text
                continue

            # 加入原版一起评选
            all_candidates = [current_text] + variants

            # CLIP 评分 (只对 subject/scene/style 做评分, motion/camera 跳过)
            if comp_name in ("subject", "scene", "style"):
                scored = self.clip.score_variants(all_candidates, video_path)
                best_text, best_score = scored[0]
                logger.info(
                    f"  [PromptDecompose] {comp_name}: "
                    f"best=\"{best_text[:60]}...\" (score={best_score:.3f})"
                )
            else:
                # motion/camera: 直接用原版 (CLIP 对运动不敏感)
                best_text = current_text

            best_components[comp_name] = best_text

        # Step 3: 组装最终 prompt
        try:
            final_prompt = self.llm.assemble(best_components)
        except Exception as e:
            logger.warning(f"  [PromptDecompose] 组装失败: {e}")
            # fallback: 简单拼接
            parts = [v for k, v in best_components.items() if v and v != "not specified"]
            final_prompt = ". ".join(parts)

        if final_prompt != original_prompt:
            logger.info(
                f"  [PromptDecompose] 优化完成: "
                f"{len(original_prompt)}→{len(final_prompt)} chars"
            )

        return final_prompt


def create_prompt_decomposer(
    api_key: str,
    api_base: str = "https://token-plan-cn.xiaomimimo.com/v1",
    model: str = "mimo-v2.5-pro",
    device: str = "cuda",
) -> PromptDecomposer:
    """工厂函数: 创建 PromptDecomposer 实例."""
    if not api_key:
        raise ValueError("LLM API key required. Set --llm_api_key or LLM_API_KEY env var.")

    llm = LLMClient(api_key=api_key, api_base=api_base, model=model)
    clip = CLIPScorer(device=device)
    return PromptDecomposer(llm, clip)
