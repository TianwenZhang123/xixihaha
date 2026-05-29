"""
Content Augmentation for Cross-Content Consistency (Optional Module).

This module is used ONLY in motion transfer mode (λ_dis > 0) to generate
diverse content prompts for the Cross-Content Consistency Loss. It is NOT
needed for faithful video reproduction (the default operating mode).

Purpose (Motion Transfer Mode):
    Generate N content-varied versions of the original caption to compute
    L_dis = Var_i(v_θ(x_t, t, Enc(p_i) + Δe)). This loss ensures Δe
    encodes only motion information, not content-specific features.

    Original caption: "A golden retriever running on the beach"
    Augmented prompts:
        - "A black cat running on the beach"
        - "A small child running on the beach"
        - "A white horse running on the beach"

    Rule: Replace ONLY the subject (WHAT is moving), preserve all motion/scene.

When to use:
    - Motion transfer mode (T_m=0.3, λ_dis=0.1): REQUIRED for disentanglement
    - Reproduction mode (T_m=1.0, λ_dis=0.0): NOT NEEDED (skip this module)

References:
    - Textual Inversion: Diverse prompt templates for training
    - DreamBooth: Class-prior preservation
    - Domain Adaptation: Cross-domain consistency regularization
"""

import logging
import json
import re
from typing import List, Optional, Dict, Any

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
# 预定义主体替换库 (Mock 模式使用)
# ═══════════════════════════════════════════════════════════════════════════════

SUBJECT_POOL = [
    # 动物
    "a black cat", "a white rabbit", "a brown horse", "a gray wolf",
    "a red fox", "a blue parrot", "a green turtle", "a yellow butterfly",
    # 人物
    "a young child", "an elderly man", "a woman in a red dress",
    "a dancer", "an athlete", "a robot",
    # 物体
    "a red sports car", "a blue bicycle", "a paper airplane",
    "a bouncing ball", "a floating balloon", "a rolling wheel",
]

# ═══════════════════════════════════════════════════════════════════════════════
# LLM Augmentation Instruction
# ═══════════════════════════════════════════════════════════════════════════════

AUGMENTATION_INSTRUCTION = """You are a prompt augmentation expert. Your task is to create content-varied versions of a video description prompt.

RULES:
1. Replace ONLY the main subject (the entity performing the action) with a different entity
2. Keep ALL motion/action descriptions EXACTLY the same
3. Keep ALL scene/environment descriptions EXACTLY the same
4. Keep ALL camera/style descriptions EXACTLY the same
5. The new subject should be physically capable of performing the described motion
6. Output ONLY a JSON array of strings, no explanation

EXAMPLE:
Input: "A golden retriever sprinting left to right across a grassy field, with the camera tracking smoothly"
Output: ["A black cat sprinting left to right across a grassy field, with the camera tracking smoothly", "A small child sprinting left to right across a grassy field, with the camera tracking smoothly", "A white horse sprinting left to right across a grassy field, with the camera tracking smoothly", "A red fox sprinting left to right across a grassy field, with the camera tracking smoothly", "A robot sprinting left to right across a grassy field, with the camera tracking smoothly"]

Now augment this prompt (generate {n} variants):
"{caption}"
"""


class ContentAugmenter:
    """
    内容增强器: 生成用于解耦训练的多样化 prompt。

    支持三种模式:
        - mock: 使用规则替换 (无需 LLM，适合测试)
        - local: 本地 LLM 推理
        - dashscope: 远程 API 调用
    """

    def __init__(
        self,
        provider: str = "mock",
        num_augmentations: int = 5,
        api_key: Optional[str] = None,
        model_name: str = "qwen-plus",
    ):
        """
        Args:
            provider: "mock" | "local" | "dashscope"
            num_augmentations: 生成的增强 prompt 数量
            api_key: DashScope API key (provider="dashscope" 时需要)
            model_name: LLM 模型名称
        """
        self.provider = provider
        self.num_augmentations = num_augmentations
        self.api_key = api_key
        self.model_name = model_name

    def augment(self, caption: str, n: Optional[int] = None) -> List[str]:
        """
        生成 N 个内容增强 prompt。

        Args:
            caption: 原始 caption
            n: 生成数量 (默认使用 self.num_augmentations)

        Returns:
            增强 prompt 列表 (不包含原始 caption)
        """
        n = n or self.num_augmentations

        if self.provider == "mock":
            return self._augment_mock(caption, n)
        elif self.provider == "dashscope":
            return self._augment_dashscope(caption, n)
        elif self.provider == "local":
            return self._augment_local(caption, n)
        else:
            logger.warning(f"Unknown provider '{self.provider}', using mock")
            return self._augment_mock(caption, n)

    def _augment_mock(self, caption: str, n: int) -> List[str]:
        """
        Mock 模式: 基于规则的主体替换。

        策略:
            1. 尝试识别 caption 中的主体 (第一个名词短语)
            2. 用预定义主体池中的实体替换
            3. 如果识别失败，在 caption 前加主体前缀
        """
        # 简单的主体识别: 找 "a/an/the + adj* + noun" 模式
        subject_pattern = r'\b(a|an|the)\s+(\w+\s+)*?\w+(?=\s+(running|walking|moving|jumping|dancing|flying|swimming|driving|riding|skating|climbing|falling|spinning|turning|sliding|rolling|bouncing|floating|crawling|sprinting|jogging|trotting|galloping|dashing|rushing|strolling|wandering|marching|stepping|leaping|hopping|skipping))'

        match = re.search(subject_pattern, caption, re.IGNORECASE)

        augmented = []
        import random
        random.seed(42)  # 确保可复现
        subjects = random.sample(SUBJECT_POOL, min(n, len(SUBJECT_POOL)))

        if match:
            original_subject = match.group(0)
            for new_subject in subjects[:n]:
                new_caption = caption.replace(original_subject, new_subject, 1)
                if new_caption != caption:
                    augmented.append(new_caption)
        else:
            # Fallback: 尝试替换开头的名词短语
            # 匹配 "A/An/The ... verb-ing" 之前的部分
            parts = re.split(r'\b(running|walking|moving|jumping|dancing|flying|swimming|driving|riding)\b', caption, maxsplit=1)
            if len(parts) >= 2:
                for new_subject in subjects[:n]:
                    new_caption = f"{new_subject} {parts[1]}" + "".join(parts[2:])
                    augmented.append(new_caption)
            else:
                # 最终 fallback: 直接用不同主体 + 原始动作描述
                for new_subject in subjects[:n]:
                    augmented.append(f"{new_subject}, {caption}")

        # 确保返回 n 个
        while len(augmented) < n:
            extra_subject = random.choice(SUBJECT_POOL)
            augmented.append(f"{extra_subject}, {caption}")

        return augmented[:n]

    def _augment_dashscope(self, caption: str, n: int) -> List[str]:
        """
        DashScope API 模式: 使用远程 LLM 生成增强 prompt。
        """
        try:
            import openai
            import os

            api_key = self.api_key or os.environ.get("DASHSCOPE_API_KEY")
            if not api_key:
                logger.warning("No DASHSCOPE_API_KEY, falling back to mock")
                return self._augment_mock(caption, n)

            client = openai.OpenAI(
                api_key=api_key,
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            )

            prompt = AUGMENTATION_INSTRUCTION.format(n=n, caption=caption)

            response = client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.8,
                max_tokens=1024,
            )

            response_text = response.choices[0].message.content.strip()
            return self._parse_augmentation_response(response_text, caption, n)

        except Exception as e:
            logger.warning(f"DashScope augmentation failed: {e}, using mock")
            return self._augment_mock(caption, n)

    def _augment_local(self, caption: str, n: int) -> List[str]:
        """
        本地 LLM 模式: 通过 transformers pipeline 调用本地模型。

        Requires a local causal LM (e.g., Qwen2-7B-Instruct) accessible via
        HuggingFace transformers. Falls back to mock if model unavailable.
        """
        try:
            from transformers import pipeline as hf_pipeline

            generator = hf_pipeline(
                "text-generation",
                model=self.model_name,
                max_new_tokens=512,
                temperature=0.8,
                do_sample=True,
            )
            prompt = AUGMENTATION_INSTRUCTION.format(n=n, caption=caption)
            output = generator(prompt, return_full_text=False)[0]["generated_text"]
            return self._parse_augmentation_response(output, caption, n)

        except Exception as e:
            logger.warning(f"Local LLM augmentation failed: {e}, falling back to mock")
            return self._augment_mock(caption, n)

    def _parse_augmentation_response(
        self, response_text: str, caption: str, n: int
    ) -> List[str]:
        """解析 LLM 返回的 JSON 数组。"""
        # 尝试直接解析
        try:
            result = json.loads(response_text)
            if isinstance(result, list):
                return [str(item) for item in result[:n]]
        except json.JSONDecodeError:
            pass

        # 尝试从 markdown code block 中提取
        json_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', response_text, re.DOTALL)
        if json_match:
            try:
                result = json.loads(json_match.group(1))
                if isinstance(result, list):
                    return [str(item) for item in result[:n]]
            except json.JSONDecodeError:
                pass

        # 尝试找 JSON 数组
        array_match = re.search(r'\[.*\]', response_text, re.DOTALL)
        if array_match:
            try:
                result = json.loads(array_match.group(0))
                if isinstance(result, list):
                    return [str(item) for item in result[:n]]
            except json.JSONDecodeError:
                pass

        # Fallback
        logger.warning("Failed to parse LLM augmentation response, using mock")
        return self._augment_mock(caption, n)
