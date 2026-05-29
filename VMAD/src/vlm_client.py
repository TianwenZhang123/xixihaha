"""
VLM Client for VMAD: Video Captioning + Motion Text Decoding.

扩展 P-Flow 的 VLM 客户端，增加 Motion Text Decoding 功能:
    - describe_video(): 生成结构化 caption (复用 P-Flow)
    - decode_motion_text(): VLM 对比解码，提取 delta_e 编码的运动信息

Motion Text Decoding 核心思路:
    1. 用同一噪声先验，分别生成 V_with (有 delta_e) 和 V_without (无 delta_e)
    2. 将两个视频送给 VLM，prompt 要求只描述运动差异
    3. VLM 输出的文字 = delta_e 编码的运动信息的"人话版本"

    优势: 因果性 (加了 delta_e 后多了什么运动) > 相关性 (视频里有什么运动)

参考:
    - P-Flow vlm_client.py: 基础 VLM 架构 (Local + DashScope)
    - EDITOR (arXiv 2025): Embedding -> text decoding
    - VGD (ICLR 2025): 视觉引导文本生成
"""

import json
import base64
import os
import io
import time
import logging
import gc
from typing import Optional, Dict, List, Any

import numpy as np

logger = logging.getLogger(__name__)

try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# ═══════════════════════════════════════════════════════════════════════════════
# Frame Extraction Utilities
# ═══════════════════════════════════════════════════════════════════════════════

def _extract_video_frames_numpy(video_path: str, num_frames: int = 8) -> Optional[np.ndarray]:
    """
    Extract uniformly-sampled frames from a video file as numpy array.

    Shared utility used by both local and API VLM clients.
    Tries decord first (fastest), falls back to imageio.

    Args:
        video_path: Path to video file
        num_frames: Number of frames to uniformly sample

    Returns:
        numpy array of shape (N, H, W, 3) in uint8, or None on failure
    """
    if not os.path.exists(video_path):
        return None

    try:
        from decord import VideoReader, cpu
        vr = VideoReader(video_path, ctx=cpu(0))
        total = len(vr)
        indices = np.linspace(0, total - 1, num_frames, dtype=int)
        return vr.get_batch(indices).asnumpy()
    except (ImportError, Exception):
        pass

    try:
        import imageio.v3 as iio
        all_frames = iio.imread(video_path, plugin="pyav")
        total = len(all_frames)
        indices = np.linspace(0, total - 1, num_frames, dtype=int)
        return all_frames[indices]
    except Exception:
        pass

    return None


def _frames_to_pil(frames: np.ndarray, max_dim: int = 1280) -> List[Any]:
    """
    Convert numpy frames to PIL Images with optional downscaling.

    Args:
        frames: (N, H, W, 3) uint8 numpy array
        max_dim: Maximum dimension (height or width) for downscaling

    Returns:
        List of PIL Image objects
    """
    from PIL import Image
    pil_frames = []
    for frame in frames:
        img = Image.fromarray(frame)
        if max(img.size) > max_dim:
            ratio = max_dim / max(img.size)
            new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
            img = img.resize(new_size, Image.LANCZOS)
        pil_frames.append(img)
    return pil_frames


def _frames_to_base64(frames: np.ndarray, max_dim: int = 1280, quality: int = 85) -> List[str]:
    """
    Convert numpy frames to base64-encoded JPEG strings.

    Args:
        frames: (N, H, W, 3) uint8 numpy array
        max_dim: Maximum dimension for downscaling
        quality: JPEG quality (1-100)

    Returns:
        List of base64-encoded JPEG strings
    """
    from PIL import Image
    frames_b64 = []
    for frame in frames:
        img = Image.fromarray(frame)
        if max(img.size) > max_dim:
            ratio = max_dim / max(img.size)
            new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
            img = img.resize(new_size, Image.LANCZOS)
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=quality)
        frames_b64.append(base64.b64encode(buffer.getvalue()).decode("utf-8"))
    return frames_b64


# ═══════════════════════════════════════════════════════════════════════════════
# Motion Text Decoding Instruction
# ═══════════════════════════════════════════════════════════════════════════════

MOTION_DECODE_INSTRUCTION = """You are a professional video motion analyst. You will receive key frames from TWO videos generated with the same initial noise but different text conditioning.

Video A (top/first set of frames): Generated WITHOUT the motion token
Video B (bottom/second set of frames): Generated WITH the motion token

Your task: Describe ONLY the motion differences between Video B and Video A.

Focus exclusively on:
- Speed changes (acceleration, deceleration, constant speed)
- Direction changes (turning, reversing, curving)
- Trajectory patterns (linear, circular, zigzag, oscillating)
- Rhythm and timing (sudden vs gradual, periodic vs aperiodic)
- Camera movement differences (pan, zoom, tilt, shake)
- Amplitude differences (larger/smaller movements)

IGNORE completely:
- Content/appearance differences (colors, textures, objects)
- Scene/background differences
- Style/quality differences

Output ONLY a concise paragraph (50-80 words) describing what additional motion Video B has compared to Video A. Start directly with the motion description, no preamble."""

CAPTION_INSTRUCTION = """You are a professional text-to-video prompt engineer. Write a structured prompt for a text-to-video model based on the given video frames. Output ONLY the prompt text as ONE continuous paragraph. No labels, no JSON, no extra text."""

STRUCTURED_CAPTION_REQUEST = """Watch these video frames carefully and write a structured text-to-video prompt in English. Follow this exact content order:
1. SUBJECT: who/what the main subject(s) are, their appearance and count.
2. ACTION: what motion/action is happening, movement direction, speed, gestures.
3. SCENE: environment, background, lighting, time of day, weather.
4. CAMERA: shot type (close-up/medium/wide), angle (low/high/eye-level), movement (static/pan/zoom/tracking).
5. STYLE: color palette, mood, atmosphere, visual quality.

STRICT RULES:
- Output ONLY the prompt text as ONE continuous paragraph. No labels, no JSON, no extra text.
- Total length: strictly 80-100 English words. Count carefully before outputting.
- Put subject and action in the FIRST 40 words (highest priority for the model).
- Be specific and vivid (e.g. 'a golden retriever sprinting left to right across green grass' not 'a dog moving')."""


# ═══════════════════════════════════════════════════════════════════════════════
# Local VLM Client (Qwen2.5-VL-7B)
# ═══════════════════════════════════════════════════════════════════════════════

class VMADVLMClient:
    """
    VMAD VLM 客户端 (本地 Qwen2.5-VL-7B)。

    功能:
        1. describe_video(): 生成结构化 caption
        2. decode_motion_text(): VLM 对比解码运动文本
    """

    def __init__(
        self,
        model_path: str = "/root/models/Qwen2.5-VL-7B-Instruct",
        temperature: float = 0.7,
        max_tokens: int = 2048,
        max_retries: int = 3,
        device: str = "cuda",
        lazy_load: bool = True,
    ):
        self.model_path = model_path
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.device = device
        self.lazy_load = lazy_load

        self._model = None
        self._processor = None
        self._loaded = False

        if not lazy_load:
            self._load_model()

    def _load_model(self):
        """加载 Qwen2.5-VL 模型。"""
        if self._loaded:
            return

        if not HAS_TORCH:
            raise ImportError("torch required for local VLM")

        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

        logger.info(f"Loading VLM from {self.model_path}...")
        load_start = time.time()

        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self._processor = AutoProcessor.from_pretrained(self.model_path)
        self._loaded = True

        logger.info(f"VLM loaded in {time.time() - load_start:.1f}s")

    def unload_model(self):
        """释放 GPU 显存。"""
        if self._loaded:
            del self._model
            del self._processor
            self._model = None
            self._processor = None
            self._loaded = False
            if HAS_TORCH:
                torch.cuda.empty_cache()
            gc.collect()
            logger.info("VLM unloaded")

    def describe_video(self, video_path: str) -> str:
        """
        生成结构化 caption (复用 P-Flow 的 prompt 工程)。

        Args:
            video_path: 视频文件路径

        Returns:
            结构化 caption (英文, 80-100 词)
        """
        self._load_model()

        frames_pil = self._extract_frames(video_path, num_frames=16)
        if not frames_pil:
            return ""

        content_list = [{"type": "image", "image": img} for img in frames_pil]
        content_list.append({"type": "text", "text": STRUCTURED_CAPTION_REQUEST})

        messages = [
            {"role": "system", "content": [{"type": "text", "text": CAPTION_INSTRUCTION}]},
            {"role": "user", "content": content_list},
        ]

        for attempt in range(self.max_retries):
            try:
                return self._generate(messages).strip()
            except Exception as e:
                logger.warning(f"describe_video failed (attempt {attempt+1}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(1)
        return ""

    def decode_motion_text(
        self,
        video_with_path: str,
        video_without_path: str,
        compare_prompt: Optional[str] = None,
    ) -> str:
        """
        Motion Text Decoding: VLM 对比解码。

        对比有/无 delta_e 生成的两个视频，提取运动差异描述。

        Args:
            video_with_path: 有 delta_e 生成的视频路径
            video_without_path: 无 delta_e 生成的视频路径
            compare_prompt: 自定义对比 prompt (可选)

        Returns:
            运动差异描述文本
        """
        self._load_model()

        # 提取两个视频的帧
        frames_without = self._extract_frames(video_without_path, num_frames=8)
        frames_with = self._extract_frames(video_with_path, num_frames=8)

        if not frames_without or not frames_with:
            logger.warning("Failed to extract frames for motion text decoding")
            return ""

        # 构建对比输入: 先放 without 的帧，再放 with 的帧
        content_list = []

        # Video A (without delta_e)
        content_list.append({"type": "text", "text": "Video A frames (WITHOUT motion token):"})
        for img in frames_without:
            content_list.append({"type": "image", "image": img})

        # Video B (with delta_e)
        content_list.append({"type": "text", "text": "Video B frames (WITH motion token):"})
        for img in frames_with:
            content_list.append({"type": "image", "image": img})

        # 对比指令
        instruction = compare_prompt or MOTION_DECODE_INSTRUCTION
        content_list.append({"type": "text", "text": "Now describe the motion differences:"})

        messages = [
            {"role": "system", "content": [{"type": "text", "text": instruction}]},
            {"role": "user", "content": content_list},
        ]

        for attempt in range(self.max_retries):
            try:
                result = self._generate(messages).strip()
                logger.info(f"  [MotionDecode] Result: {result[:100]}...")
                return result
            except Exception as e:
                logger.warning(f"decode_motion_text failed (attempt {attempt+1}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(1)

        return ""

    def _generate(self, messages: List[Dict]) -> str:
        """本地模型推理。"""
        from qwen_vl_utils import process_vision_info

        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self._processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self._model.device)

        with torch.no_grad():
            generated_ids = self._model.generate(
                **inputs,
                max_new_tokens=self.max_tokens,
                temperature=self.temperature,
                do_sample=self.temperature > 0,
            )

        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        return self._processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

    def _extract_frames(self, video_path: str, num_frames: int = 8) -> List[Any]:
        """从视频中均匀提取帧 (PIL Image)。"""
        frames = _extract_video_frames_numpy(video_path, num_frames)
        if frames is None:
            return []
        return _frames_to_pil(frames)


# ═══════════════════════════════════════════════════════════════════════════════
# DashScope API Client
# ═══════════════════════════════════════════════════════════════════════════════

class VMADVLMClientAPI:
    """
    VMAD VLM 客户端 (DashScope API 版本)。

    适用于没有本地 GPU 运行 VLM 的场景。
    """

    def __init__(
        self,
        model_name: str = "qwen-vl-max",
        api_key: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        max_retries: int = 3,
    ):
        if not HAS_OPENAI:
            raise ImportError("openai package required")

        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries

        api_key = api_key or os.environ.get("DASHSCOPE_API_KEY")
        if not api_key:
            raise ValueError("DASHSCOPE_API_KEY required")

        self.client = openai.OpenAI(
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )

    def describe_video(self, video_path: str) -> str:
        """生成结构化 caption (API 版本)。"""
        frames_b64 = self._extract_frames_base64(video_path, num_frames=16)
        if not frames_b64:
            return ""

        content = [{"type": "text", "text": STRUCTURED_CAPTION_REQUEST}]
        for fb64 in frames_b64:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{fb64}"}
            })

        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": CAPTION_INSTRUCTION},
                        {"role": "user", "content": content},
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                logger.warning(f"API describe_video failed (attempt {attempt+1}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
        return ""

    def decode_motion_text(
        self,
        video_with_path: str,
        video_without_path: str,
        compare_prompt: Optional[str] = None,
    ) -> str:
        """Motion Text Decoding (API 版本)。"""
        frames_without = self._extract_frames_base64(video_without_path, num_frames=8)
        frames_with = self._extract_frames_base64(video_with_path, num_frames=8)

        if not frames_without or not frames_with:
            return ""

        instruction = compare_prompt or MOTION_DECODE_INSTRUCTION

        content = [{"type": "text", "text": "Video A (WITHOUT motion token):"}]
        for fb64 in frames_without:
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{fb64}"}})

        content.append({"type": "text", "text": "Video B (WITH motion token):"})
        for fb64 in frames_with:
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{fb64}"}})

        content.append({"type": "text", "text": "Describe the motion differences:"})

        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": instruction},
                        {"role": "user", "content": content},
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                logger.warning(f"API decode_motion_text failed (attempt {attempt+1}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
        return ""

    def _extract_frames_base64(self, video_path: str, num_frames: int = 8) -> List[str]:
        """提取帧并编码为 base64。"""
        frames = _extract_video_frames_numpy(video_path, num_frames)
        if frames is None:
            return []
        return _frames_to_base64(frames)

    def unload_model(self):
        """API 模式无需释放。"""
        pass


# ═══════════════════════════════════════════════════════════════════════════════
# Mock Client (测试用)
# ═══════════════════════════════════════════════════════════════════════════════

class MockVLMClient:
    """Mock VLM 客户端，用于无 GPU/API 环境下的测试。"""

    def describe_video(self, video_path: str) -> str:
        return (
            "A subject moves dynamically through the scene with natural motion. "
            "The camera captures the action in a medium shot with smooth tracking. "
            "Warm natural lighting illuminates the scene with soft shadows and "
            "a balanced color palette creating a cinematic atmosphere."
        )

    def decode_motion_text(
        self, video_with_path: str, video_without_path: str, **kwargs
    ) -> str:
        return (
            "Video B shows additional forward acceleration with a slight rightward "
            "drift compared to Video A. The movement transitions from steady pace "
            "to a burst of speed around the midpoint, with subtle camera tracking "
            "that follows the subject more closely."
        )

    def unload_model(self):
        pass


# ═══════════════════════════════════════════════════════════════════════════════
# Factory
# ═══════════════════════════════════════════════════════════════════════════════

def create_vlm_client(config: Dict[str, Any]) -> Any:
    """
    工厂函数: 根据配置创建 VLM 客户端。

    Args:
        config: dict with keys:
            - provider: "local" | "dashscope" | "mock"
            - model_path: 本地模型路径
            - temperature, max_tokens, max_retries, etc.

    Returns:
        VLM 客户端实例
    """
    provider = config.get("provider", "local")

    if provider == "mock":
        return MockVLMClient()

    if provider == "local":
        try:
            return VMADVLMClient(
                model_path=config.get("model_path", "/root/models/Qwen2.5-VL-7B-Instruct"),
                temperature=config.get("temperature", 0.7),
                max_tokens=config.get("max_tokens", 2048),
                max_retries=config.get("max_retries", 3),
                device=config.get("device", "cuda"),
                lazy_load=config.get("lazy_load", True),
            )
        except Exception as e:
            logger.warning(f"Local VLM init failed: {e}, using Mock")
            return MockVLMClient()

    if provider == "dashscope":
        try:
            return VMADVLMClientAPI(
                model_name=config.get("model_name", "qwen-vl-max"),
                api_key=config.get("api_key"),
                temperature=config.get("temperature", 0.7),
                max_tokens=config.get("max_tokens", 2048),
                max_retries=config.get("max_retries", 3),
            )
        except Exception as e:
            logger.warning(f"API VLM init failed: {e}, using Mock")
            return MockVLMClient()

    return MockVLMClient()
