#!/usr/bin/env python3
"""
环境依赖检查脚本 - 一键确认所有依赖是否就绪。

Usage:
    python scripts/check_env.py

Target: 4090 (24GB) + Wan2.1-1.3B
检查：
1. Python / PyTorch / CUDA 基础环境
2. Wan2.1-T2V-1.3B 模型文件完整性
3. Pipeline 核心依赖（diffusers, transformers 等）
4. 评测依赖（scikit-image, CLIP, decord 等）
5. VLM 依赖（DashScope API Key）
6. 磁盘空间
"""

import os
import sys
import shutil
from pathlib import Path

# ============================================================
# 输出格式
# ============================================================
def ok(msg):
    print(f"  ✅ {msg}")

def warn(msg):
    print(f"  ⚠️  {msg}")

def fail(msg):
    print(f"  ❌ {msg}")

def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

results = {"pass": [], "warn": [], "fail": []}

def record_ok(item):
    results["pass"].append(item)
    ok(item)

def record_warn(item):
    results["warn"].append(item)
    warn(item)

def record_fail(item):
    results["fail"].append(item)
    fail(item)

# ============================================================
# 1. Python / PyTorch / CUDA
# ============================================================
section("1. Python / PyTorch / CUDA 基础环境")

print(f"  Python: {sys.version}")
if sys.version_info >= (3, 10):
    record_ok(f"Python {sys.version_info.major}.{sys.version_info.minor}")
else:
    record_warn(f"Python {sys.version_info.major}.{sys.version_info.minor} (建议 3.10+)")

try:
    import torch
    record_ok(f"PyTorch {torch.__version__}")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_mem / 1024**3
        record_ok(f"CUDA 可用: {gpu_name} ({gpu_mem:.0f}GB)")
        record_ok(f"CUDA 版本: {torch.version.cuda}")
        # 1.3B 只需要 ~12-16GB VRAM，4090 (24GB) 绰绰有余
        if gpu_mem >= 20:
            record_ok(f"显存 {gpu_mem:.0f}GB，1.3B 模型完全够用")
        elif gpu_mem >= 12:
            record_warn(f"显存 {gpu_mem:.0f}GB，1.3B 可跑但较紧张")
        else:
            record_fail(f"显存 {gpu_mem:.0f}GB，可能不够跑 1.3B 生成 81 帧")
    else:
        record_fail("CUDA 不可用!")

    # bfloat16 support
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        record_ok("bfloat16 支持")
    else:
        record_warn("bfloat16 不支持，可能需要 float16")
except ImportError:
    record_fail("PyTorch 未安装!")

# ============================================================
# 2. Wan2.1-T2V-1.3B 模型文件完整性
# ============================================================
section("2. Wan2.1-T2V-1.3B 模型文件")

model_path = Path("/root/autodl-tmp/models/Wan2.1-T2V-1.3B-Diffusers")

if model_path.exists():
    record_ok(f"模型目录存在: {model_path}")

    # 检查关键子目录/文件
    expected_dirs = ["transformer", "text_encoder", "vae", "scheduler", "tokenizer"]
    for d in expected_dirs:
        p = model_path / d
        if p.exists():
            size_gb = sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 1024**3
            record_ok(f"  {d}/ ({size_gb:.1f}GB)")
        else:
            record_fail(f"  {d}/ 缺失!")

    # 检查 model_index.json
    if (model_path / "model_index.json").exists():
        record_ok("  model_index.json 存在")
    else:
        record_fail("  model_index.json 缺失!")

    # 总大小
    total_size = sum(f.stat().st_size for f in model_path.rglob("*") if f.is_file()) / 1024**3
    print(f"  📦 模型总大小: {total_size:.1f}GB")
else:
    record_fail(f"模型目录不存在: {model_path}")
    print("    尝试查找其他可能的模型路径...")
    alt_paths = [
        Path("/root/autodl-tmp/models"),
        Path("/root/models"),
        Path("/data/models"),
    ]
    for ap in alt_paths:
        if ap.exists():
            contents = list(ap.iterdir())
            print(f"    {ap}: {[c.name for c in contents[:10]]}")

# ============================================================
# 3. Pipeline 核心依赖
# ============================================================
section("3. Pipeline 核心依赖")

core_deps = {
    "diffusers": "0.31.0",
    "transformers": "4.44.0",
    "accelerate": None,
    "safetensors": None,
    "numpy": None,
    "tqdm": None,
    "pyyaml": "yaml",
    "Pillow": "PIL",
    "imageio": None,
    "einops": None,
}

for pkg_name, import_name in core_deps.items():
    mod_name = import_name if import_name else pkg_name
    try:
        mod = __import__(mod_name)
        version = getattr(mod, "__version__", "?")
        record_ok(f"{pkg_name} {version}")
    except ImportError:
        record_fail(f"{pkg_name} 未安装!")

# ============================================================
# 4. 评测依赖
# ============================================================
section("4. 评测依赖")

# scikit-image (SSIM)
try:
    import skimage
    record_ok(f"scikit-image {skimage.__version__} (SSIM 指标)")
except ImportError:
    record_fail("scikit-image 未安装! (SSIM 指标需要)")
    print("    pip install scikit-image")

# CLIP (语义相似度 + Prompt-Video Alignment)
clip_available = False
try:
    import clip
    record_ok("openai-clip 可用 (CLIP-Similarity 指标)")
    clip_available = True
except ImportError:
    try:
        from transformers import CLIPModel, CLIPProcessor
        record_ok("transformers CLIP 可用 (CLIP-Similarity 指标，HF后端)")
        clip_available = True
    except ImportError:
        record_fail("CLIP 不可用! (CLIP-Similarity + Prompt-Video Alignment 需要)")
        print("    pip install git+https://github.com/openai/CLIP.git")

# decord (高性能视频读取)
try:
    import decord
    record_ok(f"decord {decord.__version__} (高性能视频读取)")
except ImportError:
    record_warn("decord 未安装 (将 fallback 到 imageio，速度较慢)")
    print("    pip install eva-decord")

# imageio (备选视频读取)
try:
    import imageio
    record_ok(f"imageio {imageio.__version__} (备选视频读取)")
    try:
        import av
        record_ok(f"  PyAV {av.__version__} (imageio 视频插件)")
    except ImportError:
        record_warn("  PyAV 未安装 (imageio 视频读取可能受限)")
        print("      pip install av")
except ImportError:
    record_warn("imageio 未安装")

# torchvision
try:
    import torchvision
    record_ok(f"torchvision {torchvision.__version__}")
except ImportError:
    record_warn("torchvision 未安装")

# scipy
try:
    import scipy
    record_ok(f"scipy {scipy.__version__} (FVD 计算)")
except ImportError:
    record_fail("scipy 未安装! (FVD 指标需要)")
    print("    pip install scipy")

# lpips
try:
    import lpips
    record_ok(f"lpips {lpips.__version__} (感知相似度)")
except ImportError:
    record_warn("lpips 未安装 (可选)")

# ============================================================
# 5. VLM 依赖 (DashScope)
# ============================================================
section("5. VLM 依赖 (Qwen-VL via DashScope)")

try:
    import openai
    record_ok(f"openai SDK {openai.__version__}")
except ImportError:
    record_fail("openai SDK 未安装!")
    print("    pip install openai")

# API Key
api_key = os.environ.get("DASHSCOPE_API_KEY", "")
if api_key:
    masked = api_key[:8] + "..." + api_key[-4:]
    record_ok(f"DASHSCOPE_API_KEY 已设置 ({masked})")
else:
    record_warn("DASHSCOPE_API_KEY 环境变量未设置 (VLM 调用需要)")
    print("    export DASHSCOPE_API_KEY='sk-xxxxxxx'")

# ============================================================
# 6. 磁盘空间
# ============================================================
section("6. 磁盘空间")

paths_to_check = [
    ("/root/autodl-tmp", "数据盘"),
    ("/", "系统盘"),
]

for path, label in paths_to_check:
    if os.path.exists(path):
        usage = shutil.disk_usage(path)
        free_gb = usage.free / 1024**3
        total_gb = usage.total / 1024**3
        used_pct = (usage.used / usage.total) * 100
        if free_gb > 20:
            record_ok(f"{label} ({path}): {free_gb:.1f}GB 可用 / {total_gb:.0f}GB 总计 ({used_pct:.0f}% 已用)")
        elif free_gb > 5:
            record_warn(f"{label} ({path}): {free_gb:.1f}GB 可用 (偏低)")
        else:
            record_fail(f"{label} ({path}): {free_gb:.1f}GB 可用 (不足!)")

# ============================================================
# 汇总
# ============================================================
section("汇总")

print(f"\n  ✅ 通过: {len(results['pass'])} 项")
print(f"  ⚠️  警告: {len(results['warn'])} 项")
print(f"  ❌ 失败: {len(results['fail'])} 项")

if results["fail"]:
    print(f"\n  {'='*50}")
    print(f"  必须修复的问题:")
    print(f"  {'='*50}")
    for item in results["fail"]:
        print(f"    ❌ {item}")

if results["warn"]:
    print(f"\n  建议修复 (非必须):")
    for item in results["warn"]:
        print(f"    ⚠️  {item}")

# 生成一键安装命令
if results["fail"]:
    print(f"\n  {'='*50}")
    print(f"  一键安装缺失依赖 (复制执行):")
    print(f"  {'='*50}")

    install_cmds = []
    fail_str = " ".join(results["fail"]).lower()

    if "scikit-image" in fail_str:
        install_cmds.append("scikit-image")
    if "clip" in fail_str:
        install_cmds.append("git+https://github.com/openai/CLIP.git")
    if "openai" in fail_str:
        install_cmds.append("openai")
    if "diffusers" in fail_str:
        install_cmds.append("diffusers>=0.31.0")
    if "transformers" in fail_str:
        install_cmds.append("transformers>=4.44.0")
    if "scipy" in fail_str:
        install_cmds.append("scipy")
    if "imageio" in fail_str:
        install_cmds.append("imageio[pyav]")

    if install_cmds:
        print(f"\n    pip install {' '.join(install_cmds)}")

    if "DASHSCOPE_API_KEY" in " ".join(results["fail"]):
        print(f"\n    export DASHSCOPE_API_KEY='your-api-key-here'")

print(f"\n{'='*60}")
if not results["fail"]:
    print("  🎉 环境就绪！可以开始跑实验了。")
    print(f"    python run.py --video <参考视频路径> --auto_prompt --i_max 5 --alpha 0.1")
else:
    print("  ⚡ 修复上述问题后重新运行本脚本确认。")
print(f"{'='*60}\n")
