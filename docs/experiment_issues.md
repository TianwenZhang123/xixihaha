# P-Flow 实验问题记录表

本文档记录了在 AutoDL 4090 上复现 P-Flow 论文实验过程中遇到的所有问题及其解决方案。

---

| # | 错误信息 | 原因 | 解决方案 | 涉及文件 | Commit |
|---|---------|------|---------|----------|--------|
| 1 | `FileNotFoundError: Reference video not found: /root/autodl-tmp/data/moviegen_bench/water_mark_out/23.mp4` | 代码中视频路径拼接了 `water_mark_out/` 子目录，但 MovieGenBench 数据集解压后视频直接在根目录下（`{index}.mp4`） | 修改 `resolve_video_and_prompt()` 中的路径拼接：`os.path.join(dataset_dir, f"{args.video_index}.mp4")` | `scripts/run_pflow_paper.py` | — |
| 2 | `ValueError: DASHSCOPE_API_KEY environment variable is required` | 用户执行了 `echo 'export ...' >> ~/.bashrc` 但未在当前终端生效 | 需要执行 `source ~/.bashrc` 或直接 `export DASHSCOPE_API_KEY="key"` | — | — |
| 3 | `ValueError: ... does not appear to have a file named model_index.json` | 下载了原始 checkpoint 格式（`Wan-AI/Wan2.1-T2V-1.3B`），该格式没有 Diffusers 所需的 `model_index.json` 元数据文件 | 改为下载 Diffusers 格式：`Wan-AI/Wan2.1-T2V-1.3B-Diffusers`（约 27GB） | `config/default.yaml`, `scripts/run_pflow_paper.py` | — |
| 4 | HuggingFace 下载中断：`SSL: UNEXPECTED_EOF_WHILE_READING` | hf-mirror 镜像网络不稳定，大文件下载中途超时 | 重新执行同一下载命令，huggingface-cli 支持断点续传 | — | — |
| 5 | `AttributeError: 'FrozenDict' object has no attribute 'scaling_factor'` | Wan 2.1 的 VAE config 是 `FrozenDict` 类型，不包含 `scaling_factor` 属性（SD 系列 VAE 才有） | 使用 `getattr` 链式回退：先查 `vae.config.scaling_factor`，再查 `pipe.vae_scaling_factor`，最后默认 `0.18215` | `pflow/flow_matching.py` | — |
| 6 | 默认 `--model_path` 仍指向旧路径 `/root/autodl-tmp/models/Wan2.1-T2V-1.3B` | `argparse` 的 `default` 值硬编码为旧路径，用户 `git pull` 后未加 `--model_path` 参数时仍用旧默认值 | 将 argparse default 更新为 `/root/autodl-tmp/models/Wan2.1-T2V-1.3B-Diffusers` | `scripts/run_pflow_paper.py` | — |
| 7 | `TypeError: WanPipeline.encode_prompt() got an unexpected keyword argument 'num_images_per_prompt'` | 代码按 Stable Diffusion Pipeline 的接口调用 `encode_prompt()`，传了 `num_images_per_prompt` 和 `do_classifier_free_guidance`，但 WanPipeline 的签名不同（使用 `num_videos_per_prompt` 等） | 用 `inspect.signature()` 动态检测 `encode_prompt` 支持的参数，只传入兼容的 kwargs | `pflow/pipeline.py` | `d60d1b3` |
| 8 | `RuntimeError: "svd_cuda_gesvdj" not implemented for 'BFloat16'` | Wan 2.1 模型推理使用 BFloat16 精度，Flow Inversion 输出的 latents 也是 BFloat16，但 CUDA 的 SVD 内核（gesvdj）只支持 Float32/Float64 | SVD 计算前 `.float()` 转为 Float32，计算完后 `.to(original_dtype)` 转回原始精度 | `pflow/svd_filter.py` | `113ca59` |

---

## 问题分类统计

| 类别 | 数量 | 问题编号 |
|------|------|---------|
| 模型格式/加载 | 3 | #3, #5, #6 |
| API 接口不兼容 | 2 | #7, #8 |
| 路径/配置 | 2 | #1, #2 |
| 网络/环境 | 1 | #4 |

---

## 根因总结

大部分问题（#3, #5, #7, #8）都源于同一个根因：**代码最初基于 Stable Diffusion Pipeline 的接口规范编写，而 Wan 2.1 Pipeline 作为新一代视频生成模型，在 VAE config 结构、encode_prompt 参数签名、默认推理精度（BFloat16）等方面与 SD 系列存在差异。** 这些差异只有在实际加载模型运行时才会暴露。
