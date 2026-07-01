#!/usr/bin/env python3
"""
统一评测脚本：6 项指标一次性计算。

指标：
  1. CLIP  (orig-gen)
  2. XCLIP (orig-gen)
  3. FVD   (Fréchet Video Distance — 分布级)
  4. LPIPS (逐帧感知距离 — 越低越好)
  5. SSIM  (逐帧结构相似度 — 越高越好)
  6. OF-EPE (Optical Flow Endpoint Error — 运动一致性)

用法:
  python -m evaluation.run_all_metrics \
      --orig-dir data/videos \
      --gen-dir outputs/1.3B_ABC_full \
      --caption-dir /root/xixihaha/test-v200/test-v200/captions \
      --output-dir outputs/1.3B_ABC_full/eval_all
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from skimage.metrics import structural_similarity

from evaluation.clip_utils import (
    build_models,
    cosine_similarity,
    get_clip_text_feature,
    get_clip_video_feature,
    get_xclip_text_feature,
    get_xclip_video_feature,
    read_text,
    sample_video_frames,
)

# ────────────────────────────────────────────
DEFAULT_CLIP_MODEL  = "/root/autodl-tmp/models/clip-vit-base-patch32"
DEFAULT_XCLIP_MODEL = "/root/autodl-tmp/models/xclip-base-patch32"
DEFAULT_SAMPLE_FRAMES = 16  # LPIPS/SSIM/Flow 用 16 帧
CLIP_FRAMES = 8               # CLIP/XCLIP 用 8 帧


def decode_video_frames(video_path: Path) -> list[np.ndarray]:
    import av
    with av.open(str(video_path)) as c:
        return [f.to_rgb().to_ndarray() for f in c.decode(video=0)]


def _resize(frame: np.ndarray, size: int) -> np.ndarray:
    t = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
    t = F.interpolate(t.unsqueeze(0), size=(size, size), mode="bilinear",
                       align_corners=False).squeeze(0)
    return (t.clamp(0, 1) * 255).byte().permute(1, 2, 0).cpu().numpy()


def sample_idx(total: int, num: int) -> np.ndarray:
    return np.linspace(0, total - 1, num=num, dtype=int)


def get_aligned_frames(
    ref_path: Path, gen_path: Path, num_frames: int, frame_size: int,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    ra = decode_video_frames(ref_path)
    ga = decode_video_frames(gen_path)
    ri = sample_idx(len(ra), num_frames)
    gi = sample_idx(len(ga), num_frames)
    ref = [ra[i] for i in ri]
    gen = [ga[i] for i in gi]
    if frame_size > 0:
        ref = [_resize(f, frame_size) for f in ref]
        gen = [_resize(f, frame_size) for f in gen]
    return ref, gen


# ─── CLIP / XCLIP ───

def compute_clip_xclip(item, clip_processor, clip_model,
                       xclip_processor, xclip_model, device):
    caption = read_text(item["caption_path"])
    of = sample_video_frames(item["orig_path"], CLIP_FRAMES)
    gf = sample_video_frames(item["gen_path"], CLIP_FRAMES)

    ct = get_clip_text_feature(caption, clip_processor, clip_model, device)
    co = get_clip_video_feature(of, clip_processor, clip_model, device)
    cg = get_clip_video_feature(gf, clip_processor, clip_model, device)

    xt = get_xclip_text_feature(caption, xclip_processor, xclip_model, device)
    xo = get_xclip_video_feature(of, xclip_processor, xclip_model, device)
    xg = get_xclip_video_feature(gf, xclip_processor, xclip_model, device)

    return {
        "clip_o_g":  float(cosine_similarity(co, cg)),
        "xclip_o_g": float(cosine_similarity(xo, xg)),
    }


# ─── FVD (R3D-18) ───

def _load_r3d18():
    from torchvision.models.video import R3D_18_Weights, r3d_18
    w = R3D_18_Weights.DEFAULT
    m = r3d_18(weights=w)
    m.fc = torch.nn.Identity()
    m.eval()
    return m, w


def _r3d18_encode(video_path: Path, model, weights,
                  num_frames: int, device: str) -> np.ndarray:
    size = weights.transforms().crop_size[0]
    frames = decode_video_frames(video_path)
    idx = sample_idx(len(frames), num_frames)
    imgs = [frames[i] for i in idx]
    t = torch.stack([
        torch.from_numpy(f).permute(2, 0, 1).float() / 255.0
        for f in imgs], dim=1)  # (C, F, H, W)
    # 逐帧 resize 到 (size, size)，避免 5D interpolate 问题
    t = F.interpolate(t.permute(1, 0, 2, 3), size=(size, size),
                       mode="bilinear", align_corners=False).permute(1, 0, 2, 3)
    # 形状变为 (C, F, size, size)
    t = weights.transforms()(t.unsqueeze(0))  # → (1, C, F, size, size)
    with torch.inference_mode():
        feat = model(t.to(device)).cpu().numpy()
    return feat.flatten()


def compute_fvd(ref_dir: Path, gen_dir: Path, num_frames: int, device: str) -> float | None:
    try:
        from scipy.linalg import sqrtm
        model, weights = _load_r3d18()
        model.to(device)
        pairs = list(_discover_pairs(ref_dir, gen_dir))
        ref_feats, gen_feats = [], []
        for _, rp, gp in tqdm(pairs, desc="FVD"):
            ref_feats.append(_r3d18_encode(rp, model, weights, num_frames, device))
            gen_feats.append(_r3d18_encode(gp, model, weights, num_frames, device))
        rf = np.stack(ref_feats)
        gf = np.stack(gen_feats)
        mu_r, mu_g = rf.mean(0), gf.mean(0)
        cov_r = np.cov(rf, rowvar=False, ddof=1)
        cov_g = np.cov(gf, rowvar=False, ddof=1)
        diff = np.sum((mu_r - mu_g) ** 2)
        covmean, _ = sqrtm(cov_r @ cov_g, disp=False)
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        return float(diff + np.trace(cov_r + cov_g - 2 * covmean))
    except Exception as e:
        print(f"  ⚠ FVD failed: {e}")
        return None


# ─── LPIPS ───

_m_lpips = None

def _lpips_model(net: str, device: str):
    global _m_lpips
    if _m_lpips is None:
        import lpips
        _m_lpips = lpips.LPIPS(net=net).to(device)
        _m_lpips.eval()
    return _m_lpips


def compute_lpips(ref_frames, gen_frames, net: str, device: str) -> float:
    m = _lpips_model(net, device)
    scores = []
    for r, g in zip(ref_frames, gen_frames):
        rt = torch.from_numpy(r).permute(2, 0, 1).float() / 255.0 * 2 - 1
        gt = torch.from_numpy(g).permute(2, 0, 1).float() / 255.0 * 2 - 1
        scores.append(m(rt.unsqueeze(0).to(device), gt.unsqueeze(0).to(device)).item())
    return float(np.mean(scores))


# ─── SSIM ───

def compute_ssim(ref_frames, gen_frames) -> float:
    scores = []
    for r, g in zip(ref_frames, gen_frames):
        scores.append(float(structural_similarity(
            r.astype(np.float32) / 255, g.astype(np.float32) / 255,
            data_range=1.0, channel_axis=2)))
    return float(np.mean(scores))


# ─── Optical Flow EPE ───

def _gray(f): return cv2.cvtColor(f, cv2.COLOR_RGB2GRAY)

def _flow(prev, nxt):
    return cv2.calcOpticalFlowFarneback(
        _gray(prev), _gray(nxt), None, 0.5, 3, 15, 3, 5, 1.2, 0)

def compute_flow_epe(ref_frames, gen_frames) -> float:
    errors = []
    for i in range(len(ref_frames) - 1):
        fr = _flow(ref_frames[i], ref_frames[i + 1])
        fg = _flow(gen_frames[i], gen_frames[i + 1])
        errors.append(float(np.mean(np.linalg.norm(fr - fg, axis=2))))
    return float(np.mean(errors))


# ─── 主流程 ───

def _discover_pairs(orig_dir: Path, gen_dir: Path) -> list[tuple[str, Path, Path]]:
    # 支持两种结构: 扁平 7.mp4 或 嵌套 sample_7/7.mp4
    gen_map = {}
    for p in sorted(gen_dir.rglob("*.mp4")):
        # 优先用文件名 stem，排除 iter_01 等中间文件
        if p.parent.name.startswith("sample_"):
            gen_map[p.parent.name.replace("sample_", "")] = p
        elif p.stem not in gen_map:
            gen_map[p.stem] = p
    return [(op.stem, op, gen_map[op.stem])
            for op in sorted(orig_dir.glob("*.mp4"))
            if op.stem in gen_map]


def main():
    parser = argparse.ArgumentParser(description="统一 6 项视频质量评测")
    parser.add_argument("--orig-dir", type=Path, default=Path("data/videos"))
    parser.add_argument("--gen-dir", type=Path, required=True)
    parser.add_argument("--caption-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--clip-model", default=DEFAULT_CLIP_MODEL)
    parser.add_argument("--xclip-model", default=DEFAULT_XCLIP_MODEL)
    parser.add_argument("--sample-frames", type=int, default=DEFAULT_SAMPLE_FRAMES)
    parser.add_argument("--frame-size", type=int, default=256)
    parser.add_argument("--lpips-net", default="alex", choices=["alex", "vgg", "squeeze"])
    parser.add_argument("--skip-fvd", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    pairs = _discover_pairs(args.orig_dir, args.gen_dir)
    if not pairs:
        print(f"No aligned pairs between {args.orig_dir} and {args.gen_dir}")
        return
    print(f"Found {len(pairs)} aligned samples")

    print("Loading CLIP / XCLIP...")
    clip_processor, clip_model, xclip_processor, xclip_model = build_models(
        args.device, args.clip_model, args.xclip_model)

    items = []
    for sid, rp, gp in tqdm(pairs, desc="Per-sample"):
        caption_path = args.caption_dir / f"{sid}.txt"
        if not caption_path.exists():
            print(f"  ⚠ skip {sid}: no caption")
            continue

        row = {"sample_id": sid}
        row.update(compute_clip_xclip(
            {"orig_path": rp, "gen_path": gp, "caption_path": caption_path},
            clip_processor, clip_model, xclip_processor, xclip_model, args.device))

        ref_frames, gen_frames = get_aligned_frames(
            rp, gp, args.sample_frames, args.frame_size)
        row["lpips"] = round(compute_lpips(ref_frames, gen_frames, args.lpips_net, args.device), 6)
        row["ssim"]  = round(compute_ssim(ref_frames, gen_frames), 6)
        row["of_epe"] = round(compute_flow_epe(ref_frames, gen_frames), 6)
        items.append(row)

    fvd = None
    if not args.skip_fvd:
        print("Computing FVD...")
        fvd = compute_fvd(args.orig_dir, args.gen_dir, args.sample_frames, args.device)

    keys = ["clip_o_g", "xclip_o_g", "lpips", "ssim", "of_epe"]
    means = {}
    for k in keys:
        vals = [it[k] for it in items if it.get(k) is not None]
        means[f"mean_{k}"] = round(np.mean(vals), 6) if vals else None

    summary = {"samples": len(items)}
    if fvd is not None:
        summary["fvd"] = round(fvd, 2)
    summary.update(means)

    csv_path = args.output_dir / "all_metrics.csv"
    with open(csv_path, "w", newline="") as f:
        fieldnames = ["sample_id", "clip_o_g", "xclip_o_g", "lpips", "ssim", "of_epe"]
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(items)

    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False))

    print("\n" + "=" * 60)
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(f"  CSV: {csv_path}")


if __name__ == "__main__":
    main()
