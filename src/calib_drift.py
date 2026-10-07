"""Topic A: đo độ nhạy của projection LiDAR->camera với calibration drift.

Chạy lại toàn bộ kết quả:
    python -m src.calib_drift sweep      # -> results/calib_drift_sweep.csv
    python -m src.calib_drift report     # -> results/figures/*.png + results/calib_drift_summary.csv
    python -m src.calib_drift ego        # -> results/ego_motion_ablation.csv (lỗi Time, nuScenes)
    python -m src.calib_drift latency    # -> results/latency.csv
    python -m src.calib_drift --help

Metric:
  pct_in_box   % điểm LiDAR nằm trong 3D box GT (tính bằng calib gốc) mà sau khi perturb vẫn chiếu
               vào trong 2D box GT của chính object đó. Cần label -> chỉ dùng để đánh giá offline.
  edge_dist_px khoảng cách trung vị (px) từ pixel "biên độ sâu" của LiDAR tới biên Canny gần nhất
               của ảnh. Không cần label -> dùng được khi chạy thật. Càng lớn = càng lệch.
"""
from __future__ import annotations

import argparse
import platform
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from starter.datasets import dataset_type, list_frames, load_frame
from starter.projection import (box3d_corners_cam, cam_to_image, draw_box2d, overlay_points,
                                perturb_extrinsic, project_velo_to_image, velo_to_cam)

RES = Path("results")
FIG = RES / "figures"
KITTI, NUSC = "data/kitti_mini", "data/nuscenes_mini_subset"
ROT_LEVELS = [0.0, 0.25, 0.5, 1.0, 2.0, 3.0]          # độ
TRANS_LEVELS = [0.0, 0.02, 0.05, 0.10]                # mét
AXES = {"yaw": "rot", "pitch": "rot", "roll": "rot", "tx": "trans", "ty": "trans", "tz": "trans"}
BINS = [("near", 0, 15), ("mid", 15, 30), ("far", 30, 1e9)]   # theo z_cam của object (m)
FILL = {"kitti": 9, "nuscenes": 25}   # kernel lấp depth thưa: 64 beam vs 32 beam (ponytail: tay chỉnh theo sensor)


def pick_frames(root: str) -> list[tuple[str, str]]:
    """(frame_id, nhóm). KITTI: 20 frame. nuScenes: mỗi 4 keyframe của 2 scene (10 ngày + 10 đêm)."""
    fr = list_frames(root)
    if dataset_type(root) == "kitti":
        return [(f, "day") for f in fr]
    return [(f, "day" if f.startswith("scene-0103") else "night") for f in fr[::4]]


def drift(calib, axis: str, level: float):
    kw = {"yaw": dict(yaw_deg=level), "pitch": dict(pitch_deg=level), "roll": dict(roll_deg=level),
          "tx": dict(t_xyz_m=(level, 0, 0)), "ty": dict(t_xyz_m=(0, level, 0)), "tz": dict(t_xyz_m=(0, 0, level))}[axis]
    return perturb_extrinsic(calib, **kw)


def points_in_box(pts_cam: np.ndarray, obj, margin: float = 0.1) -> np.ndarray:
    """Mask điểm (camera frame) nằm trong 3D box KITTI (location = tâm đáy, y xuống)."""
    h, w, l = obj.dimensions
    c, s = np.cos(obj.rotation_y), np.sin(obj.rotation_y)
    d = pts_cam - obj.location
    x, z = c * d[:, 0] - s * d[:, 2], s * d[:, 0] + c * d[:, 2]     # R(ry)^T
    return ((np.abs(x) <= l / 2 + margin) & (np.abs(z) <= w / 2 + margin)
            & (d[:, 1] <= margin) & (d[:, 1] >= -h - margin) & np.isfinite(d).all(1))


def edge_ref(image: np.ndarray) -> np.ndarray:
    """Distance transform tới biên Canny của ảnh (tính 1 lần / frame)."""
    edges = cv2.Canny(cv2.GaussianBlur(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (5, 5), 0), 50, 150)
    return cv2.distanceTransform(255 - edges, cv2.DIST_L2, 3)


def edge_dist(dt: np.ndarray, uv: np.ndarray, depth: np.ndarray, fill: int):
    """Trung vị khoảng cách (px) từ biên độ sâu LiDAR tới biên ảnh gần nhất. Trả về (score, số pixel biên)."""
    H, W = dt.shape
    inv = np.zeros((H, W), np.float32)
    ui, vi = np.clip(uv[:, 0].astype(int), 0, W - 1), np.clip(uv[:, 1].astype(int), 0, H - 1)
    np.maximum.at(inv, (vi, ui), 1.0 / depth)                          # điểm gần thắng
    k = np.ones((fill, fill), np.uint8)
    inv_f = cv2.dilate(inv, k)                                         # lấp khe giữa các beam
    valid = cv2.erode((inv_f > 0).astype(np.uint8), np.ones((fill, fill), np.uint8)) > 0
    z = np.where(inv_f > 0, 1.0 / np.maximum(inv_f, 1e-6), 0).astype(np.float32)
    g = cv2.morphologyEx(z, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    edge = valid & (g > np.maximum(1.0, 0.1 * z))                      # nhảy sâu > max(1 m, 10%)
    n = int(edge.sum())
    return (float(np.median(np.minimum(dt[edge], 50))) if n > 50 else np.nan), n


def object_sets(fr, pts_cam_gt):
    """Danh sách (obj, idx điểm trong box, bin khoảng cách) cho object đủ điểm và ít bị cắt."""
    out = []
    for o in fr["labels"]:
        if o.truncated > 0.3 or o.location[2] <= 0:
            continue
        idx = np.flatnonzero(points_in_box(pts_cam_gt, o))
        if len(idx) >= 5:
            out.append((o, idx, next(b for b, lo, hi in BINS if lo <= o.location[2] < hi)))
    return out


def sweep(args) -> None:
    rows = []
    for root in (KITTI, NUSC):
        dtype = dataset_type(root)
        for fid, grp in pick_frames(root):
            fr = load_frame(root, fid)
            pts, calib, img = fr["points"], fr["calib"], fr["image"]
            gt_cam = velo_to_cam(pts[:, :3], calib)
            objs = object_sets(fr, gt_cam)
            dt = edge_ref(img)
            configs = [("none", 0.0)] + [(a, l) for a, kind in AXES.items()
                                         for l in (ROT_LEVELS if kind == "rot" else TRANS_LEVELS) if l > 0]
            for axis, level in configs:
                c = drift(calib, axis, level) if axis != "none" else calib
                uv_all, depth, mask = project_velo_to_image(pts, c, img.shape)
                cam = velo_to_cam(pts[:, :3], c)
                fwd = np.isfinite(cam).all(1) & (cam[:, 2] > 0.1)
                score, n_edge = edge_dist(dt, uv_all, depth, FILL[dtype])
                row = dict(dataset=dtype, group=grp, frame=fid, axis=axis, level=level,
                           pct_fov=100 * mask.sum() / max(fwd.sum(), 1), edge_dist_px=score, n_edge_px=n_edge)
                full_uv = np.full((len(pts), 2), np.nan)
                full_uv[mask] = uv_all
                for b, _, _ in BINS:
                    row[f"n_{b}"] = row[f"k_{b}"] = 0
                for o, idx, b in objs:
                    x1, y1, x2, y2 = o.bbox
                    u, v = full_uv[idx, 0], full_uv[idx, 1]
                    ok = (u >= x1) & (u <= x2) & (v >= y1) & (v <= y2)       # NaN -> False
                    row[f"n_{b}"] += len(idx)
                    row[f"k_{b}"] += int(ok.sum())
                rows.append(row)
        print(f"{dtype}: xong")
    RES.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(RES / "calib_drift_sweep.csv", index=False)
    print("->", RES / "calib_drift_sweep.csv", len(rows), "dòng")


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Gộp theo (dataset, group, axis, level): pct_in_box tổng và theo bin, edge_dist trung bình."""
    g = df.groupby(["dataset", "group", "axis", "level"])
    out = g[["pct_fov", "edge_dist_px"]].mean()
    for b, _, _ in BINS:
        s = g[[f"n_{b}", f"k_{b}"]].sum()
        out[f"pct_in_box_{b}"] = 100 * s[f"k_{b}"] / s[f"n_{b}"].replace(0, np.nan)
    n = sum(g[f"n_{b}"].sum() for b, _, _ in BINS)
    k = sum(g[f"k_{b}"].sum() for b, _, _ in BINS)
    out["pct_in_box"] = 100 * k / n.replace(0, np.nan)
    out["n_obj_pts"] = n
    return out.reset_index()


def baseline_axis(df: pd.DataFrame) -> pd.DataFrame:
    """Mỗi axis cần dòng level=0: lấy từ axis 'none'."""
    base = df[df.axis == "none"]
    extra = [base.assign(axis=a) for a in AXES]
    return pd.concat([df[df.axis != "none"]] + extra, ignore_index=True)


def detection(df: pd.DataFrame) -> pd.DataFrame:
    """Ngưỡng = max edge_dist trên các frame sạch của CHÍNH dataset/nhóm đó (0 false alarm trong mẫu).
    Detection rate = % frame bị perturb có edge_dist > ngưỡng."""
    rows = []
    for (ds, grp), d in df.groupby(["dataset", "group"]):
        thr = d[d.axis == "none"].edge_dist_px.max()
        p = d[d.axis != "none"]
        for (axis, level), q in p.groupby(["axis", "level"]):
            rows.append(dict(dataset=ds, group=grp, axis=axis, level=level, threshold_px=thr,
                             detect_rate=100 * (q.edge_dist_px > thr).mean(), n_frames=len(q)))
    return pd.DataFrame(rows)


def report(args) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = pd.read_csv(RES / "calib_drift_sweep.csv")
    summ = summarize(baseline_axis(df))
    det = detection(df)
    summ = summ.merge(det[["dataset", "group", "axis", "level", "threshold_px", "detect_rate"]],
                      on=["dataset", "group", "axis", "level"], how="left")
    summ.sort_values(["dataset", "group", "axis", "level"]).round(3).to_csv(RES / "calib_drift_summary.csv", index=False)
    FIG.mkdir(parents=True, exist_ok=True)

    s = summ[summ.group != "night"]                                       # đường chính: KITTI + nuScenes ngày
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for ds, ls in (("kitti", "-"), ("nuscenes", "--")):
        for a, col in zip(("yaw", "pitch", "roll"), ("C0", "C1", "C2")):
            q = s[(s.dataset == ds) & (s.axis == a)].sort_values("level")
            ax[0].plot(q.level, q.pct_in_box, ls, color=col, marker="o", label=f"{a} ({ds})")
        for a, col in zip(("tx", "ty", "tz"), ("C3", "C4", "C5")):
            q = s[(s.dataset == ds) & (s.axis == a)].sort_values("level")
            ax[1].plot(q.level * 100, q.pct_in_box, ls, color=col, marker="o", label=f"{a} ({ds})")
    ax[0].set(xlabel="góc lệch (độ)", ylabel="% điểm vật thể còn trong 2D box", title="Xoay extrinsic")
    ax[1].set(xlabel="dịch (cm)", title="Tịnh tiến extrinsic")
    for a in ax[:2]:
        a.axhline(90, color="gray", lw=.5); a.grid(alpha=.3); a.legend(fontsize=7)
    for b, col in zip(("near", "mid", "far"), ("C3", "C1", "C0")):
        q = s[(s.dataset == "kitti") & (s.axis == "yaw")].sort_values("level")
        ax[2].plot(q.level, q[f"pct_in_box_{b}"], marker="o", color=col, label=f"{b}")
    ax[2].set(xlabel="yaw (độ)", title="Yaw theo khoảng cách (KITTI)"); ax[2].legend(); ax[2].grid(alpha=.3)
    plt.tight_layout(); plt.savefig(FIG / "drift_curves.png", dpi=130); plt.close()

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for (ds, grp), col in zip((("kitti", "day"), ("nuscenes", "day"), ("nuscenes", "night")), ("C0", "C1", "C3")):
        d = det[(det.dataset == ds) & (det.group == grp) & (det.axis == "yaw")].sort_values("level")
        e = summ[(summ.dataset == ds) & (summ.group == grp) & (summ.axis == "yaw")].sort_values("level")
        ax[0].plot(e.level, e.edge_dist_px, marker="o", color=col, label=f"{ds} {grp}")
        ax[0].axhline(d.threshold_px.iloc[0], color=col, ls=":", lw=1)
        ax[1].plot(d.level, d.detect_rate, marker="o", color=col, label=f"{ds} {grp}")
    ax[0].set(xlabel="yaw (độ)", ylabel="edge_dist_px (trung vị)", title="Alignment score; đường chấm = ngưỡng")
    ax[1].set(xlabel="yaw (độ)", ylabel="% frame phát hiện drift", title="Detection rate")
    for a in ax: a.legend(); a.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(FIG / "alignment_score.png", dpi=130); plt.close()
    print(summ[(summ.axis.isin(["yaw"]))][["dataset", "group", "level", "pct_in_box", "edge_dist_px", "detect_rate"]].round(2).to_string())
    make_failures()


def put(img, text, org, scale=0.55):
    """Chữ trắng trên nền đen cho dễ đọc."""
    (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    cv2.rectangle(img, (org[0] - 2, org[1] - h - 3), (org[0] + w + 2, org[1] + 4), (0, 0, 0), -1)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA)


def crop_pair(root, fid, obj, axis, level, name, note):
    """Ảnh 2 panel: calib gốc vs calib bị lệch, crop quanh 2D box GT, vẽ điểm trong 3D box."""
    fr = load_frame(root, fid)
    pts, calib, img = fr["points"], fr["calib"], fr["image"]
    idx = np.flatnonzero(points_in_box(velo_to_cam(pts[:, :3], calib), obj))
    x1, y1, x2, y2 = (int(v) for v in obj.bbox)
    cx, cy, half = (x1 + x2) // 2, (y1 + y2) // 2, max(x2 - x1, y2 - y1, 40) * 3
    panels = []
    for c, title in ((calib, "calib goc"), (drift(calib, axis, level), f"{axis} {level}")):
        uv, d, m = project_velo_to_image(pts[idx], c, img.shape)
        vis = overlay_points(img, uv, d, radius=2)
        vis = draw_box2d(vis, obj.bbox, label=f"{obj.type} z={obj.location[2]:.0f}m")
        H, W = img.shape[:2]
        a, b = max(cx - half, 0), max(cy - half, 0)
        crop = cv2.resize(vis[b:min(cy + half, H), a:min(cx + half, W)], (400, 400))
        inside = int(((uv[:, 0] >= x1) & (uv[:, 0] <= x2) & (uv[:, 1] >= y1) & (uv[:, 1] <= y2)).sum())
        put(crop, f"{title}: {inside}/{len(idx)} diem trong box", (5, 20))
        panels.append(crop)
    out = np.hstack(panels)
    put(out, note, (5, 390), 0.5)
    cv2.imwrite(str(FIG / name), out)
    print("->", FIG / name)


def make_failures() -> None:
    # fail_01: xe xa nhất của KITTI mini, yaw 2 độ -> điểm trôi khỏi box (Geometry)
    best = None
    for fid in list_frames(KITTI):
        fr = load_frame(KITTI, fid)
        cam = velo_to_cam(fr["points"][:, :3], fr["calib"])
        for o in fr["labels"]:
            if o.type == "Car" and o.truncated < .3 and 40 < o.location[2] and len(np.flatnonzero(points_in_box(cam, o))) >= 8:
                if best is None or o.location[2] > best[2].location[2]:
                    best = (KITTI, fid, o)
    root, fid, o = best
    crop_pair(root, fid, o, "yaw", 1.0, "fail_01_yaw_1deg_far_car.png", f"KITTI {fid}: yaw 1 deg lam xe xa mat diem")
    # fail_02: nuScenes ban đêm: score không thấy drift (Metric / Preprocess)
    df = pd.read_csv(RES / "calib_drift_sweep.csv")
    det = detection(df)
    night = det[(det.dataset == "nuscenes") & (det.group == "night") & (det.axis == "yaw") & (det.level == 3.0)]
    print("night yaw 3deg detect_rate:", night.detect_rate.values)
    fr_id = df[(df.dataset == "nuscenes") & (df.group == "night") & (df.axis == "yaw") & (df.level == 3.0)] \
        .assign(base=lambda d: d.frame.map(df[(df.axis == "none")].set_index("frame").edge_dist_px)) \
        .assign(inc=lambda d: d.edge_dist_px - d.base).sort_values("inc").iloc[0].frame
    fr = load_frame(NUSC, fr_id)
    pan = []
    for lv in (0.0, 3.0):
        c = drift(fr["calib"], "yaw", lv)
        uv, d, _ = project_velo_to_image(fr["points"], c, fr["image"].shape)
        v = overlay_points(fr["image"], uv, d, radius=3)
        cv2.putText(v, f"yaw {lv} deg", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        pan.append(cv2.resize(v, (800, 450)))
    cv2.imwrite(str(FIG / "fail_02_night_score_blind.png"), np.vstack(pan))
    print("-> fail_02 frame", fr_id)


def ego(args) -> None:
    """Lỗi Time: tắt bù ego-motion trên nuScenes, so với calib đúng + bù. Dùng cùng metric pct_in_box."""
    rows = []
    for fid, grp in pick_frames(NUSC):
        for use in (True, False):
            fr = load_frame(NUSC, fid, use_ego_motion=use)
            cam = velo_to_cam(fr["points"][:, :3], fr["calib"])
            # GT box lấy từ bản có bù (đúng thời gian camera) để so công bằng
            ref = load_frame(NUSC, fid, use_ego_motion=True)
            objs = object_sets(ref, velo_to_cam(ref["points"][:, :3], ref["calib"]))
            uv, _, m = project_velo_to_image(fr["points"], fr["calib"], fr["image"].shape)
            full = np.full((len(fr["points"]), 2), np.nan); full[m] = uv
            n = k = 0
            for o, idx, _ in objs:
                x1, y1, x2, y2 = o.bbox
                u, v = full[idx, 0], full[idx, 1]
                k += int(((u >= x1) & (u <= x2) & (v >= y1) & (v <= y2)).sum()); n += len(idx)
            rows.append(dict(frame=fid, group=grp, ego_comp=use, n=n, k=k,
                             dt_ms=(fr["timestamp_camera_us"] - fr["timestamp_lidar_us"]) / 1e3))
    d = pd.DataFrame(rows)
    d.to_csv(RES / "ego_motion_ablation.csv", index=False)
    g = d.groupby(["group", "ego_comp"])
    print((100 * g.k.sum() / g.n.sum()).round(2).rename("pct_in_box"), g.dt_ms.apply(lambda x: x.abs().mean()).round(1))


def latency(args) -> None:
    """p50/p95 của project + alignment score; bỏ 3 lần warm-up, chạy 30 lần."""
    rows = []
    for root in (KITTI, NUSC):
        dtype = dataset_type(root)
        fr = load_frame(root, pick_frames(root)[3][0])
        dt = edge_ref(fr["image"])
        for name, fn in (("project", lambda: project_velo_to_image(fr["points"], fr["calib"], fr["image"].shape)),
                         ("project+edge_score", lambda: edge_dist(dt, *project_velo_to_image(fr["points"], fr["calib"], fr["image"].shape)[:2], FILL[dtype]))):
            for _ in range(3):
                fn()
            t = []
            for _ in range(30):
                t0 = time.perf_counter(); fn(); t.append((time.perf_counter() - t0) * 1e3)
            rows.append(dict(dataset=dtype, stage=name, n_points=len(fr["points"]), runs=30,
                             p50_ms=np.percentile(t, 50), p95_ms=np.percentile(t, 95),
                             hw=f"{platform.processor() or platform.machine()} | {platform.platform()} | CPU only"))
    d = pd.DataFrame(rows).round(2)
    d.to_csv(RES / "latency.csv", index=False)
    print(d.drop(columns="hw").to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Topic A: calibration drift sensitivity (xem docstring đầu file)")
    ap.add_argument("cmd", choices=["sweep", "report", "ego", "latency", "fail"])
    a = ap.parse_args()
    {"sweep": sweep, "report": report, "ego": ego, "latency": latency, "fail": lambda _: make_failures()}[a.cmd](a)
