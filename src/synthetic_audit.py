"""Bonus B6: tự động tìm lỗi cài sẵn trong data/synthetic -> results/synthetic_bugs.csv (lỗi, frame, cách phát hiện).

    python -m src.synthetic_audit
Quy tắc: so mỗi frame với trung vị của các frame còn lại (robust với 5 frame) + kiểm tra timestamp.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from starter.datasets import list_frames, load_points

ROOT = "data/synthetic"


def main() -> None:
    frames = list_frames(ROOT)
    az_hist, nan_ratio = {}, {}
    for f in frames:
        p = load_points(ROOT, f)
        ok = np.isfinite(p).all(1)
        nan_ratio[f] = 1 - ok.mean()
        az = np.degrees(np.arctan2(p[ok, 1], p[ok, 0]))
        az_hist[f] = np.histogram(az, bins=72, range=(-180, 180))[0]      # bin 5 độ
    H = np.array([az_hist[f] for f in frames], float)
    rows = []
    for f, r in nan_ratio.items():
        if r > 1e-4:
            rows.append(("Điểm NaN/Inf (I/O)", f, f"invalid_ratio={r:.2%} > 0 (isfinite)"))
    for i, f in enumerate(frames):                                           # sector dropout
        ref = np.median(np.delete(H, i, 0), 0)
        bad = np.flatnonzero((ref > 20) & (H[i] < 0.6 * ref))
        # bỏ các bin lệch ngẫu nhiên: chỉ lấy dải liên tục >= 3 bin (15 độ)
        runs = np.split(bad, np.flatnonzero(np.diff(bad) > 1) + 1) if len(bad) else []
        for run in (r for r in runs if len(r) >= 3):
            lo, hi = -180 + 5 * run[0], -180 + 5 * (run[-1] + 1)
            rows.append(("Mất cả dải azimuth / sector dropout (Preprocess/sensor)", f,
                         f"azimuth [{lo}°,{hi}°] còn <60% số điểm so với trung vị các frame khác"))
    ts = np.loadtxt(f"{ROOT}/training/timestamps.txt")
    dt = np.diff(ts)
    for i in np.flatnonzero(np.abs(dt - np.median(dt)) > 0.5 * np.median(dt)):
        rows.append(("Khoảng cách timestamp bất thường / mất frame (Time)", frames[i + 1],
                     f"dt={dt[i]:.2f}s so với trung vị {np.median(dt):.2f}s (frame {frames[i]}→{frames[i+1]})"))
    out = pd.DataFrame(rows, columns=["loi", "frame", "cach_phat_hien"])
    out.to_csv("results/synthetic_bugs.csv", index=False, encoding="utf-8")
    print(out.to_string().encode("ascii", "replace").decode())


if __name__ == "__main__":
    main()
