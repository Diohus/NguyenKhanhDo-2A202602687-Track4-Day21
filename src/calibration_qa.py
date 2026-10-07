"""Topic A: reproducible projection QA and rubric bonus experiments.

Original lab implementation with Codex assistance; uses supplied starter APIs.
Run from repository root: python -m src.calibration_qa --help
"""
from __future__ import annotations

import argparse
import platform
from pathlib import Path
from time import perf_counter_ns

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from starter.datasets import list_frames, load_frame
from starter.data_health import point_stats
from starter.projection import (cam_to_image, draw_box2d, overlay_points,
                                perturb_extrinsic, project_velo_to_image, velo_to_cam)

CLASSES = {"Car", "Van", "Truck", "Bus", "Pedestrian", "Cyclist", "Bicycle"}


def self_check():
    """Analytic geometry checks, independent of plotted benchmark outputs."""
    from starter.kitti_io import KittiCalib, KittiObject
    frame = load_frame("data/synthetic", "000000")
    point = np.array([[10., 0., 0.]])
    cam = velo_to_cam(point, frame["calib"])
    uv, depth, valid = cam_to_image(cam, frame["calib"].P2, frame["image"].shape)
    np.testing.assert_allclose(depth, [9.72732109], atol=1e-6)
    np.testing.assert_allclose(uv, [[613.96414869, 175.00653723]], atol=1e-5)
    assert valid.tolist() == [True]
    P = np.array([[10., 0, 10, 0], [0, 10, 10, 0], [0, 0, 1, 0]])
    points = np.array([[0, 0, 1], [np.nan, 0, 1], [np.inf, 0, 1],
                       [0, 0, -1], [0, 0, .1], [-1, -1, 1], [1, 0, 1]])
    uv, depth, mask = cam_to_image(points, P, (20, 20, 3))
    assert mask.tolist() == [True, False, False, False, False, True, False]
    np.testing.assert_allclose(uv, [[10, 10], [0, 0]])
    empty = cam_to_image(np.empty((0, 3)), P, (20, 20))
    assert empty[0].shape == (0, 2) and empty[2].shape == (0,)
    # Non-identity rectification AND translation, checked against column-vector formula.
    calib = KittiCalib(P, np.array([[0., -1, 0], [1, 0, 0], [0, 0, 1]]),
                       np.array([[1., 0, 0, 2], [0, 1, 0, 3], [0, 0, 1, 4]]))
    np.testing.assert_allclose(velo_to_cam(np.array([[1., 2, 3]]), calib), [[-5, 3, 7]])
    translated = P.copy(); translated[2, 3] = -2
    assert not cam_to_image(np.array([[0., 0, 2]]), translated, (20, 20))[2].any()
    obj = KittiObject("Car", 0, 0, 0, np.array([0, 0, 20, 20]),
                      np.array([2., 2, 4]), np.array([0., 0, 10]), np.pi / 2)
    assert object_indices(np.array([[0., -1, 11.5], [1.5, -1, 10], [0, .1, 10]]), obj).tolist() == [0]
    print("PASS: CP2 reference, nonidentity rectification, translation, NaN/Inf, depth, FOV boundaries, empty data and rotated boxes")


def synthetic_audit(root, out, figures):
    frames = [load_frame(root, fid) for fid in list_frames(root)]
    histograms = []
    for frame in frames:
        p = frame["points"]
        p = p[np.isfinite(p).all(axis=1)]
        histograms.append(np.histogram(np.degrees(np.arctan2(p[:, 1], p[:, 0])),
                                      bins=72, range=(-180, 180))[0])
    timestamps = np.loadtxt(Path(root) / "training" / "timestamps.txt")
    gaps = np.r_[np.nan, np.diff(timestamps)]
    normal_gap = np.nanmedian(gaps)
    rows = []
    for i, frame in enumerate(frames):
        # Leave-one-out reference avoids incorporating a frame into its own anomaly baseline.
        reference = np.median(np.delete(np.asarray(histograms), i, axis=0), axis=0)
        ratio = histograms[i] / np.maximum(reference, 1)
        deficient = np.flatnonzero(ratio < .45)
        runs = np.split(deficient, np.flatnonzero(np.diff(deficient) != 1) + 1)
        longest = max(runs, key=len)
        sector = len(longest) >= 4
        stats = point_stats(frame["points"])
        rows.append(dict(frame_id=frame["frame_id"], **stats,
                         invalid_count=int((~np.isfinite(frame["points"]).all(axis=1)).sum()),
                         time_gap_s=gaps[i], reference_gap_s=normal_gap,
                         sector_start_deg=-180 + 5 * longest[0] if sector else np.nan,
                         sector_end_deg=-180 + 5 * (longest[-1] + 1) if sector else np.nan,
                         sector_density_ratio=float(ratio[longest].mean()) if sector else np.nan,
                         invalid_flag=stats["invalid_ratio"] > 0,
                         sector_flag=sector, time_flag=bool(gaps[i] > 1.5 * normal_gap)))
    table = pd.DataFrame(rows)
    table.to_csv(out / "synthetic_audit.csv", index=False, float_format="%.9f")
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), constrained_layout=True)
    axes[0].bar(table.frame_id, table.invalid_count); axes[0].set_ylabel("Invalid points")
    for i, hist in enumerate(histograms):
        axes[1].plot(np.arange(-177.5, 180, 5), hist, label=f"{i:06d}")
    axes[1].set_xlabel("Azimuth (degrees)"); axes[1].set_ylabel("Points per 5-degree bin")
    axes[1].legend(fontsize=7)
    axes[2].bar(table.frame_id, table.time_gap_s); axes[2].axhline(1.5 * normal_gap, color="red", linestyle="--")
    axes[2].set_ylabel("Time gap (seconds)")
    for axis in [axes[0], axes[2]]: axis.tick_params(axis="x", rotation=45)
    fig.savefig(figures / "synthetic_audit.png", dpi=140); plt.close(fig)


def projection_arrays(points, calib, shape):
    uv, depth, valid = project_velo_to_image(points, calib, shape)
    full = np.full((len(points), 2), np.nan)
    full[valid] = uv
    return full, depth, valid


def object_indices(cam, obj):
    """Inverse KITTI yaw; location is bottom centre, local y in [-h, 0]."""
    delta = cam - obj.location
    c, s = np.cos(obj.rotation_y), np.sin(obj.rotation_y)
    local = delta @ np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    h, w, length = obj.dimensions
    inside = ((np.abs(local[:, 0]) <= length / 2) &
              (local[:, 1] >= -h) & (local[:, 1] <= 0) &
              (np.abs(local[:, 2]) <= w / 2))
    return np.flatnonzero(inside)


def prepare(frame, max_range, min_points):
    raw = frame["points"]
    finite = np.isfinite(raw).all(axis=1)
    points = raw[finite & (np.linalg.norm(raw[:, :3], axis=1) <= max_range)]
    cam = velo_to_cam(points[:, :3], frame["calib"])
    groups = []
    for index, obj in enumerate(frame["labels"]):
        if obj.type not in CLASSES or obj.location[2] <= .1:
            continue
        ids = object_indices(cam, obj)
        if len(ids) >= min_points:
            groups.append((index, obj, ids))
    return points, groups


def score_objects(full, valid, groups):
    rows = []
    for index, obj, ids in groups:
        uv = full[ids]
        x1, y1, x2, y2 = obj.bbox
        hit = (valid[ids] & (uv[:, 0] >= x1) & (uv[:, 0] <= x2) &
               (uv[:, 1] >= y1) & (uv[:, 1] <= y2))
        rows.append(dict(object_id=index, class_name=obj.type,
                         distance_m=float(np.linalg.norm(obj.location)),
                         n_object_points=len(ids), n_inside_box=int(hit.sum()),
                         box_score=float(hit.mean())))
    return rows


def save_overlay(frame, points, calib, path, title, depth_range=None):
    uv, depth, _ = project_velo_to_image(points, calib, frame["image"].shape)
    if depth_range:
        keep = (depth >= depth_range[0]) & (depth < depth_range[1])
        uv, depth = uv[keep], depth[keep]
    image = overlay_points(frame["image"], uv, depth, radius=1)
    for obj in frame["labels"]:
        if obj.type in CLASSES:
            image = draw_box2d(image, obj.bbox, label=obj.type)
    cv2.rectangle(image, (0, 0), (image.shape[1], 32), (0, 0, 0), -1)
    cv2.putText(image, title, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 1)
    if not cv2.imwrite(str(path), image):
        raise OSError(f"Cannot write {path}")


def plot_sweep(table, path, kind="yaw"):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for dataset, group in table.groupby("dataset", sort=True):
        axes[0].plot(group.level, 100 * group.box_score, "o-", label=dataset)
        axes[1].plot(group.level, 100 * group.fov_ratio, "o-", label=dataset)
    for axis in axes:
        axis.set_xlabel("Yaw drift (degrees)" if kind == "yaw" else "Translation along LiDAR z (m)")
        axis.grid(alpha=.3)
        axis.legend()
    axes[0].set_ylabel("Object-balanced box alignment (%)")
    axes[1].set_ylabel("Points inside image (%)")
    fig.savefig(path, dpi=140)
    plt.close(fig)


def benchmark(args):
    out = Path(args.out_dir)
    figures = out / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    rows, objects, stress, latency = [], [], [], []
    cached = {}
    for root in args.data_roots:
        dataset = Path(root).name
        ids = list_frames(root)
        if args.frames:
            ids = [fid for fid in ids if fid in args.frames]
        if not ids:
            raise ValueError(f"No selected frames in {root}")
        for frame_index, fid in enumerate(ids):
            frame = load_frame(root, fid)
            points, groups = prepare(frame, args.max_range, args.min_points)
            baseline_uv, _, baseline_valid = projection_arrays(points, frame["calib"], frame["image"].shape)
            split = "calibration" if frame_index % 5 == 0 else "evaluation"
            for kind, levels in (("yaw", args.yaw_levels), ("translation_z", [0, .02, .05, .1])):
                for level in levels:
                    calib = (perturb_extrinsic(frame["calib"], yaw_deg=level) if kind == "yaw"
                             else perturb_extrinsic(frame["calib"], t_xyz_m=(0, 0, level)))
                    full, _, valid = projection_arrays(points, calib, frame["image"].shape)
                    obj_rows = score_objects(full, valid, groups)
                    common = baseline_valid & valid
                    shifts = np.linalg.norm(full[common] - baseline_uv[common], axis=1)
                    rows.append(dict(dataset=dataset, frame_id=fid, split=split, kind=kind,
                                     level=level, seed=args.seed, max_range_m=args.max_range,
                                     min_object_points=args.min_points, n_points=len(points),
                                     n_objects=len(groups), n_fov=int(valid.sum()),
                                     fov_ratio=float(valid.mean()) if len(valid) else np.nan,
                                     box_score=float(np.mean([r["box_score"] for r in obj_rows])) if obj_rows else np.nan,
                                     pixel_shift_p50=float(np.median(shifts)) if len(shifts) else np.nan,
                                     delta_time_ms=(frame.get("timestamp_camera_us", 0) - frame.get("timestamp_lidar_us", 0))/1000
                                     if "timestamp_camera_us" in frame else np.nan))
                    if kind == "yaw":
                        objects.extend(dict(dataset=dataset, frame_id=fid, yaw_deg=level, **r) for r in obj_rows)
            # Two independent degradation types; reuse one seeded mask/noise field across levels.
            rng = np.random.default_rng(args.seed + frame_index)
            uniforms = rng.random(len(points))
            normal = rng.normal(size=(len(points), 3))
            original_memberships = sum(len(ids) for _, _, ids in groups)
            for kind, levels in (("dropout", [1., .9, .7, .5, .3]), ("noise", [0., .02, .05, .1, .2])):
                for level in levels:
                    if kind == "dropout":
                        keep = uniforms < level
                        altered = points[keep]
                        remap = np.full(len(points), -1, dtype=int)
                        remap[keep] = np.arange(keep.sum())
                        changed_groups = [(i, o, remap[idx[keep[idx]]]) for i, o, idx in groups]
                    else:
                        altered = points.copy()
                        altered[:, :3] += level * normal
                        changed_groups = groups
                    full, _, valid = projection_arrays(altered, frame["calib"], frame["image"].shape)
                    scored = score_objects(full, valid, changed_groups) if kind == "noise" else []
                    if kind == "dropout":
                        # Denominator is pre-dropout support: dropping a point counts as losing evidence.
                        scored = []
                        for (_, obj, idx), (_, _, original) in zip(changed_groups, groups):
                            r = score_objects(full, valid, [(0, obj, idx)])[0] if len(idx) else {"n_inside_box": 0}
                            scored.append({"box_score": r["n_inside_box"] / len(original)})
                    stress.append(dict(dataset=dataset, frame_id=fid, kind=kind, level=level,
                                       seed=args.seed + frame_index, n_points=len(altered),
                                       fov_ratio=float(valid.mean()) if len(valid) else np.nan,
                                       object_support=sum(len(idx) for _, _, idx in changed_groups),
                                       baseline_object_support=original_memberships,
                                       retained_alignment=float(np.mean([r["box_score"] for r in scored])) if scored else np.nan))
            if frame_index == 0:
                cached[dataset] = (frame, points, groups)
                # Warm-up excluded, then at least 20 repeats, projection only, no disk/plot/labels.
                for config in ("float64", "float32"):
                    def timed_projection():
                        if config == "float64":
                            return project_velo_to_image(points, frame["calib"], frame["image"].shape)
                        T = frame["calib"].T_cam_velo.astype(np.float32)
                        hom = np.column_stack((points[:, :3], np.ones(len(points), dtype=np.float32)))
                        cam = (hom @ T.T)[:, :3]
                        return cam_to_image(cam, frame["calib"].P2, frame["image"].shape)
                    timed_projection()
                    reference, _, ref_valid = projection_arrays(points, frame["calib"], frame["image"].shape)
                    u, _, valid = timed_projection()
                    full = np.full_like(reference, np.nan)
                    full[valid] = u
                    same = valid & ref_valid
                    max_error = float(np.max(np.linalg.norm(full[same] - reference[same], axis=1))) if same.any() else 0.
                    for repeat in range(args.repeats):
                        start = perf_counter_ns()
                        timed_projection()
                        elapsed = (perf_counter_ns() - start) / 1e6
                        latency.append(dict(dataset=dataset, frame_id=fid, config=config,
                                            repeat=repeat, latency_ms=elapsed, n_points=len(points),
                                            mask_difference=int(np.count_nonzero(valid != ref_valid)),
                                            max_pixel_error=max_error, cpu=platform.processor(),
                                            machine=platform.machine(), platform=platform.platform(),
                                            python=platform.python_version(), numpy=np.__version__, gpu="not used"))
            print(f"{dataset}/{fid}: {len(points)} points, {len(groups)} supported objects", flush=True)
    frame_table = pd.DataFrame(rows)
    frame_table.to_csv(out / "calibration_frames.csv", index=False, float_format="%.9f")
    object_table = pd.DataFrame(objects)
    object_table.to_csv(out / "calibration_objects.csv", index=False, float_format="%.9f")
    object_table["distance_band"] = pd.cut(object_table.distance_m, [0, 15, 30, np.inf],
                                           labels=["0-15m", "15-30m", "30m+"])
    object_table.groupby(["dataset", "yaw_deg", "distance_band"], observed=True).agg(
        n_objects=("object_id", "count"), box_score=("box_score", "mean"),
        n_points=("n_object_points", "sum")
    ).to_csv(out / "calibration_distance_summary.csv", float_format="%.9f")
    summary = frame_table.groupby(["dataset", "kind", "level"], sort=True).agg(
        n_frames=("frame_id", "count"), n_supported_frames=("box_score", "count"),
        box_score=("box_score", "mean"), fov_ratio=("fov_ratio", "mean"),
        pixel_shift_p50=("pixel_shift_p50", "median"), n_objects=("n_objects", "sum"),
        delta_time_ms=("delta_time_ms", "mean")).reset_index()
    summary.to_csv(out / "calibration_summary.csv", index=False, float_format="%.9f")
    plot_sweep(summary[summary.kind == "yaw"], figures / "yaw_sweep.png")
    plot_sweep(summary[summary.kind == "translation_z"], figures / "translation_sweep.png", "translation")
    stress_table = pd.DataFrame(stress)
    stress_table.to_csv(out / "stress_frames.csv", index=False, float_format="%.9f")
    stress_summary = stress_table.groupby(["dataset", "kind", "level"]).agg(
        retained_alignment=("retained_alignment", "mean"), object_support=("object_support", "sum"),
        n_points=("n_points", "mean"), fov_ratio=("fov_ratio", "mean")).reset_index()
    stress_summary.to_csv(out / "stress_summary.csv", index=False, float_format="%.9f")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for axis, kind in zip(axes, ["dropout", "noise"]):
        for dataset, group in stress_summary[stress_summary.kind == kind].groupby("dataset"):
            axis.plot(group.level, 100 * group.retained_alignment, "o-", label=dataset)
        axis.set_xlabel("Retained point probability" if kind == "dropout" else "Gaussian sigma xyz (m)")
        axis.set_ylabel("Alignment / original object support (%)")
        axis.legend(); axis.grid(alpha=.3)
    fig.savefig(figures / "stress_test.png", dpi=140); plt.close(fig)
    latency_table = pd.DataFrame(latency)
    latency_table.to_csv(out / "latency.csv", index=False, float_format="%.9f")
    latency_table.groupby(["dataset", "config"]).agg(
        repeats=("latency_ms", "count"), p50_ms=("latency_ms", "median"),
        p95_ms=("latency_ms", lambda s: s.quantile(.95)),
        mask_difference=("mask_difference", "max"), max_pixel_error=("max_pixel_error", "max")
    ).to_csv(out / "latency_summary.csv", float_format="%.9f")
    # Threshold selected ONLY on baseline calibration frames (every fifth frame).
    detection = []
    thresholds = {}
    for dataset, group in frame_table[frame_table.kind == "yaw"].groupby("dataset"):
        calibration = group[(group.split == "calibration") & (group.level == 0)].box_score.dropna()
        if calibration.empty:
            raise ValueError("Need supported baseline calibration frames, including yaw=0")
        threshold = max(0., float(calibration.quantile(.05)) - .05)
        thresholds[dataset] = threshold
        for level, evaluation in group[group.split == "evaluation"].groupby("level"):
            supported = evaluation.box_score.dropna()
            detection.append(dict(dataset=dataset, yaw_deg=level, threshold=threshold,
                                  calibration_frames=len(calibration), evaluation_frames=len(supported),
                                  n_unscorable=int(evaluation.box_score.isna().sum()),
                                  flagged=int((supported < threshold).sum()),
                                  flag_rate=float((supported < threshold).mean()) if len(supported) else np.nan))
    detection_table = pd.DataFrame(detection)
    detection_table.to_csv(out / "drift_detection.csv", index=False, float_format="%.9f")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for dataset, group in detection_table.groupby("dataset"):
        axes[0].plot(group.yaw_deg, group.flag_rate * 100, "o-", label=dataset)
        baseline = frame_table[(frame_table.dataset == dataset) & (frame_table.kind == "yaw")]
        scores = baseline.groupby("level").box_score.mean()
        axes[1].plot(scores.index, scores.values, "o-", label=dataset)
        axes[1].axhline(thresholds[dataset], linestyle="--", label=f"{dataset} threshold")
    axes[0].set_ylabel("Flagged evaluation frames (%)"); axes[1].set_ylabel("Box alignment score")
    for axis in axes:
        axis.set_xlabel("Yaw drift (degrees)"); axis.legend(fontsize=8); axis.grid(alpha=.3)
    fig.savefig(figures / "drift_detection.png", dpi=140); plt.close(fig)
    # Basic: three distance bands on same frame and same calibration.
    for dataset, (frame, points, _) in cached.items():
        for lo, hi in [(0, 15), (15, 30), (30, 70)]:
            save_overlay(frame, points, frame["calib"], figures / f"demo_{dataset}_{lo}_{hi}m.png",
                         f"{dataset}/{frame['frame_id']} baseline, depth {lo}-{hi} m", (lo, hi))
    # Geometry failure and a threshold blind spot selected from held-out frames.
    candidates = frame_table[(frame_table.kind == "yaw") & (frame_table.split == "evaluation") &
                             (frame_table.level >= 1) & frame_table.box_score.notna()].copy()
    candidates["threshold"] = candidates.dataset.map(thresholds)
    blind = candidates[candidates.box_score >= candidates.threshold]
    chosen = [candidates.sort_values("box_score").iloc[0]]
    if len(blind):
        chosen.append(blind.sort_values(["level", "box_score"], ascending=[False, False]).iloc[0])
    for number, row in enumerate(chosen, 1):
        root = next(root for root in args.data_roots if Path(root).name == row.dataset)
        frame = load_frame(root, row.frame_id)
        points, _ = prepare(frame, args.max_range, args.min_points)
        baseline, changed = [], []
        # Matplotlib comparison avoids creating temporary image files.
        for axis_name, yaw in [(baseline, 0), (changed, row.level)]:
            uv, depth, _ = project_velo_to_image(points, perturb_extrinsic(frame["calib"], yaw_deg=yaw), frame["image"].shape)
            vis = overlay_points(frame["image"], uv, depth, radius=1)
            for obj in frame["labels"]:
                if obj.type in CLASSES:
                    vis = draw_box2d(vis, obj.bbox, label=obj.type)
            axis_name.append(vis)
        fig, axes = plt.subplots(2, 1, figsize=(12, 8), constrained_layout=True)
        axes[0].imshow(cv2.cvtColor(baseline[0], cv2.COLOR_BGR2RGB)); axes[0].set_title("Baseline yaw 0 deg")
        axes[1].imshow(cv2.cvtColor(changed[0], cv2.COLOR_BGR2RGB))
        axes[1].set_title(f"{row.dataset}/{row.frame_id}: yaw {row.level:g} deg, score={row.box_score:.3f}, threshold={row.threshold:.3f}")
        for axis in axes: axis.axis("off")
        fig.savefig(figures / f"fail_{number:02d}_{'geometry' if number == 1 else 'score_blind_spot'}.png", dpi=140)
        plt.close(fig)
    pd.DataFrame(chosen).to_csv(out / "failure_cases.csv", index=False, float_format="%.9f")
    synthetic_audit(args.synthetic_root, out, figures)
    print("Completed benchmark; numeric CSVs are deterministic except latency.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-roots", nargs="+", default=["data/kitti_mini", "data/nuscenes_mini_subset"])
    parser.add_argument("--frames", nargs="+", help="Optional exact frame IDs; default all frames")
    parser.add_argument("--yaw-levels", nargs="+", type=float, default=[-3, -2, -1, -.5, 0, .5, 1, 2, 3])
    parser.add_argument("--seed", type=int, default=21)
    parser.add_argument("--max-range", type=float, default=70.)
    parser.add_argument("--min-points", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--out-dir", default="results")
    parser.add_argument("--synthetic-root", default="data/synthetic")
    parser.add_argument("--self-check", action="store_true", help="Run analytic geometry checks and exit")
    args = parser.parse_args()
    if args.self_check:
        self_check()
        return
    if args.repeats < 20 or args.min_points < 1 or args.max_range <= 0 or 0 not in args.yaw_levels:
        parser.error("Require repeats>=20, min-points>=1, max-range>0 and a baseline yaw=0")
    if not any(level >= 1 for level in args.yaw_levels):
        parser.error("Include a positive yaw>=1 for failure analysis")
    benchmark(args)


if __name__ == "__main__":
    main()
