"""all-scenes driver: serial-per-GPU, 2 scenes in parallel across the 2 RTX 4090s.

Six benchmark scenes:
    seq_0_frame_135
    seq_1_frame_185
    seq_1_frame_438
    seq_2_frame_160
    seq_2_frame_300
    seq_2_frame_105

Each scene = full upstream-default training (max_steps=2000, init_num_pts=20000,
init_scale=0.5) on one RTX 4090 (~21 min/scene observed).
With 2 scenes in parallel, expected total wall time ~63 minutes for 6 scenes.

Per-scene workflow (one subprocess on a pinned GPU):
  1. nvidia-smi check on the chosen GPU (abort that scene if compute-busy).
  2. Run adapter (3DPS -> RadarSplat format) if not already cached.
  3. Run run_radarsplat_scene.py with CUDA_VISIBLE_DEVICES pinned.
  4. finalize_metrics.py auto-runs at end (writes metrics.json + PNGs).

Driver guarantees:
  - Each scene runs on exactly one GPU at a time.
  - Each GPU runs exactly one scene at a time (no sharing).
  - GPU occupancy by graphics processes (Xorg / gnome-shell / Cursor on GPU 1)
    is OK; what we forbid is another COMPUTE process on the GPU we want.

Outputs:
  baselines/radarsplat/results/<scene>/metrics.json
  baselines/radarsplat/results/<scene>/{gt,rasterized}_ra_{linear,dB}.png
  baselines/radarsplat/results/<scene>/run_log.txt
  baselines/radarsplat/results/aggregate.json
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from typing import List

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))

PHASE4_SCENES: List[str] = [
    "seq_0_frame_135",
    "seq_1_frame_185",
    "seq_1_frame_438",
    "seq_2_frame_160",
    "seq_2_frame_300",
    "seq_2_frame_105",
]

DEFAULT_MAX_STEPS = 2000
DEFAULT_INIT_NUM_PTS = 20000

RADARSPLAT_PY = os.environ.get("RADARSPLAT_PYTHON", "python")
PYTHON = "python"

DATA_ROOT = os.path.join(_REPO, "baselines/radarsplat/data_radarsplat")
RESULTS_ROOT = os.path.join(_REPO, "baselines/radarsplat/results")
UPSTREAM_RESULTS_ROOT = os.path.join(_REPO, "baselines/radarsplat/upstream_results")


def _gpu_compute_busy(gpu_idx: int) -> bool:
    """True if any compute (CUDA) process is active on the given GPU.

    Display-only processes (Xorg, gnome-shell, Cursor) are listed under
    --query-graphics-apps and are tolerated; we only abort on compute apps.
    """
    out = subprocess.check_output(
        ["nvidia-smi",
         "--query-compute-apps=gpu_uuid,pid",
         f"--id={gpu_idx}",
         "--format=csv,noheader"],
        text=True,
    ).strip()
    return bool(out)


def _ensure_adapter_data(scene: str) -> None:
    out_dir = os.path.join(DATA_ROOT, scene)
    manifest = os.path.join(out_dir, "adapter_manifest.json")
    if os.path.isfile(manifest):
        # Already adapted.
        return
    print(f"[adapter] {scene}", flush=True)
    subprocess.run(
        [PYTHON, "-m", "baselines.radarsplat.adapter.mm3dps_to_radarsplat",
         "--scene", scene,
         "--out-root", DATA_ROOT],
        cwd=_REPO,
        check=True,
        stdout=subprocess.DEVNULL,
    )


def _spawn_scene(scene: str, gpu: int, max_steps: int, init_num_pts: int) -> subprocess.Popen:
    out_dir = os.path.join(RESULTS_ROOT, scene)
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, "run_log.txt")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["MM3DPS_PYTHON"] = PYTHON

    cmd = [
        RADARSPLAT_PY, "-m", "baselines.radarsplat.runner.run_radarsplat_scene",
        "--scene", scene,
        "--data-root", DATA_ROOT,
        "--result-root", UPSTREAM_RESULTS_ROOT,
        "--out-dir", out_dir,
        "--max-steps", str(max_steps),
        "--init-num-pts", str(init_num_pts),
    ]
    log = open(log_path, "w")
    log.write(f"# {scene} on GPU {gpu}, started {time.strftime('%F %T')}\n")
    log.write(f"# cmd: {' '.join(cmd)}\n\n")
    log.flush()
    return subprocess.Popen(cmd, cwd=_REPO, env=env, stdout=log, stderr=subprocess.STDOUT)


def _wait_any(running: dict) -> tuple[str, int, int]:
    """Block until any subprocess exits. Returns (scene, gpu, returncode)."""
    while True:
        for scene, (proc, gpu) in list(running.items()):
            rc = proc.poll()
            if rc is not None:
                return scene, gpu, rc
        time.sleep(2)


def aggregate_results(scenes: List[str]) -> dict:
    rows = []
    ra_corrs = []
    rp_corrs = []
    for scene in scenes:
        mp = os.path.join(RESULTS_ROOT, scene, "metrics.json")
        if not os.path.isfile(mp):
            rows.append({"scene": scene, "status": "MISSING"})
            continue
        m = json.load(open(mp))
        rows.append({
            "scene": scene,
            "ra_corr": m["ra_corr"],
            "range_profile_corr": m.get("range_profile_corr"),
            "wall_time_seconds": m.get("wall_time_seconds"),
            "peak_gpu_mem_mib": m.get("peak_gpu_mem_mib"),
        })
        ra_corrs.append(float(m["ra_corr"]))
        if m.get("range_profile_corr") is not None:
            rp_corrs.append(float(m["range_profile_corr"]))

    import statistics

    def _agg(xs):
        if not xs:
            return None
        return {
            "mean": statistics.mean(xs),
            "stdev": statistics.stdev(xs) if len(xs) > 1 else 0.0,
            "n": len(xs),
        }

    out = {
        "baseline": "radarsplat",
        "scenes": rows,
        "n_scenes": len(rows),
        "ra_corr": _agg(ra_corrs),
        "range_profile_corr": _agg(rp_corrs),
    }
    agg_path = os.path.join(RESULTS_ROOT, "aggregate.json")
    os.makedirs(os.path.dirname(agg_path), exist_ok=True)
    with open(agg_path, "w") as f:
        json.dump(out, f, indent=2)
    return out


def main_cli() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="*", default=PHASE4_SCENES)
    ap.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    ap.add_argument("--init-num-pts", type=int, default=DEFAULT_INIT_NUM_PTS)
    ap.add_argument("--gpus", nargs="+", type=int, default=[0, 1])
    ap.add_argument("--skip-existing", action="store_true",
                    help="Skip scenes whose results/<scene>/metrics.json already exists.")
    args = ap.parse_args()

    print(f"RadarSplat — {len(args.scenes)} scenes, {len(args.gpus)} GPUs", flush=True)
    print(f"Scenes : {args.scenes}", flush=True)
    print(f"GPUs   : {args.gpus}", flush=True)
    print(f"Steps  : {args.max_steps} | init_num_pts: {args.init_num_pts}", flush=True)

    # GPU compute-occupancy gate (display-only = OK).
    for g in args.gpus:
        if _gpu_compute_busy(g):
            print(f"ABORT: GPU {g} has an active compute process. Clear and retry.", flush=True)
            return 2

    # Pre-build adapter data for all scenes (sequential, ~seconds/scene).
    for scene in args.scenes:
        _ensure_adapter_data(scene)

    # Optionally skip already-done scenes.
    todo = []
    for scene in args.scenes:
        mp = os.path.join(RESULTS_ROOT, scene, "metrics.json")
        if args.skip_existing and os.path.isfile(mp):
            print(f"[skip] {scene} (metrics.json exists)", flush=True)
            continue
        todo.append(scene)
    print(f"To run: {todo}", flush=True)

    # Parallel scheduler: at most one scene per GPU at a time.
    available_gpus = list(args.gpus)
    pending = list(todo)
    running: dict[str, tuple[subprocess.Popen, int]] = {}
    completed = []
    failed = []

    t_start = time.time()
    while pending or running:
        # Launch onto any free GPU.
        while pending and available_gpus:
            scene = pending.pop(0)
            gpu = available_gpus.pop(0)
            print(f"[launch] {scene} on GPU {gpu}", flush=True)
            proc = _spawn_scene(scene, gpu, args.max_steps, args.init_num_pts)
            running[scene] = (proc, gpu)

        if not running:
            break

        # Wait for one to finish.
        scene, gpu, rc = _wait_any(running)
        elapsed = time.time() - t_start
        del running[scene]
        available_gpus.append(gpu)
        if rc == 0:
            completed.append(scene)
            mp = os.path.join(RESULTS_ROOT, scene, "metrics.json")
            if os.path.isfile(mp):
                m = json.load(open(mp))
                ra = m.get("ra_corr")
                rp = m.get("range_profile_corr")
                wt = m.get("wall_time_seconds")
                print(f"[done ] {scene} GPU{gpu} ra={ra:.3f} rp={rp:.3f} "
                      f"train={wt:.0f}s wall_total={elapsed:.0f}s", flush=True)
            else:
                print(f"[done ] {scene} GPU{gpu} BUT no metrics.json", flush=True)
        else:
            failed.append(scene)
            print(f"[FAIL ] {scene} GPU{gpu} rc={rc} (see results/{scene}/run_log.txt)",
                  flush=True)

    print(f"\nCompleted: {completed}\nFailed   : {failed}", flush=True)
    agg = aggregate_results(args.scenes)
    print("\nAggregate:", flush=True)
    print(json.dumps(agg, indent=2), flush=True)
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main_cli())
