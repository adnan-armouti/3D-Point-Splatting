"""all-scenes driver for RadarFields: serial-per-GPU, 2 scenes in parallel.

Six benchmark scenes:
    seq_0_frame_135
    seq_1_frame_185
    seq_1_frame_438
    seq_2_frame_160
    seq_2_frame_300
    seq_2_frame_105

Each scene = 400 iters (upstream default 800 catastrophically NaN-explodes the
HashGrid after ~700 iters in our 8-frame cascade regime — see deviation
list in metrics.json).
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

DEFAULT_MAX_ITERS = 400  # upstream default 800 NaN-explodes after ~700 iters

RADARFIELDS_PY = os.environ.get("RADARFIELDS_PYTHON", "python")
PYTHON = "python"

RESULTS_ROOT = os.path.join(_REPO, "baselines/radarfields/results")
WORKSPACE_ROOT = os.path.join(_REPO, "baselines/radarfields/upstream_results")


def _gpu_compute_busy(gpu_idx: int) -> bool:
    out = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid",
         f"--id={gpu_idx}", "--format=csv,noheader"],
        text=True,
    ).strip()
    return bool(out)


def _ensure_adapter_data(scene: str) -> None:
    upstream_data = os.path.join(_REPO, "baselines/radarfields/upstream/data", scene)
    if os.path.isfile(os.path.join(upstream_data, "adapter_manifest.json")):
        return
    print(f"[adapter] {scene}", flush=True)
    subprocess.run(
        [PYTHON, "-m", "baselines.radarfields.adapter.mm3dps_to_radarfields",
         "--scene", scene],
        cwd=_REPO,
        check=True,
        stdout=subprocess.DEVNULL,
    )


def _spawn_scene(scene: str, gpu: int, max_iters: int) -> subprocess.Popen:
    out_dir = os.path.join(RESULTS_ROOT, scene)
    workspace = os.path.join(WORKSPACE_ROOT, scene)
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, "run_log.txt")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["MM3DPS_PYTHON"] = PYTHON

    cmd = [
        RADARFIELDS_PY, "-m", "baselines.radarfields.runner.run_radarfields_scene",
        "--scene", scene,
        "--out-dir", out_dir,
        "--workspace", WORKSPACE_ROOT,
        "--max-iters", str(max_iters),
    ]
    log = open(log_path, "w")
    log.write(f"# {scene} on GPU {gpu}, started {time.strftime('%F %T')}\n")
    log.write(f"# cmd: {' '.join(cmd)}\n\n")
    log.flush()
    return subprocess.Popen(cmd, cwd=_REPO, env=env, stdout=log, stderr=subprocess.STDOUT)


def _wait_any(running: dict) -> tuple:
    while True:
        for scene, (proc, gpu) in list(running.items()):
            rc = proc.poll()
            if rc is not None:
                return scene, gpu, rc
        time.sleep(2)


def aggregate(scenes: List[str]) -> dict:
    rows, ras, rps = [], [], []
    for s in scenes:
        mp = os.path.join(RESULTS_ROOT, s, "metrics.json")
        if not os.path.isfile(mp):
            rows.append({"scene": s, "status": "MISSING"})
            continue
        m = json.load(open(mp))
        rows.append({
            "scene": s,
            "ra_corr": m["ra_corr"],
            "range_profile_corr": m.get("range_profile_corr"),
            "wall_time_seconds": m.get("wall_time_seconds"),
            "peak_gpu_mem_mib": m.get("peak_gpu_mem_mib"),
        })
        if isinstance(m["ra_corr"], (int, float)) and not (m["ra_corr"] != m["ra_corr"]):
            ras.append(float(m["ra_corr"]))
        if m.get("range_profile_corr") is not None and not (
            m["range_profile_corr"] != m["range_profile_corr"]
        ):
            rps.append(float(m["range_profile_corr"]))

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
        "baseline": "radarfields",
        "scenes": rows,
        "n_scenes": len(rows),
        "ra_corr": _agg(ras),
        "range_profile_corr": _agg(rps),
    }
    with open(os.path.join(RESULTS_ROOT, "aggregate.json"), "w") as f:
        json.dump(out, f, indent=2)
    return out


def main_cli() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="*", default=PHASE4_SCENES)
    ap.add_argument("--max-iters", type=int, default=DEFAULT_MAX_ITERS)
    ap.add_argument("--gpus", nargs="+", type=int, default=[0, 1])
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()

    print(f"RadarFields — {len(args.scenes)} scenes, {len(args.gpus)} GPUs", flush=True)
    print(f"Scenes: {args.scenes}", flush=True)
    print(f"GPUs:   {args.gpus}", flush=True)
    print(f"Iters:  {args.max_iters}", flush=True)

    for g in args.gpus:
        if _gpu_compute_busy(g):
            print(f"ABORT: GPU {g} compute-busy", flush=True)
            return 2

    for s in args.scenes:
        _ensure_adapter_data(s)

    todo = []
    for s in args.scenes:
        mp = os.path.join(RESULTS_ROOT, s, "metrics.json")
        if args.skip_existing and os.path.isfile(mp):
            print(f"[skip] {s}", flush=True)
            continue
        todo.append(s)
    print(f"To run: {todo}", flush=True)

    avail = list(args.gpus)
    pending = list(todo)
    running: dict = {}
    completed, failed = [], []
    t_start = time.time()

    while pending or running:
        while pending and avail:
            s = pending.pop(0)
            g = avail.pop(0)
            print(f"[launch] {s} on GPU {g}", flush=True)
            running[s] = (_spawn_scene(s, g, args.max_iters), g)
        if not running:
            break
        s, g, rc = _wait_any(running)
        elapsed = time.time() - t_start
        del running[s]
        avail.append(g)
        if rc == 0:
            completed.append(s)
            mp = os.path.join(RESULTS_ROOT, s, "metrics.json")
            if os.path.isfile(mp):
                m = json.load(open(mp))
                ra = m.get("ra_corr"); rp = m.get("range_profile_corr"); wt = m.get("wall_time_seconds")
                ra_str = f"{ra:.3f}" if isinstance(ra, (int, float)) and ra == ra else str(ra)
                rp_str = f"{rp:.3f}" if isinstance(rp, (int, float)) and rp == rp else str(rp)
                print(f"[done ] {s} GPU{g} ra={ra_str} rp={rp_str} train={wt:.0f}s wall={elapsed:.0f}s",
                      flush=True)
        else:
            failed.append(s)
            print(f"[FAIL ] {s} GPU{g} rc={rc} (see results/{s}/run_log.txt)", flush=True)

    print(f"\nCompleted: {completed}\nFailed:    {failed}", flush=True)
    agg = aggregate(args.scenes)
    print("\nAggregate:")
    print(json.dumps(agg, indent=2), flush=True)
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main_cli())
