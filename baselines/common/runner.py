"""All-scenes driver for the baselines: one scene per GPU at a time.

Usage::

    python -m baselines.common.runner --baseline dart --gpus 0 1

For each scene the driver builds the adapter data if it is not cached, then
starts training under the baseline's own environment (set ``DART_PYTHON``,
``RADARFIELDS_PYTHON`` or ``RADARSPLAT_PYTHON``). Training ends by running
``baselines.common.finalize`` under the mm3DPS environment. Results go to
``baselines/<name>/results/<scene>/``, with ``results/aggregate.json`` on top.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
import subprocess
import time

from baselines.common.nvs_split import BENCHMARK_SCENES, REPO_ROOT


class BaselineRunner:
    """What every baseline shares. Subclasses set the names and the arguments."""

    name = ""
    title = ""
    python_env = ""
    adapter_module = ""
    scene_module = ""

    def __init__(self):
        root = os.path.join(REPO_ROOT, "baselines", self.name)
        self.root = root
        self.results_root = os.path.join(root, "results")
        self.scene_python = os.environ.get(self.python_env, "python")
        self.python = os.environ.get("MM3DPS_PYTHON", "python")

    # Per-baseline parts.

    def add_arguments(self, ap: argparse.ArgumentParser) -> None:
        """Add the baseline's training arguments."""

    def describe(self, args: argparse.Namespace) -> str:
        """One line naming the training settings."""
        return ""

    def adapter_manifest(self, scene: str) -> str:
        """File that exists once the adapter has run for ``scene``."""
        raise NotImplementedError

    def adapter_args(self, scene: str) -> list:
        return ["--scene", scene]

    def scene_args(self, scene: str, out_dir: str, args: argparse.Namespace) -> list:
        """Arguments of the per-scene training module."""
        raise NotImplementedError

    def update_scene_env(self, env: dict) -> None:
        """Change the environment of the per-scene training process."""

    # Shared scheduling.

    def ensure_adapter_data(self, scene: str) -> None:
        if os.path.isfile(self.adapter_manifest(scene)):
            return
        print(f"[adapter] {scene}", flush=True)
        subprocess.run(
            [self.python, "-m", self.adapter_module, *self.adapter_args(scene)],
            cwd=REPO_ROOT, check=True, stdout=subprocess.DEVNULL,
        )

    def scene_command(self, scene: str, out_dir: str, args: argparse.Namespace) -> list:
        return [self.scene_python, "-m", self.scene_module,
                *self.scene_args(scene, out_dir, args)]

    def spawn_scene(self, scene: str, gpu: int, args: argparse.Namespace) -> subprocess.Popen:
        out_dir = os.path.join(self.results_root, scene)
        os.makedirs(out_dir, exist_ok=True)

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        env["MM3DPS_PYTHON"] = self.python
        self.update_scene_env(env)

        cmd = self.scene_command(scene, out_dir, args)
        log = open(os.path.join(out_dir, "run_log.txt"), "w")
        log.write(f"# {scene} on GPU {gpu}, started {time.strftime('%F %T')}\n")
        log.write(f"# cmd: {' '.join(cmd)}\n\n")
        log.flush()
        return subprocess.Popen(cmd, cwd=REPO_ROOT, env=env,
                                stdout=log, stderr=subprocess.STDOUT)

    def metrics_path(self, scene: str) -> str:
        return os.path.join(self.results_root, scene, "metrics.json")

    def aggregate(self, scenes: list) -> dict:
        rows, ras, rps = [], [], []
        for scene in scenes:
            if not os.path.isfile(self.metrics_path(scene)):
                rows.append({"scene": scene, "status": "MISSING"})
                continue
            with open(self.metrics_path(scene)) as f:
                m = json.load(f)
            rows.append({
                "scene": scene,
                "ra_corr": m["ra_corr"],
                "range_profile_corr": m.get("range_profile_corr"),
                "wall_time_seconds": m.get("wall_time_seconds"),
                "peak_gpu_mem_mib": m.get("peak_gpu_mem_mib"),
            })
            if _is_number(m.get("ra_corr")):
                ras.append(float(m["ra_corr"]))
            if _is_number(m.get("range_profile_corr")):
                rps.append(float(m["range_profile_corr"]))

        out = {
            "baseline": self.name,
            "scenes": rows,
            "n_scenes": len(rows),
            "ra_corr": _summary(ras),
            "range_profile_corr": _summary(rps),
        }
        os.makedirs(self.results_root, exist_ok=True)
        with open(os.path.join(self.results_root, "aggregate.json"), "w") as f:
            json.dump(out, f, indent=2)
        return out

    def run(self, args: argparse.Namespace) -> int:
        print(f"{self.title} — {len(args.scenes)} scenes, {len(args.gpus)} GPUs", flush=True)
        print(f"Scenes: {args.scenes}", flush=True)
        print(f"GPUs:   {args.gpus}", flush=True)
        print(self.describe(args), flush=True)

        for gpu in args.gpus:
            if _gpu_compute_busy(gpu):
                print(f"ABORT: GPU {gpu} has an active compute process.", flush=True)
                return 2

        for scene in args.scenes:
            self.ensure_adapter_data(scene)

        pending = []
        for scene in args.scenes:
            if args.skip_existing and os.path.isfile(self.metrics_path(scene)):
                print(f"[skip] {scene}", flush=True)
                continue
            pending.append(scene)
        print(f"To run: {pending}", flush=True)

        available = list(args.gpus)
        running = {}
        completed, failed = [], []
        t_start = time.time()
        while pending or running:
            while pending and available:
                scene, gpu = pending.pop(0), available.pop(0)
                print(f"[launch] {scene} on GPU {gpu}", flush=True)
                running[scene] = (self.spawn_scene(scene, gpu, args), gpu)
            scene, gpu, rc = _wait_any(running)
            del running[scene]
            available.append(gpu)
            if rc == 0:
                completed.append(scene)
                self._report_done(scene, gpu, time.time() - t_start)
            else:
                failed.append(scene)
                print(f"[FAIL ] {scene} GPU{gpu} rc={rc} "
                      f"(see results/{scene}/run_log.txt)", flush=True)

        print(f"\nCompleted: {completed}\nFailed:    {failed}", flush=True)
        print("\nAggregate:")
        print(json.dumps(self.aggregate(args.scenes), indent=2), flush=True)
        return 0 if not failed else 1

    def _report_done(self, scene: str, gpu: int, elapsed: float) -> None:
        if not os.path.isfile(self.metrics_path(scene)):
            print(f"[done ] {scene} GPU{gpu} BUT no metrics.json", flush=True)
            return
        with open(self.metrics_path(scene)) as f:
            m = json.load(f)
        print(f"[done ] {scene} GPU{gpu} ra={_fmt(m.get('ra_corr'))} "
              f"rp={_fmt(m.get('range_profile_corr'))} "
              f"train={m.get('wall_time_seconds')}s wall={elapsed:.0f}s", flush=True)


class DartRunner(BaselineRunner):
    name = "dart"
    title = "DART"
    python_env = "DART_PYTHON"
    adapter_module = "baselines.dart.adapter.mm3dps_to_dart"
    scene_module = "baselines.dart.runner.run_dart_scene"

    def __init__(self):
        super().__init__()
        self.data_root = os.path.join(self.root, "data_dart")

    def add_arguments(self, ap):
        ap.add_argument("--epochs", type=int, default=300)
        ap.add_argument("--batch", type=int, default=8)

    def describe(self, args):
        return f"Epochs: {args.epochs} | batch: {args.batch}"

    def adapter_manifest(self, scene):
        return os.path.join(self.data_root, scene, "adapter_manifest.json")

    def adapter_args(self, scene):
        return ["--scene", scene, "--out-root", self.data_root]

    def scene_args(self, scene, out_dir, args):
        return ["--scene", scene, "--data-root", self.data_root, "--out-dir", out_dir,
                "--epochs", str(args.epochs), "--batch", str(args.batch)]

    def update_scene_env(self, env):
        # JAX needs the CUDA libraries of its own environment.
        python = shutil.which(self.scene_python)
        if python:
            env_lib = os.path.join(os.path.dirname(os.path.dirname(python)), "lib")
            env["LD_LIBRARY_PATH"] = f"{env_lib}:{env.get('LD_LIBRARY_PATH', '')}"


class RadarFieldsRunner(BaselineRunner):
    name = "radarfields"
    title = "RadarFields"
    python_env = "RADARFIELDS_PYTHON"
    adapter_module = "baselines.radarfields.adapter.mm3dps_to_radarfields"
    scene_module = "baselines.radarfields.runner.run_radarfields_scene"

    def add_arguments(self, ap):
        # The upstream default of 800 diverges to NaN after about 700 iterations here.
        ap.add_argument("--max-iters", type=int, default=400)

    def describe(self, args):
        return f"Iters: {args.max_iters}"

    def adapter_manifest(self, scene):
        return os.path.join(self.root, "upstream", "data", scene, "adapter_manifest.json")

    def scene_args(self, scene, out_dir, args):
        return ["--scene", scene, "--out-dir", out_dir,
                "--workspace", os.path.join(self.root, "upstream_results"),
                "--max-iters", str(args.max_iters)]


class RadarSplatRunner(BaselineRunner):
    name = "radarsplat"
    title = "RadarSplat"
    python_env = "RADARSPLAT_PYTHON"
    adapter_module = "baselines.radarsplat.adapter.mm3dps_to_radarsplat"
    scene_module = "baselines.radarsplat.runner.run_radarsplat_scene"

    def __init__(self):
        super().__init__()
        self.data_root = os.path.join(self.root, "data_radarsplat")

    def add_arguments(self, ap):
        ap.add_argument("--max-steps", type=int, default=2000)
        ap.add_argument("--init-num-pts", type=int, default=20000)

    def describe(self, args):
        return f"Steps: {args.max_steps} | init_num_pts: {args.init_num_pts}"

    def adapter_manifest(self, scene):
        return os.path.join(self.data_root, scene, "adapter_manifest.json")

    def adapter_args(self, scene):
        return ["--scene", scene, "--out-root", self.data_root]

    def scene_args(self, scene, out_dir, args):
        return ["--scene", scene, "--data-root", self.data_root,
                "--result-root", os.path.join(self.root, "upstream_results"),
                "--out-dir", out_dir,
                "--max-steps", str(args.max_steps),
                "--init-num-pts", str(args.init_num_pts)]


RUNNERS = {cls.name: cls for cls in (DartRunner, RadarFieldsRunner, RadarSplatRunner)}


def _is_number(x) -> bool:
    return isinstance(x, (int, float)) and x == x


def _fmt(x) -> str:
    return f"{x:.3f}" if _is_number(x) else str(x)


def _summary(xs: list):
    if not xs:
        return None
    return {
        "mean": statistics.mean(xs),
        "stdev": statistics.stdev(xs) if len(xs) > 1 else 0.0,
        "n": len(xs),
    }


def _gpu_compute_busy(gpu: int) -> bool:
    """True if a compute process is on the GPU. Display processes are fine."""
    out = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid",
         f"--id={gpu}", "--format=csv,noheader"],
        text=True,
    ).strip()
    return bool(out)


def _wait_any(running: dict) -> tuple:
    """Block until any scene exits. Returns (scene, gpu, returncode)."""
    while True:
        for scene, (proc, gpu) in list(running.items()):
            rc = proc.poll()
            if rc is not None:
                return scene, gpu, rc
        time.sleep(2)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--baseline", required=True, choices=sorted(RUNNERS))
    known, _ = ap.parse_known_args()
    runner = RUNNERS[known.baseline]()
    ap.add_argument("--scenes", nargs="*", default=list(BENCHMARK_SCENES))
    ap.add_argument("--gpus", nargs="+", type=int, default=[0, 1])
    ap.add_argument("--skip-existing", action="store_true",
                    help="Skip scenes whose metrics.json already exists.")
    runner.add_arguments(ap)
    return runner.run(ap.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
