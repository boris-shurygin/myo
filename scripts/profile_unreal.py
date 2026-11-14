"""Bounded desktop load experiments. No installation, training or auto-ramping."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

from unreal_profile_win32 import ProcessJob, system_sample, verify_job

ROOT = Path(__file__).resolve().parent.parent
GIB = 1024**3
STOP_FILE = ROOT / ".cache/unreal/STOP_PROFILE"


def gpu_sample():
    executable = shutil.which("nvidia-smi")
    if not executable:
        raise RuntimeError("nvidia-smi is required for the desktop load experiment")
    result = subprocess.run([
        executable, "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
        "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True, timeout=3,
        creationflags=subprocess.CREATE_NO_WINDOW)
    rows = [[float(value.strip()) for value in line.split(",")] for line in result.stdout.strip().splitlines()]
    if len(rows) != 1:
        raise RuntimeError("This experiment expects the single RTX 3060 desktop GPU")
    return dict(zip(("utilization_percent", "vram_used_mib", "vram_total_mib", "temperature_c", "power_w"), rows[0]))


def build_command(args, output):
    engine = Path(args.engine_root)
    version = json.loads((engine / "Engine/Build/Build.version").read_text())
    expected = json.loads((ROOT / "integrations/unreal/engine-version.json").read_text())
    if [version[key] for key in ("MajorVersion", "MinorVersion", "PatchVersion", "Changelist")] != [
        expected[key] for key in ("major", "minor", "patch", "changelist")]:
        raise RuntimeError("Unvalidated engine build")
    if args.stage in ("viewer", "render-bones"):
        preparation = ROOT / ".cache/unreal/User/Saved/Myo/bones-import.json"
        prepared = json.loads(preparation.read_text(encoding="utf-8")) if preparation.exists() else {}
        if not prepared.get("ok") or not prepared.get("mesh_files") or not all(Path(p).is_file() for p in prepared["mesh_files"]):
            raise RuntimeError("Persisted bone assets are missing. Run the prepare-bones stage first.")
        if args.view == "muscles" and (not prepared.get("muscle_mesh_files") or not all(Path(p).is_file() for p in prepared["muscle_mesh_files"])):
            raise RuntimeError("Muscle surfaces are missing. Export myo.unreal_muscles and run prepare-bones first.")
    project = ROOT / "integrations/unreal/MyoD1/MyoD1.uproject"
    cache = ROOT / ".cache/unreal"
    (cache / "ShaderWorking").mkdir(parents=True, exist_ok=True)
    os.environ["UE-LocalDataCachePath"] = str(cache / "DerivedDataCache")
    executable = "UnrealEditor.exe" if args.visible else "UnrealEditor-Cmd.exe"
    command = [str(engine / "Engine/Binaries/Win64" / executable), str(project),
               f"-UserDir={cache / 'User'}", f"-corelimit={args.cores}", "-NoSplash", "-NoLiveCoding",
               "-NoP4", "-NoSound", "-unattended", "-NoZen", "-ddc=InstalledNoZenLocalFallback",
               "-stdout", "-FullStdOutLogOutput", f"-abslog={output / 'engine.log'}"]
    command.append(f"-ShaderWorkingDir={cache / 'ShaderWorking'}")
    if args.stage in ("probe", "prepare-bones"):
        script = "probe_editor.py" if args.stage == "probe" else "prepare_bones.py"
        command += ["-NullRHI", "-run=pythonscript",
                    f"-script={ROOT / 'integrations/unreal/scripts' / script}"]
    else:
        # UnrealEdMisc reads only the FIRST post-project token as a map.
        map_name = f"/Game/Generated/{'Muscles' if args.view == 'muscles' else 'Bones'}_v1/L_{'Muscles' if args.view == 'muscles' else 'Bones'}_{args.pose}"
        command.insert(2, "/Engine/Maps/Entry" if args.stage == "render-entry" else map_name)
        if args.stage == "viewer":
            command.append("-game")
        exec_commands = f"t.MaxFPS {args.fps}"
        if args.stage == "viewer" and args.capture_frame:
            # Startup ExecCmds runs before asynchronous meshes/materials finish.
            # The native CSV frame command takes the image after warm-up.
            command += ["-csvCaptureFrames=160",
                        f"-csvExecCmds=120:HighResShot {args.width}x{args.height}"]
        command += ["-d3d11", "-windowed", f"-ResX={args.width}", f"-ResY={args.height}",
                    f"-ExecCmds={exec_commands}"]
        if not args.visible:
            command.append("-RenderOffscreen")
        if args.stage == "render-bones":
            os.environ["MYO_PREVIEW_REPORT"] = str(output / "preview.json")
            os.environ["MYO_PREVIEW_CAPTURE"] = "1" if args.capture_frame else "0"
            os.environ["MYO_PREVIEW_WIDTH"] = str(args.width)
            os.environ["MYO_PREVIEW_HEIGHT"] = str(args.height)
            os.environ["MYO_PREVIEW_VIEW"] = args.view
            command.append(f"-ExecutePythonScript={ROOT / 'integrations/unreal/scripts/preview_bones.py'}")
    return command


def violation(sample, args):
    if STOP_FILE.exists():
        return "manual_stop_file"
    if sample["ram_available_bytes"] < args.min_free_gib * GIB:
        return "system_ram_reserve"
    if sample["commit_headroom_bytes"] < args.min_free_gib * GIB:
        return "commit_reserve"
    if sample["gpu"]["vram_used_mib"] > args.max_vram_mib:
        return "vram_budget"
    if sample["gpu"]["temperature_c"] >= args.max_gpu_c:
        return "gpu_temperature"
    if sample.get("interval_s", 0) > 5:
        return "monitor_stall"
    job = sample.get("job", {})
    if max(job.get("private_bytes_sum", 0), job.get("peak_job_commit_bytes", 0)) > args.job_gib * GIB * 0.9:
        return "job_commit_reserve"
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("selftest", "baseline", "probe", "prepare-bones", "render-entry", "render-bones", "viewer"), default="baseline")
    parser.add_argument("--seconds", type=int, default=15)
    parser.add_argument("--cores", type=int, choices=(2, 3, 4), default=2)
    parser.add_argument("--job-gib", type=float, default=6)
    parser.add_argument("--min-free-gib", type=float, default=6)
    parser.add_argument("--max-vram-mib", type=float, default=8192)
    parser.add_argument("--max-gpu-c", type=float, default=75)
    parser.add_argument("--fps", type=int, choices=(10, 20, 30), default=10)
    parser.add_argument("--width", type=int, choices=(640, 960, 1280), default=640)
    parser.add_argument("--height", type=int, choices=(360, 540, 720), default=360)
    parser.add_argument("--visible", action="store_true", help="Open a rendering window under the same job limits")
    parser.add_argument("--capture-frame", action="store_true", help="Capture a bone frame (and verify the Play camera in editor mode)")
    parser.add_argument("--pose", type=int, choices=(15, 45, 90, 120), default=90)
    parser.add_argument("--view", choices=("bones", "muscles"), default="bones")
    parser.add_argument("--engine-root", default=r"C:\Program Files\Epic Games\UE_5.8")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.visible and not (args.stage.startswith("render-") or args.stage == "viewer"):
        parser.error("--visible requires a rendering stage or viewer")
    if args.capture_frame and args.stage not in ("render-bones", "viewer"):
        parser.error("--capture-frame requires render-bones or viewer")
    if not 1 <= args.seconds <= 120 or not 0.125 <= args.job_gib <= 8 or args.min_free_gib < 4:
        parser.error("Use 1..120 seconds, 0.125..8 GiB job commit and at least 4 GiB free RAM")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = args.output or ROOT / ".cache/unreal/profiles" / f"{stamp}-{args.stage}"
    output.mkdir(parents=True, exist_ok=False)
    report = {"stage": args.stage, "settings": vars(args) | {"output": str(output)},
              "started_utc": stamp, "status": "running", "samples": 0}
    (output / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    job = None
    records = []
    start = time.monotonic()
    try:
        if args.stage == "selftest":
            report["selftest"] = verify_job(output)
            report["status"] = "passed"
        else:
            previous = system_sample()
            initial = previous | {"gpu": gpu_sample()}
            if violation(initial, args):
                raise RuntimeError(f"Preflight refused launch: {violation(initial, args)}")
            if args.stage != "baseline":
                command = build_command(args, output)
                report["command"] = command
                job = ProcessJob(args.cores, int(args.job_gib * GIB))
                job.start(command, ROOT, output / "console.log")
                report["root_pid"] = job.pid
                report["affinity_mask"] = job.affinity_mask
            last_time, previous_job_cpu = start, 0
            with (output / "samples.jsonl").open("w", encoding="utf-8") as log:
                while True:
                    sample = system_sample()
                    sample["gpu"] = gpu_sample()
                    now = time.monotonic()
                    sample["elapsed_s"], sample["interval_s"] = now - start, now - last_time
                    sample["utc"] = datetime.now(timezone.utc).isoformat()
                    total = sample["cpu_total_ticks"] - previous["cpu_total_ticks"]
                    idle = sample["cpu_idle_ticks"] - previous["cpu_idle_ticks"]
                    sample["system_cpu_percent"] = 100 * (total - idle) / total if total else None
                    if job:
                        sample["job"] = job.sample()
                        sample["job_cpu_percent_of_machine"] = (sample["job"]["cpu_ticks"] - previous_job_cpu) / 1e7 / (now - last_time) / os.cpu_count() * 100
                        previous_job_cpu = sample["job"]["cpu_ticks"]
                    records.append(sample)
                    log.write(json.dumps(sample) + "\n")
                    log.flush()
                    reason = violation(sample, args)
                    if reason:
                        report["status"], report["stop_reason"] = "stopped_by_guard", reason
                        break
                    if job and sample["job"]["root_exit_code"] != 259:
                        report["editor_exit_code"] = sample["job"]["root_exit_code"]
                        report["status"] = "completed" if sample["job"]["root_exit_code"] == 0 else "editor_failed"
                        break
                    if now - start >= args.seconds:
                        report["status"] = "sampled" if not job else "stopped_at_time_budget"
                        break
                    previous, last_time = sample, now
                    time.sleep(0.75)
    except BaseException as error:
        report["status"], report["error"] = "failed", repr(error)
    finally:
        if job:
            try:
                job.stop()
                for _ in range(30):
                    state = job.sample()
                    if not state["active_processes"]:
                        break
                    time.sleep(0.1)
                report["active_processes_after_stop"] = state["active_processes"]
                if state["active_processes"]:
                    report["status"] = "cleanup_failed"
            except Exception as error:
                report["status"], report["cleanup_error"] = "cleanup_failed", repr(error)
            finally:
                job.close()
        report["wall_seconds"], report["samples"] = time.monotonic() - start, len(records)
        console_path = output / "console.log"
        preview_path = output / "preview.json"
        if preview_path.exists():
            report["preview"] = json.loads(preview_path.read_text(encoding="utf-8"))
        if args.stage == "render-bones" and not report.get("preview", {}).get("ok"):
            report["status"] = "preview_failed"
        if console_path.exists():
            console = console_path.read_text(encoding="utf-8", errors="replace")
            if args.stage == "viewer" and args.capture_frame:
                captures = re.findall(r'High resolution screenshot saved as (.+)', console)
                if captures:
                    source = Path(captures[-1].strip()).resolve()
                    screenshot_root = (ROOT / ".cache/unreal/User/Saved/Screenshots").resolve()
                    if source.is_relative_to(screenshot_root) and source.is_file():
                        shutil.copyfile(source, output / "frame.png")
                        report["frame"] = str(output / "frame.png")
                if not report.get("frame"):
                    report["status"] = "viewer_capture_failed"
            report["engine_initialized"] = "Engine is initialized. Leaving FEngineLoop::Init()" in console
            report["configured_shader_workers"] = [int(n) for n in re.findall(r"Using (\d+) local workers for shader compilation", console)]
            report["loaded_maps"] = re.findall(r'Cmd: MAP LOAD FILE="([^"]+)"', console)
            report["loaded_maps"] += re.findall(r'LogLoad: LoadMap:\s*(\S+)', console)
            if args.stage.startswith("render-") or args.stage == "viewer":
                expected_map = "Entry" if args.stage == "render-entry" else f"L_{'Muscles' if args.view == 'muscles' else 'Bones'}_{args.pose}"
                report["target_map_loaded"] = any(Path(path.split("?")[0]).stem == expected_map for path in report["loaded_maps"])
                if not report["target_map_loaded"] and report["status"] == "stopped_at_time_budget":
                    report["status"] = "target_not_loaded"
            report["fatal_log_lines"] = [line for line in console.splitlines() if any(marker in line for marker in (
                "Fatal error:", "Assertion failed:", "Ensure condition failed:", "GPU crashed", "DXGI_ERROR_DEVICE_REMOVED"))][:10]
            if report["fatal_log_lines"]:
                report["status"] = "engine_log_failure"
        if records:
            report["metrics"] = {
                "min_free_ram_gib": min(s["ram_available_bytes"] for s in records) / GIB,
                "max_system_cpu_percent": max(s["system_cpu_percent"] or 0 for s in records),
                "mean_system_cpu_percent": sum(s["system_cpu_percent"] or 0 for s in records) / len(records),
                "max_vram_mib": max(s["gpu"]["vram_used_mib"] for s in records),
                "max_gpu_percent": max(s["gpu"]["utilization_percent"] for s in records),
                "mean_gpu_percent": sum(s["gpu"]["utilization_percent"] for s in records) / len(records),
                "max_gpu_c": max(s["gpu"]["temperature_c"] for s in records),
                "max_interval_s": max(s["interval_s"] for s in records),
                "peak_job_commit_gib": max(s.get("job", {}).get("peak_job_commit_bytes", 0) for s in records) / GIB,
                "max_shader_workers": max(sum(p["name"] == "ShaderCompileWorker.exe" for p in s.get("job", {}).get("processes", [])) for s in records),
            }
        (output / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps({"output": str(output), **report}, indent=2))
    return 0 if report["status"] in ("passed", "sampled", "completed", "stopped_at_time_budget") else 1


if __name__ == "__main__":
    sys.exit(main())
