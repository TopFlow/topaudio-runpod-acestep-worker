import os
import json
import time
import signal
import subprocess
from pathlib import Path

import runpod
import requests
import pandas as pd

BASE_DIR = Path(os.getenv("TOPAUDIO_BASE_DIR", "/runpod-volume/topaudio_ai"))
DATASET_DIR = BASE_DIR / "commercial_v1_top50"
ACE_REPO_DIR = Path(os.getenv("ACESTEP_REPO_DIR", "/opt/ACE-Step-1.5"))
ACE_HOST = os.getenv("ACESTEP_HOST", "127.0.0.1")
ACE_PORT = int(os.getenv("ACESTEP_PORT", "8001"))
ACE_API_BASE = f"http://{ACE_HOST}:{ACE_PORT}"

LOG_DIR = BASE_DIR / "logs"
RESULTS_DIR = BASE_DIR / "acestep_results"
PREPARED_DATASET_DIR = BASE_DIR / "commercial_v1_top50_acestep"

server_process = None


def safe_path(rel_path: str) -> Path:
    rel_path = str(rel_path or ".").strip().lstrip("/")
    target = (BASE_DIR / rel_path).resolve()
    base = BASE_DIR.resolve()
    if not str(target).startswith(str(base)):
        raise ValueError("Path outside BASE_DIR is not allowed")
    return target


def action_health(job_input):
    import torch

    return {
        "ok": True,
        "base_dir": str(BASE_DIR),
        "dataset_dir": str(DATASET_DIR),
        "ace_repo_dir": str(ACE_REPO_DIR),
        "ace_repo_exists": ACE_REPO_DIR.exists(),
        "cuda_available": torch.cuda.is_available(),
        "torch_version": torch.__version__,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "ace_api_base": ACE_API_BASE,
    }


def action_inspect_path(job_input):
    try:
        target = safe_path(job_input.get("path", "."))
    except Exception as e:
        return {"ok": False, "error": str(e)}

    if not target.exists():
        return {"ok": False, "error": "Path does not exist", "path": str(target)}

    dirs = []
    files = []

    if target.is_dir():
        for x in sorted(target.iterdir()):
            item = {
                "name": x.name,
                "path": str(x),
                "is_dir": x.is_dir(),
                "size": x.stat().st_size if x.is_file() else None,
            }
            if x.is_dir():
                dirs.append(item)
            else:
                files.append(item)

    counts = {
        "wav": len(list(target.rglob("*.wav"))) if target.is_dir() else 0,
        "txt": len(list(target.rglob("*.txt"))) if target.is_dir() else 0,
        "caption_txt": len(list(target.rglob("*.caption.txt"))) if target.is_dir() else 0,
        "lyrics_txt": len(list(target.rglob("*.lyrics.txt"))) if target.is_dir() else 0,
        "json": len(list(target.rglob("*.json"))) if target.is_dir() else 0,
        "csv": len(list(target.rglob("*.csv"))) if target.is_dir() else 0,
        "all_files": len([x for x in target.rglob("*") if x.is_file()]) if target.is_dir() else 1,
    }

    return {
        "ok": True,
        "path": str(target),
        "is_dir": target.is_dir(),
        "counts": counts,
        "dirs": dirs[:100],
        "files": files[:100],
    }


def action_prepare_ace_dataset_layout(job_input):
    """
    ACE-Step docs recognize:
    - audio files
    - {filename}.txt as lyrics
    - {filename}.caption.txt as caption
    - {filename}.json annotation metadata

    Our dataset has:
    commercial_v1_top50/audio/{stem}.wav
    commercial_v1_top50/captions/{stem}.txt

    This action creates a flat ACE-friendly folder:
    commercial_v1_top50_acestep/{stem}.wav
    commercial_v1_top50_acestep/{stem}.caption.txt
    commercial_v1_top50_acestep/{stem}.lyrics.txt empty, because tracks are instrumental
    commercial_v1_top50_acestep/{stem}.json with caption metadata
    """
    import shutil

    src_dataset = safe_path(job_input.get("dataset_path", "commercial_v1_top50"))
    dst = safe_path(job_input.get("out_path", "commercial_v1_top50_acestep"))

    audio_dir = src_dataset / "audio"
    captions_dir = src_dataset / "captions"
    manifest_path = src_dataset / "manifest.csv"

    if not audio_dir.exists():
        return {"ok": False, "error": f"Audio dir not found: {audio_dir}"}
    if not captions_dir.exists():
        return {"ok": False, "error": f"Captions dir not found: {captions_dir}"}

    dst.mkdir(parents=True, exist_ok=True)

    manifest = None
    if manifest_path.exists():
        manifest = pd.read_csv(manifest_path)

    processed = []
    errors = []

    for wav in sorted(audio_dir.glob("*.wav")):
        stem = wav.stem
        cap = captions_dir / f"{stem}.txt"

        if not cap.exists():
            errors.append({"stem": stem, "error": f"caption not found: {cap}"})
            continue

        caption = cap.read_text(encoding="utf-8", errors="ignore").strip()

        dst_wav = dst / f"{stem}.wav"
        dst_caption = dst / f"{stem}.caption.txt"
        dst_lyrics = dst / f"{stem}.lyrics.txt"
        dst_json = dst / f"{stem}.json"

        shutil.copy2(wav, dst_wav)
        dst_caption.write_text(caption + "\n", encoding="utf-8")

        # Instrumental dataset: no lyrics.
        dst_lyrics.write_text("", encoding="utf-8")

        meta = {
            "filename": f"{stem}.wav",
            "caption": caption,
            "lyrics": "",
            "instrumental": True,
            "language": "instrumental",
        }

        if manifest is not None and "stem" in manifest.columns:
            row = manifest[manifest["stem"].astype(str) == stem]
            if len(row):
                r = row.iloc[0].to_dict()
                for k in ["bpm", "key", "source_title", "genre_1", "mood_top_5", "instruments_top_7"]:
                    if k in r and pd.notna(r[k]):
                        meta[k] = str(r[k])

        dst_json.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

        processed.append(stem)

    return {
        "ok": True,
        "src_dataset": str(src_dataset),
        "prepared_dataset": str(dst),
        "processed_count": len(processed),
        "errors": errors[:20],
        "counts": {
            "wav": len(list(dst.glob("*.wav"))),
            "caption_txt": len(list(dst.glob("*.caption.txt"))),
            "lyrics_txt": len(list(dst.glob("*.lyrics.txt"))),
            "json": len(list(dst.glob("*.json"))),
        }
    }


def find_possible_entrypoints():
    if not ACE_REPO_DIR.exists():
        return []

    candidates = []
    for pattern in [
        "start_gradio_ui.sh",
        "acestep/acestep_v15_pipeline.py",
        "app.py",
        "server.py",
        "api*.py",
        "*api*.py",
        "scripts/**/*.py",
    ]:
        for p in ACE_REPO_DIR.glob(pattern):
            if p.is_file():
                candidates.append(str(p))

    return sorted(set(candidates))[:200]


def action_list_ace_repo(job_input):
    entries = []
    if ACE_REPO_DIR.exists():
        for x in sorted(ACE_REPO_DIR.iterdir()):
            entries.append({
                "name": x.name,
                "is_dir": x.is_dir(),
                "size": x.stat().st_size if x.is_file() else None
            })

    return {
        "ok": True,
        "ace_repo_dir": str(ACE_REPO_DIR),
        "entries": entries[:200],
        "possible_entrypoints": find_possible_entrypoints(),
    }


def action_start_acestep_api(job_input):
    """
    First attempt: inspect available scripts and try starting known API/server if present.
    This is intentionally conservative. If ACE-Step changed entrypoints, output will tell us.
    """
    global server_process

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / "acestep_api.log"

    if server_process is not None and server_process.poll() is None:
        return {
            "ok": True,
            "message": "ACE-Step API already running",
            "pid": server_process.pid,
            "api_base": ACE_API_BASE,
            "log_path": str(log_path),
        }

    # ACE-Step 1.5 API launch.
    # Official docs mention: uv run acestep --enable-api --port 8001
    # In Docker we try installed CLI first, then python module fallbacks.
    no_init = bool(job_input.get("no_init", True))

    commands = [
        ["acestep-api", "--host", ACE_HOST, "--port", str(ACE_PORT)] + (["--no-init"] if no_init else []),
        ["acestep-api", "--port", str(ACE_PORT)] + (["--no-init"] if no_init else []),
        ["python3.11", "-m", "acestep.api_server", "--host", ACE_HOST, "--port", str(ACE_PORT)] + (["--no-init"] if no_init else []),
        ["python3.11", "-m", "acestep.api_server", "--port", str(ACE_PORT)] + (["--no-init"] if no_init else []),
    ]

    attempted = []

    for cmd in commands:
        # Only try if file/module seems plausible.
        attempted.append(" ".join(cmd))

        try:
            log_f = open(log_path, "a", buffering=1)
            log_f.write("\n\n=== Starting command: " + " ".join(cmd) + " ===\n")

            server_process = subprocess.Popen(
                cmd,
                cwd=str(ACE_REPO_DIR),
                stdout=log_f,
                stderr=log_f,
                preexec_fn=os.setsid,
            )

            time.sleep(8)

            if server_process.poll() is None:
                return {
                    "ok": True,
                    "message": "Started ACE-Step API candidate",
                    "pid": server_process.pid,
                    "cmd": cmd,
                    "api_base": ACE_API_BASE,
                    "log_path": str(log_path),
                }

        except Exception as e:
            attempted.append(f"ERROR: {e}")

    return {
        "ok": False,
        "error": "Could not start ACE-Step API from guessed commands",
        "attempted": attempted,
        "possible_entrypoints": find_possible_entrypoints(),
        "log_path": str(log_path),
    }


def action_api_health(job_input):
    urls = [
        f"{ACE_API_BASE}/health",
        f"{ACE_API_BASE}/v1/health",
        f"{ACE_API_BASE}/docs",
        f"{ACE_API_BASE}/openapi.json",
    ]

    results = []

    for url in urls:
        try:
            r = requests.get(url, timeout=5)
            results.append({
                "url": url,
                "status_code": r.status_code,
                "text": r.text[:500],
            })
        except Exception as e:
            results.append({
                "url": url,
                "error": str(e),
            })

    return {
        "ok": True,
        "api_base": ACE_API_BASE,
        "results": results,
    }


def action_tail_log(job_input):
    rel = job_input.get("log", "logs/acestep_api.log")
    path = safe_path(rel) if not str(rel).startswith("/opt") else Path(rel)

    if not path.exists():
        return {"ok": False, "error": f"log not found: {path}"}

    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    n = int(job_input.get("lines", 120))

    return {
        "ok": True,
        "path": str(path),
        "tail": "\n".join(lines[-n:]),
    }

def restart_acestep_api_volume_root():
    import os
    import signal
    import subprocess
    import time
    from pathlib import Path

    base_dir = Path(os.environ.get("TOPAUDIO_BASE_DIR", "/runpod-volume/topaudio_ai"))
    log_dir = base_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    # Stop existing acestep-api processes safely
    subprocess.run(
        ["bash", "-lc", "pgrep -f 'acestep-api' || true"],
        capture_output=True,
        text=True,
        timeout=10,
    )

    pids = subprocess.run(
        ["pgrep", "-f", "acestep-api"],
        capture_output=True,
        text=True,
        timeout=10,
    ).stdout.strip().splitlines()

    killed = []
    for pid_text in pids:
        try:
            pid = int(pid_text.strip())
            os.kill(pid, signal.SIGTERM)
            killed.append(pid)
        except Exception:
            pass

    time.sleep(2)

    env = os.environ.copy()
    env["PYTHONPATH"] = "/opt/ACE-Step-1.5:" + env.get("PYTHONPATH", "")
    env["HF_HOME"] = str(base_dir / "hf_cache")
    env["TRANSFORMERS_CACHE"] = str(base_dir / "hf_cache")
    env["TORCH_HOME"] = str(base_dir / "torch_cache")
    env["ACESTEP_NO_INIT"] = "true"

    log_path = log_dir / "acestep_api_volume_root.log"
    log_file = open(log_path, "ab")

    proc = subprocess.Popen(
        [
            "acestep-api",
            "--host", "127.0.0.1",
            "--port", "8001",
            "--no-init",
        ],
        cwd=str(base_dir),
        env=env,
        stdout=log_file,
        stderr=log_file,
        start_new_session=True,
    )

    time.sleep(5)

    return {
        "ok": True,
        "pid": proc.pid,
        "killed": killed,
        "cwd": str(base_dir),
        "api_base": "http://127.0.0.1:8001",
        "log_path": str(log_path),
    }

def action_run_shell(job_input):
    """
    Limited diagnostic shell runner for ACE-Step setup.
    Use only for safe read-only/debug commands.
    """
    cmd = job_input.get("cmd")
    timeout = int(job_input.get("timeout", 60))

    if not cmd:
        return {"ok": False, "error": "cmd is required"}

    blocked = ["rm ", "shutdown", "reboot", "mkfs", ":(){", "dd ", "sudo "]
    if any(b in cmd for b in blocked):
        return {"ok": False, "error": "blocked command"}

    try:
        r = subprocess.run(
            cmd,
            shell=True,
            cwd=str(ACE_REPO_DIR),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return {
            "ok": r.returncode == 0,
            "returncode": r.returncode,
            "stdout": r.stdout[-8000:],
            "stderr": r.stderr[-8000:],
            "cmd": cmd,
        }
    except subprocess.TimeoutExpired as e:
        return {
            "ok": False,
            "error": "timeout",
            "cmd": cmd,
            "stdout": (e.stdout or "")[-4000:] if isinstance(e.stdout, str) else "",
            "stderr": (e.stderr or "")[-4000:] if isinstance(e.stderr, str) else "",
        }

def handler(job):
    job_input = job.get("input", {})
    action = job_input.get("action", "health")

    if action == "health":
        return action_health(job_input)

    if action == "inspect_path":
        return action_inspect_path(job_input)

    if action == "prepare_ace_dataset_layout":
        return action_prepare_ace_dataset_layout(job_input)

    if action == "list_ace_repo":
        return action_list_ace_repo(job_input)

    if action == "start_acestep_api":
        return action_start_acestep_api(job_input)

    if action == "api_health":
        return action_api_health(job_input)

    if action == "tail_log":
        return action_tail_log(job_input)

    if action == "restart_api_volume_root":
        return restart_acestep_api_volume_root()

    if action == "run_shell":
        return action_run_shell(job_input)

    return {
        "ok": False,
        "error": f"Unknown action: {action}"
    }


runpod.serverless.start({"handler": handler})
