"""Run one arm on one dataset for one or more seeds, writing raw attempts + a manifest.

  python -m benchmarking.run --arm base --dataset math500
  python -m benchmarking.run --arm base --dataset calculus --limit 5          # smoke test
  python -m benchmarking.run --resume benchmarking/results/raw/<run_id>       # continue an interrupted run

Output: benchmarking/results/raw/<run_id>/{manifest.json, attempts.jsonl}
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path

from .arms import ArmUnavailable, RunContext, get_arm
from .datasets import load_dataset
from .grading import grade, grader_info
from .ollama_client import OllamaClient, OllamaError

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
PROMPTS = HERE / "prompts"
DEFAULT_OUT = HERE / "results" / "raw"

# Headline protocol (see README "Design choices"). Change via flags, never silently.
DEFAULTS = {
    "model": "gemma3:4b",
    "seeds": [0, 1, 2],
    "temperature": 0.2,
    "num_ctx": 8192,
    "num_predict": 4096,
    "prompt": "base_v1",
}


def _git(*args: str) -> str | None:
    try:
        return subprocess.check_output(["git", "-C", str(REPO), *args], text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def machine_info() -> dict:
    info = {"os": platform.platform(), "python": platform.python_version(),
            "cpu": platform.processor() or platform.machine(), "hostname": platform.node()}
    if shutil.which("nvidia-smi"):
        try:
            info["gpu"] = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                text=True, timeout=10).strip()
        except (OSError, subprocess.SubprocessError):
            pass
    return info


def load_prompt(name: str) -> tuple[str, str]:
    path = PROMPTS / f"{name}.txt"
    if not path.exists():
        raise SystemExit(f"no prompt {path}")
    text = path.read_text(encoding="utf-8")
    if "{problem}" not in text:
        raise SystemExit(f"{path} must contain {{problem}}")
    return text, hashlib.sha256(text.encode()).hexdigest()


def _slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9.]+", "-", s).strip("-")


def _done_keys(path: Path) -> set[tuple[str, int]]:
    if not path.exists():
        return set()
    with open(path, encoding="utf-8") as f:
        return {(r["problem_id"], r["seed"]) for r in map(json.loads, f) if r}


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", default="base", help="base | sympy | rag (pending) | full (pending)")
    ap.add_argument("--dataset", default="math500", help="math500 | calculus | path/to/set.jsonl")
    ap.add_argument("--model", default=DEFAULTS["model"])
    ap.add_argument("--seeds", type=int, nargs="+", default=DEFAULTS["seeds"])
    ap.add_argument("--temperature", type=float, default=DEFAULTS["temperature"])
    ap.add_argument("--num-ctx", type=int, default=DEFAULTS["num_ctx"])
    ap.add_argument("--num-predict", type=int, default=DEFAULTS["num_predict"])
    ap.add_argument("--prompt", default=DEFAULTS["prompt"], help="file name in benchmarking/prompts/ without .txt")
    ap.add_argument("--limit", type=int, help="only the first N problems (smoke tests; not for reported numbers)")
    ap.add_argument("--ollama-host", help="default: OLLAMA_HOST or http://localhost:11434")
    ap.add_argument("--out", default=str(DEFAULT_OUT), help="parent folder for run folders")
    ap.add_argument("--run-name", help="folder name (default: <dataset>-<arm>-<model>-<timestamp>)")
    ap.add_argument("--resume", help="existing run folder; reuses its manifest settings and skips finished attempts")
    return ap.parse_args(argv)


def main(argv=None) -> None:
    sys.stdout.reconfigure(errors="replace")
    args = parse_args(argv)

    if args.resume:
        run_dir = Path(args.resume)
        manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
        cfg = manifest["config"]
    else:
        cfg = {
            "arm": args.arm, "dataset": args.dataset, "model": args.model, "seeds": args.seeds,
            "options": {"temperature": args.temperature, "num_ctx": args.num_ctx, "num_predict": args.num_predict},
            "prompt": args.prompt, "limit": args.limit,
        }
        stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        run_id = args.run_name or f"{_slug(cfg['dataset'])}-{cfg['arm']}-{_slug(cfg['model'])}-{stamp}"
        run_dir = Path(args.out) / run_id
        manifest = None

    client = OllamaClient(args.ollama_host)
    try:
        ollama_version = client.version()
        model_info = client.model_info(cfg["model"])
    except OllamaError as exc:
        raise SystemExit(f"{exc}\nStart Ollama and pull the model first: ollama pull {cfg['model']}")

    dataset = load_dataset(cfg["dataset"])
    problems = dataset.problems[: cfg["limit"]] if cfg["limit"] else dataset.problems
    prompt_text, prompt_sha = load_prompt(cfg["prompt"])
    ctx = RunContext(client=client, model=cfg["model"], options=cfg["options"],
                     prompt_template=prompt_text, prompt_name=cfg["prompt"])

    arm = get_arm(cfg["arm"])
    try:
        arm_notes = arm.setup(ctx)
    except ArmUnavailable as exc:
        raise SystemExit(str(exc))

    if manifest is not None:  # resuming: refuse to mix settings within one run
        for key, now in (("prompt_sha256", prompt_sha), ("dataset_sha256", dataset.sha256),
                         ("model_digest", model_info.get("digest"))):
            if manifest.get(key) != now:
                raise SystemExit(f"cannot resume: {key} changed since the run started")
    else:
        run_dir.mkdir(parents=True, exist_ok=False)
        manifest = {
            "run_id": run_dir.name,
            "started_at": dt.datetime.now().isoformat(timespec="seconds"),
            "command": sys.argv,
            "config": cfg,
            "arm": {"name": arm.name, "description": arm.description, **arm_notes},
            "git": {"sha": _git("rev-parse", "HEAD"), "branch": _git("branch", "--show-current"),
                    "dirty": bool(_git("status", "--porcelain"))},
            "ollama": {"host": client.host, "version": ollama_version},
            "model": model_info,
            "model_digest": model_info.get("digest"),
            "prompt_text": prompt_text,
            "prompt_sha256": prompt_sha,
            "dataset": {"name": dataset.name, "source": dataset.source, "n_total": len(dataset.problems),
                        "n_run": len(problems)},
            "dataset_sha256": dataset.sha256,
            "grader": grader_info(),
            "machine": machine_info(),
        }
        (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    attempts_path = run_dir / "attempts.jsonl"
    done = _done_keys(attempts_path)
    total = len(problems) * len(cfg["seeds"])
    print(f"run {run_dir}  ({arm.name} on {dataset.name}, {len(problems)} problems x seeds {cfg['seeds']}; "
          f"{len(done)} already done)")

    count = len(done)
    with open(attempts_path, "a", encoding="utf-8") as out:
        for seed in cfg["seeds"]:  # seed-major: a stopped run still has whole seeds
            correct = 0
            for p in problems:
                if (p.id, seed) in done:
                    continue
                attempt = arm.solve(p, seed, ctx).to_dict()
                g = grade(p.answer_latex, attempt["final_text"], truncated=attempt["truncated"],
                          error=attempt["error"])
                correct += g["outcome"] == "correct"
                row = {"problem_id": p.id, "seed": seed, "meta": p.meta, "gold": p.answer_latex,
                       **attempt, **g}
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                out.flush()
                count += 1
                print(f"[{count}/{total}] seed {seed} {p.id}: {g['outcome']}"
                      f" ({attempt['latency_s']:.1f}s, route={attempt['route']})")
            print(f"seed {seed}: {correct} correct this session")

    manifest["finished_at"] = dt.datetime.now().isoformat(timespec="seconds")
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\ndone. summarize with:  python -m benchmarking.report {run_dir}")


if __name__ == "__main__":
    main()
