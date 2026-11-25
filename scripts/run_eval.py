"""Run the full evaluation on every dataset of a role (the test set by default), step by step. Resumable.

Usage:
    python scripts/run_eval.py --text-dir DIR [--config configs/eval_test.yaml] [--hours 11] [--no-state] [--fresh]
        [--dry-run]

Steps, in order: the judgments; per dataset, visual search, the text search baselines and reranking; fusion over all
datasets; per dataset, answers and their grading; then the answer accuracy. Each step runs the usual scripts, each in
its own process, and is skipped once its results exist, so a rerun picks up where the last one stopped. Raw data and
page text are prepared once per session, only for the datasets a step needs.

The answers and the judge's explanations contain document text, so they go to --text-dir, never to RESULTS_ROOT.
Unless --no-state, RESULTS_ROOT and --text-dir are saved after every step to a private Kaggle Dataset
(<KAGGLE_USERNAME>/<state_dataset>), and restored at the start of a session that has no state yet: on Kaggle, rerun
the same cell until it reports that every step is done. --fresh starts without restoring. No step starts after
--hours; a step still running then is stopped and loses only its unfinished dataset.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import tarfile
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from doc_qa.core.config import REPO_ROOT, get_settings, load_yaml, role_name
from doc_qa.core.parallel import read_jsonl
from doc_qa.core.run_info import git_state

STATE_FILE = "state.tar.gz"
MANIFEST = "run_eval_state.json"  # in RESULTS_ROOT once a state is restored or saved
TEXT_SCRIPTS = ("embed_text", "index_text", "eval_text")  # the text search baselines, in order


@dataclass
class Step:
    """One unit of work.

    Attributes:
        name (str): E.g. "visual v3_hr".
        commands (list[list[str]]): Commands run in order.
        done (Callable[[], bool]): Whether the step's results exist.
        datasets (list[str]): Datasets whose raw data and page text the step needs.
        cleanup (Callable[[], None] | None): Run after the step, e.g. to free disk.
    """

    name: str
    commands: list[list[str]]
    done: Callable[[], bool]
    datasets: list[str] = field(default_factory=list)
    cleanup: Callable[[], None] | None = None


def script(name: str, *args: str) -> list[str]:
    """Build the command that runs one of the repo's scripts.

    Args:
        name (str): Script name without ".py".
        *args (str): Its arguments.

    Returns:
        list[str]: Command.
    """
    return [sys.executable, str(REPO_ROOT / "scripts" / f"{name}.py"), *args]


def query_ids(path: Path) -> set[str]:
    """Read the query ids of a JSONL file.

    Args:
        path (Path): File; a missing one has none.

    Returns:
        set[str]: Ids.
    """
    return {row["query_id"] for row in read_jsonl([path])}


def answers_done(folder: Path, settings: list[str]) -> bool:
    """Check that every setting's records cover the questions the answer summary lists.

    Args:
        folder (Path): Results folder of answer_questions.py for one dataset.
        settings (list[str]): Setting names, e.g. ["5x1280"].

    Returns:
        bool: True when every question is answered under every setting.
    """
    summary = folder / "summary.json"
    if not summary.exists():
        return False
    questions = json.loads(summary.read_text(encoding="utf-8"))["questions"]
    return all(len(query_ids(folder / f"records_{name}.jsonl")) >= questions for name in settings)


def judge_done(answers: Path, verdicts: Path, settings: list[str]) -> bool:
    """Check that every answer has a verdict.

    Args:
        answers (Path): Results folder of answer_questions.py for one dataset.
        verdicts (Path): Results folder of judge_answers.py for the same dataset.
        settings (list[str]): Setting names.

    Returns:
        bool: True when the answers are complete and all graded.
    """
    return answers_done(answers, settings) and all(
        query_ids(answers / f"records_{name}.jsonl") <= query_ids(verdicts / f"verdicts_{name}.jsonl")
        for name in settings
    )


def plan_steps(
    cfg: dict[str, Any], cfgs: dict[str, dict[str, Any]], datasets: list[str], results: Path, data: Path, text: Path
) -> list[Step]:
    """List every step of the evaluation, in order.

    Args:
        cfg (dict[str, Any]): Run config (configs/eval_test.yaml).
        cfgs (dict[str, dict[str, Any]]): The step configs it names, loaded, by key.
        datasets (list[str]): Dataset keys.
        results (Path): RESULTS_ROOT.
        data (Path): DATA_ROOT.
        text (Path): Folder for document text (answers and explanations).

    Returns:
        list[Step]: Steps.
    """
    paths, role = cfg["configs"], cfg["role"]
    visual, rerank, fusion, answer = cfgs["visual"], cfgs["rerank"], cfgs["fusion"], cfgs["answer"]
    judge = cfgs["judges"]["judge"]
    settings = [f"{p}x{t}" for p in sorted(answer["pages"]) for t in sorted(answer["page_tokens"])]
    answers_dir = results / f"{answer['name']}_answers"
    verdicts_dir = results / f"judge_{judge}"
    qrels_dir = results / role_name("qrels", role)

    steps = [Step(
        "judgments", [script("export_qrels", "--role", role)],
        lambda: all((qrels_dir / f"{d}.jsonl").exists() for d in datasets), list(datasets),
    )]
    for d in datasets:
        steps.append(Step(
            f"visual {d}",
            [
                script("embed_visual", "--config", paths["visual"], "--dataset", d),
                script("eval_visual", "--config", paths["visual"], "--dataset", d, "--local-only"),
            ],
            lambda d=d: (results / f"{visual['name']}_eval" / d / "metrics.json").exists(),
            [d],
            lambda d=d: shutil.rmtree(data / "derived" / "visual" / d, ignore_errors=True),  # page vectors: GBs
        ))
    for d in datasets:
        steps.append(Step(
            f"text search {d}",
            [script(name, "--config", paths["text"], "--dataset", d) for name in TEXT_SCRIPTS],
            lambda d=d: (results / "text_eval" / d / "metrics.json").exists(),
            [d],
        ))
    for d in datasets:
        steps.append(Step(
            f"rerank {d}",
            [script("rerank_pages", "--config", paths["rerank"], "--dataset", d)],
            lambda d=d: (results / rerank["name"] / d / "rankings_rerank.jsonl").exists(),
            [d],
        ))
    fused = results / fusion["results_name"]
    fused_files = [fused / "summary.json", *(fused / d / "rankings_fused.jsonl" for d in datasets)]
    steps.append(Step(
        "fusion",
        [script("eval_fusion", "--config", paths["fusion"])],
        lambda: all(path.exists() for path in fused_files),
    ))
    for d in datasets:
        steps.append(Step(
            f"answers {d}",
            [script("answer_questions", "--config", paths["answer"], "--dataset", d, "--out", str(text))],
            lambda d=d: answers_done(answers_dir / d, settings),
            [d],
        ))
    for d in datasets:
        steps.append(Step(
            f"grading {d}",
            [script(
                "judge_answers", "--config", paths["judges"], "--judge", judge, "--dataset", d,
                "--answers", str(text / answer["name"] / d), "--out", str(text),
            )],
            lambda d=d: judge_done(answers_dir / d, verdicts_dir / d, settings),
        ))
    steps.append(Step(
        "answer accuracy",
        [script(
            "answer_accuracy", "--role", role, "--judges-config", paths["judges"], "--answer-config", paths["answer"],
        )],
        lambda: False,  # cheap: rerun whenever reached
    ))
    return steps


def preparation(dataset: str, cfg: dict[str, Any], data: Path) -> list[list[str]]:
    """List the commands that get a dataset's raw data and page text into this session.

    Args:
        dataset (str): Dataset key.
        cfg (dict[str, Any]): Run config.
        data (Path): DATA_ROOT.

    Returns:
        list[list[str]]: Commands; download_data.py skips files already there.
    """
    commands = [script("download_data", "--only", dataset)]
    if not (data / "derived" / "chunks" / dataset / "pages.jsonl").exists():
        if cfg["text_source"] == "ocr":
            commands.append(script("ocr_store", "pull", "--dataset", dataset, "--force"))
        commands.append(script(
            "build_chunks", "--config", cfg["configs"]["text"], "--dataset", dataset, "--source", cfg["text_source"],
        ))
    return commands


def stop(proc: subprocess.Popen) -> None:
    """Stop a command and every process it started.

    Args:
        proc (subprocess.Popen): Process started in its own session on Linux.
    """
    posix = os.name == "posix"
    if posix:
        os.killpg(proc.pid, signal.SIGTERM)
    else:
        proc.terminate()
    try:
        proc.wait(timeout=60)
    except subprocess.TimeoutExpired:
        if posix:
            os.killpg(proc.pid, signal.SIGKILL)
        else:
            proc.kill()
        proc.wait()


def run(cmd: list[str], deadline: float) -> bool:
    """Run a command until it ends or the deadline passes.

    Args:
        cmd (list[str]): Command.
        deadline (float): `time.monotonic()` value after which the command is stopped.

    Returns:
        bool: True if it finished, False if the deadline stopped it (or had passed).
    """
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        return False
    print(f"\n$ {' '.join([Path(cmd[1]).name, *cmd[2:]])}", flush=True)
    proc = subprocess.Popen(cmd, cwd=REPO_ROOT, start_new_session=os.name == "posix")
    try:
        code = proc.wait(timeout=remaining)
    except subprocess.TimeoutExpired:
        stop(proc)
        return False
    if code:
        raise SystemExit(f"Exit code {code}: {' '.join(cmd[1:])}")
    return True


def pack(roots: dict[str, Path], archive: Path) -> None:
    """Pack folders into one archive, each under its name.

    Args:
        roots (dict[str, Path]): {name in the archive: folder}.
        archive (Path): .tar.gz file to write.
    """
    with tarfile.open(archive, "w:gz") as tar:
        for name, root in roots.items():
            if root.exists():
                tar.add(root, arcname=name)


def unpack(archive: Path, roots: dict[str, Path]) -> None:
    """Restore folders from an archive made by `pack`, merging into what is there.

    Args:
        archive (Path): .tar.gz file.
        roots (dict[str, Path]): {name in the archive: folder}.
    """
    with tempfile.TemporaryDirectory() as tmp, tarfile.open(archive) as tar:
        tar.extractall(tmp, filter="data")
        for name, root in roots.items():
            if (Path(tmp) / name).exists():
                shutil.copytree(Path(tmp) / name, root, dirs_exist_ok=True)


def restore_from(folder: Path, roots: dict[str, Path]) -> None:
    """Restore a saved state from a downloaded Dataset version, packed or already unpacked.

    Args:
        folder (Path): Downloaded version.
        roots (dict[str, Path]): {name in the archive: folder}.
    """
    archives = [path for path in sorted(folder.rglob(STATE_FILE)) if path.is_file()]
    if archives:
        unpack(archives[0], roots)
        return
    manifests = sorted(folder.rglob(MANIFEST))  # Kaggle unpacked the archive on upload
    if not manifests:
        raise SystemExit(f"No saved state in {folder}.")
    base = manifests[0].parent.parent
    for name, root in roots.items():
        if (base / name).is_dir():
            shutil.copytree(base / name, root, dirs_exist_ok=True)


def not_found(error: Exception) -> bool:
    """Tell whether a Kaggle API error means the Dataset does not exist.

    Args:
        error (Exception): Error raised by kagglehub.

    Returns:
        bool: True for a 404 or a "Not found" reply.
    """
    status = getattr(getattr(error, "response", None), "status_code", None)
    text = str(error).lower()
    return status == 404 or "404" in text or "not found" in text


class StateStore:
    """Save and restore the run's results and text as versions of a private Kaggle Dataset.

    Args:
        handle (str): `<username>/<dataset>`.
        roots (dict[str, Path]): {"results": RESULTS_ROOT, "text": text folder}.
    """

    def __init__(self, handle: str, roots: dict[str, Path]) -> None:
        self.handle, self.roots = handle, roots

    def restore(self) -> bool:
        """Copy the latest saved state into the folders.

        Returns:
            bool: False when nothing was saved yet.
        """
        import kagglehub

        try:
            folder = Path(kagglehub.dataset_download(self.handle))
        except Exception as error:
            if not_found(error):
                print(f"No saved state in {self.handle} yet: starting from the beginning.", flush=True)
                return False
            raise SystemExit(f"Could not restore {self.handle}: {error}. Use --fresh only if nothing was saved.")
        restore_from(folder, self.roots)
        manifest = json.loads((self.roots["results"] / MANIFEST).read_text(encoding="utf-8"))
        print(f"Restored the state saved at {manifest['saved_at']} ({manifest['note']}).", flush=True)
        return True

    def save(self, note: str) -> None:
        """Upload the folders as a new version; a failed upload only warns, and the next save retries.

        Args:
            note (str): What the state contains, e.g. "visual v3_hr done".
        """
        import kagglehub

        commit, dirty = git_state()
        manifest = {
            "saved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "note": note,
            "git_commit": commit, "git_dirty": dirty,
        }
        self.roots["results"].mkdir(parents=True, exist_ok=True)
        (self.roots["results"] / MANIFEST).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        with tempfile.TemporaryDirectory() as tmp:
            pack(self.roots, Path(tmp) / STATE_FILE)
            try:
                kagglehub.dataset_upload(self.handle, tmp, version_notes=note)
            except Exception as error:
                print(f"WARNING: could not save the state to {self.handle}: {error}", flush=True)
                return
        print(f"Saved the state to {self.handle}: {note}", flush=True)


def main() -> None:
    """Restore the state, run every step not done yet, and save the state after each one."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=Path("configs/eval_test.yaml"))
    parser.add_argument("--text-dir", type=Path, required=True, help="Folder for answers and explanations.")
    parser.add_argument("--hours", type=float, default=11.0, help="No step starts after this many hours.")
    parser.add_argument("--no-state", action="store_true", help="Don't restore or save the state on Kaggle.")
    parser.add_argument("--fresh", action="store_true", help="Don't restore the saved state.")
    parser.add_argument("--dry-run", action="store_true", help="List the steps left and stop.")
    args = parser.parse_args()

    deadline = time.monotonic() + args.hours * 3600
    settings = get_settings()
    cfg = load_yaml(args.config)
    cfgs = {key: load_yaml(Path(path)) for key, path in cfg["configs"].items()}
    specs = load_yaml(Path(cfg["datasets_config"]))["datasets"]
    datasets = [key for key, spec in specs.items() if spec["role"] == cfg["role"]]
    text_dir = args.text_dir.resolve()
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    store = None
    if not args.no_state:
        username = os.environ.get("KAGGLE_USERNAME")
        if not username:
            raise SystemExit("KAGGLE_USERNAME is not set (attach the Kaggle secrets and run kaggle_setup.py).")
        store = StateStore(f"{username}/{cfg['state_dataset']}", {"results": settings.results_root, "text": text_dir})
        if not args.fresh and not (settings.results_root / MANIFEST).exists():
            store.restore()

    steps = plan_steps(cfg, cfgs, datasets, settings.results_root, settings.data_root, text_dir)
    pending = [step for step in steps if not step.done()]
    print(f"{len(pending)} of {len(steps)} steps left: {', '.join(step.name for step in pending)}", flush=True)
    if args.dry_run:
        return

    prepared: set[str] = set()
    for step in pending:
        try:
            finished = True
            for dataset in step.datasets:
                if dataset not in prepared:
                    finished = all(run(cmd, deadline) for cmd in preparation(dataset, cfg, settings.data_root))
                    if not finished:
                        break
                    prepared.add(dataset)
            finished = finished and all(run(cmd, deadline) for cmd in step.commands)
        except SystemExit:
            if store:
                store.save(f"stopped by an error in {step.name}")
            raise
        if not finished:
            if store:
                store.save(f"stopped by the time limit in {step.name}")
            print(f"\nTime limit reached during '{step.name}'. Run the same cell again in a new session.")
            return
        if step.cleanup:
            step.cleanup()
        if store:
            store.save(f"{step.name} done")
    print("\nEvery step is done.")


if __name__ == "__main__":
    main()
