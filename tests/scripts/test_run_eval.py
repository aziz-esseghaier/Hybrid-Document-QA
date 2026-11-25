"""Tests for the full evaluation runner."""

import importlib.util
import json
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("run_eval", ROOT / "scripts" / "run_eval.py")
run_eval = importlib.util.module_from_spec(spec)
sys.modules["run_eval"] = run_eval  # dataclasses look their module up
spec.loader.exec_module(run_eval)

CFG = {
    "role": "test", "text_source": "markdown",
    "configs": {key: f"configs/{key}.yaml" for key in ("text", "visual", "rerank", "fusion", "answer", "judges")},
}
CFGS = {
    "visual": {"name": "vis"}, "rerank": {"name": "rr"}, "fusion": {"results_name": "fus"},
    "answer": {"name": "answer", "pages": [5], "page_tokens": [1280]}, "judges": {"judge": "j"},
}


def write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_answers_and_grading_are_done_when_every_question_is_covered(tmp_path):
    answers, verdicts = tmp_path / "answers", tmp_path / "verdicts"
    assert not run_eval.answers_done(answers, ["5x1280"])
    write(answers / "records_5x1280.jsonl", [{"query_id": "q1"}])
    (answers / "summary.json").write_text(json.dumps({"questions": 2}))
    assert not run_eval.answers_done(answers, ["5x1280"])
    write(answers / "records_5x1280.jsonl", [{"query_id": "q1"}, {"query_id": "q2"}, {"query_id": "q2"}])
    assert run_eval.answers_done(answers, ["5x1280"])
    write(verdicts / "verdicts_5x1280.jsonl", [{"query_id": "q1"}])
    assert not run_eval.judge_done(answers, verdicts, ["5x1280"])
    write(verdicts / "verdicts_5x1280.jsonl", [{"query_id": "q1"}, {"query_id": "q2"}])
    assert run_eval.judge_done(answers, verdicts, ["5x1280"])


def test_steps_run_in_order_and_skip_finished_results(tmp_path):
    results, data, text = tmp_path / "results", tmp_path / "data", tmp_path / "text"
    steps = run_eval.plan_steps(CFG, CFGS, ["a", "b"], results, data, text)
    names = [step.name for step in steps]
    assert names == [
        "judgments", "visual a", "visual b", "text search a", "text search b", "rerank a", "rerank b", "fusion",
        "answers a", "answers b", "grading a", "grading b", "answer accuracy",
    ]
    assert not any(step.done() for step in steps)
    (results / "vis_eval" / "a").mkdir(parents=True)
    (results / "vis_eval" / "a" / "metrics.json").write_text("{}")
    assert [step.name for step in steps if step.done()] == ["visual a"]
    grading = steps[names.index("grading a")]
    assert grading.datasets == [] and "--answers" in grading.commands[0]
    assert str(text / "answer" / "a") in grading.commands[0]


def test_visual_cleanup_frees_the_page_vectors(tmp_path):
    data = tmp_path / "data"
    (data / "derived" / "visual" / "a" / "model" / "pages").mkdir(parents=True)
    steps = run_eval.plan_steps(CFG, CFGS, ["a"], tmp_path / "results", data, tmp_path / "text")
    next(step for step in steps if step.name == "visual a").cleanup()
    assert not (data / "derived" / "visual" / "a").exists()


def test_preparation_builds_page_text_only_once(tmp_path):
    commands = run_eval.preparation("a", CFG, tmp_path)
    assert [Path(cmd[1]).stem for cmd in commands] == ["download_data", "build_chunks"]
    assert commands[1][-2:] == ["--source", "markdown"]
    write(tmp_path / "derived" / "chunks" / "a" / "pages.jsonl", [{"page_id": "a/1", "text": ""}])
    assert [Path(cmd[1]).stem for cmd in run_eval.preparation("a", CFG, tmp_path)] == ["download_data"]


def test_run_stops_at_the_deadline_and_fails_loudly():
    start = time.monotonic()
    assert not run_eval.run([sys.executable, "-c", "import time; time.sleep(30)"], time.monotonic() + 1)
    assert time.monotonic() - start < 20
    assert run_eval.run([sys.executable, "-c", "pass"], time.monotonic() + 60)
    assert not run_eval.run([sys.executable, "-c", "pass"], time.monotonic() - 1)
    with pytest.raises(SystemExit):
        run_eval.run([sys.executable, "-c", "raise SystemExit(3)"], time.monotonic() + 60)


def test_pack_and_unpack_merge_the_folders(tmp_path):
    results, text = tmp_path / "results", tmp_path / "text"
    write(results / "x" / "rows.jsonl", [{"query_id": "q"}])
    write(text / "answer" / "a.jsonl", [{"answer": "secret"}])
    run_eval.pack({"results": results, "text": text}, tmp_path / "state.tar.gz")
    new_results, new_text = tmp_path / "new_results", tmp_path / "new_text"
    write(new_results / "kept.jsonl", [{"query_id": "k"}])
    run_eval.unpack(tmp_path / "state.tar.gz", {"results": new_results, "text": new_text})
    assert (new_results / "x" / "rows.jsonl").read_text() == (results / "x" / "rows.jsonl").read_text()
    assert (new_results / "kept.jsonl").exists() and (new_text / "answer" / "a.jsonl").exists()


def test_restore_accepts_a_packed_or_an_unpacked_state(tmp_path):
    saved = tmp_path / "saved"
    write(saved / "results" / run_eval.MANIFEST.removesuffix(".json") / "x.jsonl", [])
    (saved / "results" / run_eval.MANIFEST).write_text("{}")
    write(saved / "text" / "answer" / "a.jsonl", [{"answer": "secret"}])
    roots = {"results": tmp_path / "r1", "text": tmp_path / "t1"}
    run_eval.restore_from(saved, roots)  # unpacked: folders as they were
    assert (tmp_path / "r1" / run_eval.MANIFEST).exists() and (tmp_path / "t1" / "answer" / "a.jsonl").exists()
    packed = tmp_path / "packed"
    packed.mkdir()
    run_eval.pack({"results": saved / "results", "text": saved / "text"}, packed / run_eval.STATE_FILE)
    roots = {"results": tmp_path / "r2", "text": tmp_path / "t2"}
    run_eval.restore_from(packed, roots)
    assert (tmp_path / "r2" / run_eval.MANIFEST).exists() and (tmp_path / "t2" / "answer" / "a.jsonl").exists()
    with pytest.raises(SystemExit):
        run_eval.restore_from(tmp_path / "t2", roots)


def test_not_found_recognizes_a_missing_dataset():
    class Response:
        status_code = 404

    class ApiError(Exception):
        response = Response()

    assert run_eval.not_found(ApiError("missing"))
    assert run_eval.not_found(Exception("404 Client Error: Not Found"))
    assert run_eval.not_found(Exception('POST failed with: {"errors":["Not found"],"error":{"code":5}}'))
    assert not run_eval.not_found(Exception("403 Forbidden"))
