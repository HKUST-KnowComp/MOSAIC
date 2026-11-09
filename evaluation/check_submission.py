#!/usr/bin/env python3
"""Recompute a MOSAIC leaderboard submission and comment on the pull request.

The script reads the base repository's refusal phrases. It parses judge
responses the same way ``evaluation/judge/eval_judge.py`` does. It does not
execute anything from the pull request.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import ipaddress
import json
import math
import os
import re
import socket
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

MARKER = "<!-- mosaic-submission-check -->"
WANTED = ("samples.jsonl", "results.jsonl", "config.yaml")
MAX_BYTES = 80_000_000
URL_RE = re.compile(r"https://[^\s)>\]]+")


def load_refusal_patterns(repo_root: Path) -> list[str]:
    source = (repo_root / "evaluation" / "_common.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == "UNANSWERABLE_PATTERNS":
                value = ast.literal_eval(node.value)
                if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
                    raise ValueError("UNANSWERABLE_PATTERNS is not a list of strings")
                return value
    raise ValueError("UNANSWERABLE_PATTERNS was not found")


def assert_public_https(url: str) -> None:
    parsed = urlparse(url)
    host = parsed.hostname
    if parsed.scheme != "https" or not host:
        raise ValueError(f"refusing non-https URL: {url}")
    if host in {"localhost", "metadata.google.internal"} or host.endswith(".local"):
        raise ValueError(f"refusing local URL: {url}")
    try:
        port = parsed.port or 443
        infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise ValueError(f"could not resolve {host}") from exc
    for info in infos:
        address = ipaddress.ip_address(info[4][0])
        if not address.is_global:
            raise ValueError(f"refusing non-public address for {host}")


class _SafeRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[no-untyped-def]
        assert_public_https(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def download(url: str, dest: Path) -> None:
    assert_public_https(url)
    opener = urllib.request.build_opener(_SafeRedirect)
    request = urllib.request.Request(url, headers={"User-Agent": "mosaic-submission-check"})
    with opener.open(request, timeout=60) as response, dest.open("wb") as handle:
        total = 0
        while True:
            block = response.read(1024 * 1024)
            if not block:
                break
            total += len(block)
            if total > MAX_BYTES:
                raise ValueError(f"download exceeded {MAX_BYTES} bytes")
            handle.write(block)


def parse_json_robust(text: str) -> Any:
    text = text.strip()
    fenced = re.match(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        pass
    try:
        import json_repair
    except ImportError:
        raise
    return json_repair.loads(text)


def judge_score(raw: object) -> float:
    """Parse one stored judge response into a score in [0, 1]."""
    text = raw
    if isinstance(text, list):
        text = text[0] if text else ""
    if isinstance(text, str):
        loaded = parse_json_robust(text)
        if isinstance(loaded, list):
            loaded = loaded[0] if loaded else {}
        if isinstance(loaded, str):
            loaded = parse_json_robust(loaded)
    else:
        loaded = text
    if not isinstance(loaded, dict):
        raise ValueError("judge response is not an object")
    score = float(loaded.get("score", 0))
    if not math.isfinite(score):
        raise ValueError("judge score is not finite")
    return min(1.0, max(0.0, score))


def normalize(text: str) -> str:
    value = text.strip().lower()
    for prefix in ("answer:", "the answer is:", "response:"):
        if value.startswith(prefix):
            value = value[len(prefix) :].strip()
    return " ".join(value.split())


def is_refusal(prediction: str, patterns: list[str]) -> bool:
    normalized = normalize(prediction or "")
    return any(pattern in normalized for pattern in patterns)


def answerability_of(row: dict[str, Any]) -> str:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    value = metadata.get("answerability", row.get("answerability", ""))
    return str(value).strip().lower()


def prediction_of(row: dict[str, Any]) -> str:
    value = row.get("prediction", row.get("predicted_answer", ""))
    return "" if value is None else str(value)


def load_samples(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"samples.jsonl line {line_number} is not an object")
            rows.append(row)
    if not rows:
        raise ValueError("samples.jsonl is empty")
    return rows


def recompute(samples_path: Path, patterns: list[str]) -> dict[str, Any]:
    rows = load_samples(samples_path)
    answerable: list[float] = []
    unanswerable_scores: list[float] = []
    correct = false = 0
    missing_judge = 0
    parse_errors: list[str] = []
    for index, row in enumerate(rows, start=1):
        kind = answerability_of(row)
        raw_judge = row.get("judge_response")
        if raw_judge in (None, "", []):
            score = 0.0
            missing_judge += 1
        else:
            try:
                score = judge_score(raw_judge)
            except (ValueError, TypeError, json.JSONDecodeError) as exc:
                score = 0.0
                if len(parse_errors) < 5:
                    question = row.get("question_id", index)
                    parse_errors.append(f"{question}: {exc}")
        refused = is_refusal(prediction_of(row), patterns)
        if kind == "answerable":
            answerable.append(score)
            false += int(refused)
        elif kind == "unanswerable":
            unanswerable_scores.append(score)
            correct += int(refused)
        else:
            raise ValueError(f"row {index} has no answerability")
    if not answerable:
        raise ValueError("no answerable rows")
    return {
        "rows": len(rows),
        "answerable_count": len(answerable),
        "unanswerable_count": len(unanswerable_scores),
        "answerable": math.fsum(answerable) / len(answerable),
        "answerable_sum": math.fsum(answerable),
        "correct_refusal": correct,
        "false_refusal": false,
        "missing_judge": missing_judge,
        "parse_errors": parse_errors,
        "sha256": hashlib.sha256(samples_path.read_bytes()).hexdigest(),
    }


def aggregate_means(path: Path) -> dict[str, float] | None:
    if not path.exists():
        return None
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        return None
    tasks = document.get("tasks")
    if not isinstance(tasks, list) or not tasks or not isinstance(tasks[0], dict):
        return None
    metrics = tasks[0].get("metrics")
    if not isinstance(metrics, dict):
        return None
    found = {}
    for key in ("judge_score", "correct_refusal_rate", "false_refusal_rate"):
        value = metrics.get(key)
        if isinstance(value, dict) and isinstance(value.get("mean"), (int, float)):
            found[key] = float(value["mean"])
    return found or None


def _close(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=1e-9, abs_tol=1e-12)


def render(result: dict[str, Any], claimed: dict[str, float] | None) -> tuple[str, bool]:
    correct_rate = (
        result["correct_refusal"] / result["unanswerable_count"]
        if result["unanswerable_count"]
        else 0.0
    )
    false_rate = result["false_refusal"] / result["answerable_count"]
    lines = [
        MARKER,
        "Recomputed from the linked per-question file with the official judge parser and the refusal phrases in `evaluation/_common.py`.",
        "",
        "| Rate | Recomputed |",
        "| --- | ---: |",
        f"| Answerable score | {result['answerable']:.12f} ({result['answerable_sum']:g} / {result['answerable_count']}) |",
        f"| Correct refusal | {result['correct_refusal']} / {result['unanswerable_count']} |",
        f"| False refusal | {result['false_refusal']} / {result['answerable_count']} |",
        "",
        f"Rows: {result['rows']}. Samples SHA-256: `{result['sha256']}`.",
    ]
    if result["missing_judge"]:
        lines.append(f"Rows without a judge response, scored zero: {result['missing_judge']}.")
    matched = True
    if claimed is None:
        lines.append("No aggregate means were found in `results.jsonl` to compare.")
    else:
        checks = [
            ("answerable score", result["answerable"], claimed.get("judge_score")),
            ("correct refusal", correct_rate, claimed.get("correct_refusal_rate")),
            ("false refusal", false_rate, claimed.get("false_refusal_rate")),
        ]
        mismatches = [
            name
            for name, got, expected in checks
            if expected is not None and not _close(got, expected)
        ]
        if mismatches:
            matched = False
            lines.append("These differ from `results.jsonl`: " + ", ".join(mismatches) + ".")
        else:
            lines.append("These match the means in `results.jsonl`.")
    if result["rows"] != 923 or result["answerable_count"] != 784:
        matched = False
        lines.append("The leaderboard requires all 923 questions, with 784 answerable.")
    if result["parse_errors"]:
        matched = False
        lines.append("Judge responses that did not parse, scored zero:")
        lines.extend(f"- {item}" for item in result["parse_errors"])
    lines.extend(
        [
            "",
            "A maintainer adds the leaderboard row after review. This check does not merge the pull request or republish the files.",
        ]
    )
    return "\n".join(lines) + "\n", matched


def request_comment() -> str:
    return "\n".join(
        [
            MARKER,
            "Thanks for the submission. Please share these three files from the measured run:",
            "",
            "- `samples.jsonl`: per-question predictions and raw judge responses",
            "- `results.jsonl`: the aggregate written by the evaluator",
            "- `config.yaml`: the resolved config, with credentials removed",
            "",
            "A link is enough. The files are used only to recompute the row and will not be republished.",
        ]
    )


def artifact_urls(text: str) -> list[str]:
    found = []
    for match in URL_RE.findall(text):
        url = match.rstrip(".,;")
        path = urlparse(url).path.lower()
        if path.endswith(".zip") or any(path.endswith(name) for name in WANTED):
            found.append(url)
    return found


def collect_urls(body: str, comments: list[dict[str, Any]]) -> list[str]:
    texts = [body]
    for comment in comments:
        if MARKER in str(comment.get("body") or ""):
            continue
        texts.append(str(comment.get("body") or ""))
    found: list[str] = []
    for text in texts:
        for url in artifact_urls(text):
            if url not in found:
                found.append(url)
    return found


def materialize(urls: list[str], dest: Path) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    for url in urls:
        name = Path(urlparse(url).path).name
        target = dest / name
        download(url, target)
        if name.lower().endswith(".zip"):
            with zipfile.ZipFile(target) as archive:
                for info in archive.infolist():
                    base = Path(info.filename).name
                    if base not in WANTED or info.is_dir():
                        continue
                    extracted = dest / base
                    with archive.open(info) as source, extracted.open("wb") as handle:
                        handle.write(source.read(MAX_BYTES + 1))
                    if extracted.stat().st_size > MAX_BYTES:
                        raise ValueError(f"{base} exceeds {MAX_BYTES} bytes")
    if not (dest / "samples.jsonl").exists():
        raise ValueError("the linked files do not include samples.jsonl")
    return dest


def is_submission(title: str, body: str, files: list[str], urls: list[str]) -> bool:
    if urls:
        return True
    if "docs/leaderboard.json" in files:
        return True
    text = f"{title}\n{body}"
    return re.search(r"leaderboard|answerable", text, re.IGNORECASE) is not None


def github_request(method: str, path: str, payload: dict[str, Any] | None = None) -> Any:
    token = os.environ["GH_TOKEN"]
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(
        f"https://api.github.com{path}",
        data=data,
        method=method,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "mosaic-submission-check",
        },
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        raw = response.read()
    if not raw:
        return None
    return json.loads(raw)


def upsert_comment(repo: str, number: int, comments: list[dict[str, Any]], body: str) -> None:
    existing = next((comment for comment in comments if MARKER in str(comment.get("body") or "")), None)
    if existing is None:
        github_request("POST", f"/repos/{repo}/issues/{number}/comments", {"body": body})
        return
    github_request("PATCH", f"/repos/{repo}/issues/comments/{existing['id']}", {"body": body})


def run_from_pr(repo_root: Path, repo: str, number: int) -> int:
    pull = github_request("GET", f"/repos/{repo}/pulls/{number}")
    comments = github_request("GET", f"/repos/{repo}/issues/{number}/comments?per_page=100")
    files = github_request("GET", f"/repos/{repo}/pulls/{number}/files?per_page=100")
    if not isinstance(comments, list):
        comments = []
    changed = [item.get("filename", "") for item in files or []]
    urls = collect_urls(str(pull.get("body") or ""), comments)
    if not is_submission(str(pull.get("title") or ""), str(pull.get("body") or ""), changed, urls):
        return 0
    if not urls:
        upsert_comment(repo, number, comments, request_comment())
        return 0
    with tempfile.TemporaryDirectory() as temporary:
        try:
            folder = materialize(urls, Path(temporary))
            result = recompute(folder / "samples.jsonl", load_refusal_patterns(repo_root))
            body, matched = render(result, aggregate_means(folder / "results.jsonl"))
        except (ValueError, OSError, json.JSONDecodeError, zipfile.BadZipFile) as exc:
            body = f"{MARKER}\nCould not recompute the submission from the linked files: {exc}\n"
            matched = False
    upsert_comment(repo, number, comments, body)
    return 0 if matched else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--dir", type=Path)
    parser.add_argument("--from-pr", type=int)
    parser.add_argument("--repo", default=os.environ.get("GITHUB_REPOSITORY", ""))
    args = parser.parse_args()
    if args.from_pr is not None:
        if not args.repo:
            raise SystemExit("--repo is required with --from-pr")
        return run_from_pr(args.repo_root, args.repo, args.from_pr)
    if args.dir is None:
        raise SystemExit("pass --dir or --from-pr")
    result = recompute(args.dir / "samples.jsonl", load_refusal_patterns(args.repo_root))
    body, matched = render(result, aggregate_means(args.dir / "results.jsonl"))
    sys.stdout.write(body)
    return 0 if matched else 1


if __name__ == "__main__":
    raise SystemExit(main())
