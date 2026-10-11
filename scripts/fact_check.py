"""
Fact-check a cover letter against your CV, with no model involved.

The same check the agent runs on every letter it writes. Useful on its own:
point it at a letter you wrote, or one a chatbot wrote for you, before you
send it.

    python scripts/fact_check.py \
        --resume samples/resume.example.md \
        --job samples/job.example.txt \
        --letter samples/letter.padded.txt

Exit code 0 means nothing at high severity was found; 1 means the letter
claims something the CV does not support and should not be sent as it is.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.logging_config import configure_logging  # noqa: E402
from app.services.fabrication_guard import verify_against_resume  # noqa: E402

# The guard logs its findings; this script prints them itself.
configure_logging(level="ERROR", fmt="json", log_dir="", component="fact-check")
# On Windows the logging setup can leave stdout wrapped by colorama, which
# strips colour from redirected output. This script handles colour itself.
sys.stdout = sys.__stdout__

_RED, _GREEN, _YELLOW, _DIM, _BOLD, _RESET = (
    ("\033[31m", "\033[32m", "\033[33m", "\033[2m", "\033[1m", "\033[0m")
    if sys.stdout.isatty() or os.environ.get("FORCE_COLOR")
    else ("", "", "", "", "", "")
)


def _read(path: str) -> str:
    return pathlib.Path(path).read_text(encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[1])
    parser.add_argument("--resume", required=True, help="Your CV as .md or .txt")
    parser.add_argument("--letter", required=True, help="The cover letter to check")
    parser.add_argument(
        "--job", default="", help="The job ad (enables the borrowed-skill check)"
    )
    parser.add_argument(
        "--company", default="", help="Employer name, so it is never flagged"
    )
    args = parser.parse_args()

    report = verify_against_resume(
        _read(args.letter),
        _read(args.resume),
        job_description=_read(args.job) if args.job else "",
        company=args.company,
    )

    print(f"{_DIM}fact check: {args.letter} against {args.resume}{_RESET}")
    if not report.flags:
        print(f"\n  {_GREEN}{_BOLD}PASS{_RESET}  every claim is backed by the CV\n")
        return 0

    high = [f for f in report.flags if f.severity == "high"]
    verdict = (
        f"{_RED}{_BOLD}DISCARD{_RESET}  {len(high)} claim{'s' if len(high) != 1 else ''} "
        "the CV does not support"
        if high
        else f"{_YELLOW}{_BOLD}REVIEW{_RESET}  worth a second look before sending"
    )
    print(f"\n  {verdict}\n")
    for flag in report.flags:
        colour = _RED if flag.severity == "high" else _YELLOW
        print(f"  {colour}[{flag.severity}]{_RESET} {flag.kind}: {flag.detail}")
    print()
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
