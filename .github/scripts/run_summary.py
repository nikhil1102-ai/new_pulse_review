"""
Emit a Markdown briefing for the GitHub Actions run summary.

Printed to ``$GITHUB_STEP_SUMMARY`` by the generate job so that whoever is
asked to approve delivery can read the actual report first, in the Actions UI,
without downloading the artifact.

Reads the handoff file the generate stage wrote. Prints a clear notice instead
of failing when that file is absent, since this runs with ``if: always()`` and
must never mask the real error from a failed pipeline step.
"""

import json
import os
import sys

PENDING = os.path.join("data", "reports", "pending_delivery.json")


def _kb(path: str) -> str:
    """Human-readable size for a file that may not exist."""
    try:
        return f"{os.path.getsize(path) / 1024:.0f} KB"
    except OSError:
        return "missing"


def main() -> int:
    if not os.path.exists(PENDING):
        print("## Weekly Review Pulse\n")
        print("No report was produced - the pipeline did not reach the ")
        print("generate stage. See the failing step above for the cause.")
        return 0

    with open(PENDING, "r", encoding="utf-8") as f:
        run = json.load(f)

    pdf_path = run.get("pdf_path") or ""
    status = "passed" if run.get("validation_passed") else "passed with warnings"
    errors = run.get("validation_errors") or []

    print(f"## {run.get('product', 'Weekly')} - Review Pulse\n")
    print(
        f"**Window:** {run.get('week_start', '?')} to {run.get('week_end', '?')}  "
    )
    print(f"**Generated:** {run.get('generated_at', 'unknown')} UTC\n")

    print("| Metric | Value |")
    print("| --- | --- |")
    print(f"| Reviews analysed | {run.get('review_count', 0):,} |")
    print(f"| Themes found | {run.get('theme_count', 0)} |")
    print(f"| Validation | {status} |")
    print(f"| Detailed PDF | {os.path.basename(pdf_path) or 'not built'} "
          f"({_kb(pdf_path)}) |")
    print()

    if errors:
        print("<details><summary>Validation warnings "
              f"({len(errors)})</summary>\n")
        for err in errors[:20]:
            print(f"- {err}")
        print("\n</details>\n")

    report = (run.get("report_markdown") or "").strip()
    if report:
        print("---\n")
        print(report)
        print()

    fee = (run.get("fee_explainer") or "").strip()
    if fee:
        print("---\n")
        print("### Fee explainer\n")
        if run.get("fee_pain_point"):
            print(f"**Identified confusion:** {run['fee_pain_point']}\n")
        print(fee)
        print()

    print("---\n")
    print("> Download the **pulse-** artifact for the full PDF, which has the ")
    print("> theme table, charts, version breakdown and methodology.")
    print(">")
    print("> Approving the **Deliver** job uploads this PDF to Google Drive ")
    print("> and creates the email draft. Nothing has been sent yet.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
