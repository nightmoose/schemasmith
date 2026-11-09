"""SchemaSmith Space. Mock mode does not load a model.

Gradio is Apache-2.0. This file is MIT, same as the rest of the repository.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if SRC.is_dir() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import gradio as gr  # noqa: E402

from schemasmith.space_demo import (  # noqa: E402
    check_output,
    contract_text,
    example_failure,
    labeled_json,
    record_by_id,
    sample_records,
    scorecard_markdown,
)

CSS = """
body, .gradio-container { background: #0d1117; color: #e6edf3; }
.banner { background: #3b2300; color: #ffd58a; text-align: center; padding: 6px; }
.panel { border: 1px solid #30363d; border-radius: 10px; padding: 12px; background: #161b22; }
label, .prose, p { color: #e6edf3 !important; }
"""


def _choices() -> list[str]:
    return [row.ticket_id for row in sample_records()]


def load_sample(ticket_id: str):
    record = record_by_id(ticket_id)
    return record.text, labeled_json(record), example_failure(record)


def extract(ticket_id: str, mock: bool):
    record = record_by_id(ticket_id)
    if mock:
        output = labeled_json(record)
        note = "Mock mode: labeled JSON for this sample. Not a model output."
    else:
        output = ""
        note = (
            "Model mode is not wired in this Space build. "
            "Leave mock mode on, or run the compare command on a GPU."
        )
    report = check_output(record, output) if output else {"banner": note, "errors": []}
    return output, note, _format_report(report)


def validate_only(ticket_id: str, output: str):
    record = record_by_id(ticket_id)
    return _format_report(check_output(record, output))


def refresh_scorecard(path: str):
    return scorecard_markdown(path.strip() or None)


def _format_report(report: dict) -> str:
    lines = [report.get("banner", "")]
    if "parsed" in report:
        lines.append(f"Parsed: {report['parsed']}")
        lines.append(f"Schema valid: {report['schema_valid']}")
        lines.append(f"Field hits: {report['field_hits']}")
        lines.append(f"Unsupported fields: {report['unsupported_fields'] or 'none'}")
    if report.get("errors"):
        lines.append("Errors:")
        lines.extend(f"- {item}" for item in report["errors"])
    return "\n".join(lines)


def build_demo() -> gr.Blocks:
    first = _choices()[0]
    text, labeled, failure = load_sample(first)
    with gr.Blocks(css=CSS, title="SchemaSmith") as demo:
        gr.HTML(
            "<div class='banner'>Mock mode uses labeled examples. "
            "The scorecard stays TBD until a compare results file is loaded.</div>"
        )
        gr.Markdown(
            "# SchemaSmith\n"
            "Turn a support ticket into JSON that matches a data contract. "
            "Core model: Tiny Aya Global (`CohereLabs/tiny-aya-global`). "
            "This page does not load it."
        )
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Data contract")
                gr.Code(contract_text(), language="yaml", label="support_ticket")
                gr.Markdown(
                    "Pipeline: CuratorKIT, then AlignTune SFT and DPO, then AuditKIT "
                    "and this contract validator. Artifacts are not published by this page."
                )
            with gr.Column():
                gr.Markdown("### Try a ticket")
                picker = gr.Dropdown(_choices(), value=first, label="Sample")
                mock = gr.Checkbox(value=True, label="Mock mode")
                ticket = gr.Textbox(value=text, lines=8, label="Ticket")
                with gr.Row():
                    extract_btn = gr.Button("Extract")
                    validate_btn = gr.Button("Validate only")
                    load_btn = gr.Button("Load sample")
                note = gr.Textbox(label="Note", value="Mock mode is on.")
                report = gr.Textbox(label="Contract check", lines=6)
            with gr.Column():
                gr.Markdown("### Example failure and labeled target")
                gr.Markdown(
                    "The left box is a planted failure. The right box is the label. "
                    "Neither box is a base-model or tuned-model measurement."
                )
                failure_box = gr.Code(failure, language="json", label="Planted failure")
                labeled_box = gr.Code(labeled, language="json", label="Labeled target")
        gr.Markdown("### Audit scorecard")
        results_path = gr.Textbox(
            label="Compare results JSON",
            placeholder="Leave empty to keep TBD",
        )
        scorecard = gr.Markdown(scorecard_markdown(None))
        load_results_btn = gr.Button("Load scorecard")

        load_btn.click(
            load_sample,
            inputs=[picker],
            outputs=[ticket, labeled_box, failure_box],
        )
        extract_btn.click(
            extract,
            inputs=[picker, mock],
            outputs=[labeled_box, note, report],
        )
        validate_btn.click(
            validate_only,
            inputs=[picker, labeled_box],
            outputs=[report],
        )
        load_results_btn.click(refresh_scorecard, inputs=[results_path], outputs=[scorecard])
    return demo


demo = build_demo()


if __name__ == "__main__":
    demo.launch()
