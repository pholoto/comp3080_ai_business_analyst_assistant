"""Configuration helpers for the AI Business Analyst assistant."""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = BASE_DIR.parent

# Default path to the DOCX capstone template (used by the report exporter)
DEFAULT_DOCX_TEMPLATE = (
    WORKSPACE_ROOT / "back_end" / "templates" / "VinUni-CECS-Capstone-Project-template.docx"
)


def get_template_path() -> Path:
    """Return the path to the report template, raising if it does not exist."""
    if not DEFAULT_DOCX_TEMPLATE.exists():
        raise FileNotFoundError(
            "Expected DOCX template not found at " f"{DEFAULT_DOCX_TEMPLATE}"
        )
    return DEFAULT_DOCX_TEMPLATE
