"""Run the FastAPI app via `python -m AI`."""
from __future__ import annotations

import uvicorn


def main() -> None:
    uvicorn.run("AI.app:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    main()
