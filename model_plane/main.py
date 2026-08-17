"""CLI entry point — runs the Model Plane server."""

from __future__ import annotations

import uvicorn

from model_plane.config import settings


def cli() -> None:
    uvicorn.run(
        "model_plane.app:app",
        host=settings.host,
        port=settings.port,
        workers=settings.workers,
        log_level=settings.log_level,
        reload=settings.env == "development",
    )


if __name__ == "__main__":
    cli()
