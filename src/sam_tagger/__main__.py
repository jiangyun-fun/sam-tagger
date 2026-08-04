"""Entry point placeholder — wired up in the orchestrator stage."""

from __future__ import annotations


def main() -> None:
    """CLI entry point. Implemented once ``cli`` and ``lib`` land."""
    from sam_tagger.cli import app

    app()


if __name__ == "__main__":
    main()
