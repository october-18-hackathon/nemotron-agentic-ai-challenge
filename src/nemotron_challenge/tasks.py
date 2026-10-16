"""Teams implement these stages with their coding agents."""
from pathlib import Path
from typing import Any

from .config import Settings

STAGES = ("extract", "match", "compare", "agent", "site", "evaluate")


def _unfinished(stage: str) -> None:
    raise NotImplementedError(
        f"TODO: implement the {stage} milestone. Read docs/milestones.md#{stage}. "
        "The starter has no result for this stage."
    )


def extract(settings: Settings, work_dir: Path) -> dict[str, Any]:
    """TODO: define your extraction and entry segmentation process."""
    _unfinished("extract")


def match(settings: Settings, work_dir: Path) -> dict[str, Any]:
    """TODO: define how your team finds and verifies article matches."""
    _unfinished("match")


def compare(settings: Settings, work_dir: Path) -> dict[str, Any]:
    """TODO: define your vocabulary, frequency, or topic comparison."""
    _unfinished("compare")


def agent(settings: Settings, work_dir: Path) -> dict[str, Any]:
    """TODO: implement your Hermes tools and evidence checks."""
    _unfinished("agent")


def site(settings: Settings, work_dir: Path) -> dict[str, Any]:
    """TODO: build your LaTeX documents and static website."""
    _unfinished("site")


def evaluate(settings: Settings, work_dir: Path) -> dict[str, Any]:
    """TODO: review your results and document successes and failures."""
    _unfinished("evaluate")
