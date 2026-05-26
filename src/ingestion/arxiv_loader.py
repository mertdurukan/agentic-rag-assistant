"""arXiv ingestion.

Why arXiv? Free, clean, well-structured metadata + abstract; matches the AI/ML
domain exactly. Focused categories (cs.AI/cs.LG/cs.CL) give us a meaningful
corpus without bloating the MVP.

Note: arXiv expects polite usage; the `arxiv` package handles rate limiting and
pagination on our behalf.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

from src.config import Settings, get_settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Paper:
    """The fields of a single arXiv paper that we need."""

    paper_id: str
    title: str
    abstract: str
    authors: list[str]
    categories: list[str]
    published: str
    pdf_url: str
    entry_url: str

    @property
    def text(self) -> str:
        """Raw text to chunk: title + abstract."""
        return f"{self.title}\n\n{self.abstract}"


def fetch_papers(settings: Settings | None = None) -> list[Paper]:
    """Fetch paper metadata + abstracts from the selected categories."""
    import arxiv

    settings = settings or get_settings()
    query = " OR ".join(f"cat:{c}" for c in settings.arxiv_category_list)
    logger.info("arXiv query: %s (max=%d)", query, settings.arxiv_max_results)

    search = arxiv.Search(
        query=query,
        max_results=settings.arxiv_max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
    )
    client = arxiv.Client(page_size=100, delay_seconds=3, num_retries=3)

    papers: list[Paper] = []
    for result in client.results(search):
        # Strip the version suffix from the arXiv id (e.g. 2401.01234v2 → 2401.01234)
        raw_id = result.entry_id.rsplit("/", 1)[-1]
        paper_id = raw_id.split("v")[0]
        papers.append(
            Paper(
                paper_id=paper_id,
                title=result.title.strip().replace("\n", " "),
                abstract=result.summary.strip().replace("\n", " "),
                authors=[a.name for a in result.authors],
                categories=list(result.categories),
                published=result.published.isoformat() if result.published else "",
                pdf_url=result.pdf_url or "",
                entry_url=result.entry_id,
            )
        )
    logger.info("%d papers fetched.", len(papers))
    return papers


def save_papers(papers: list[Paper], path: str | Path) -> Path:
    """Write papers to disk as JSONL (for idempotent ingestion)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for p in papers:
            f.write(json.dumps(asdict(p), ensure_ascii=False) + "\n")
    return path


def load_papers(path: str | Path) -> list[Paper]:
    """Read papers from disk."""
    path = Path(path)
    papers: list[Paper] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                papers.append(Paper(**json.loads(line)))
    return papers
