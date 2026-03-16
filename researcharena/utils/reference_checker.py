"""Verify whether references in a LaTeX paper are real or hallucinated.

Extracts citations from LaTeX source, then checks each one against
Semantic Scholar and CrossRef APIs. Reports which references are
verified, unverified, or likely fabricated.
"""

from __future__ import annotations

import re
import time
import urllib.parse
import urllib.request
import json
from dataclasses import dataclass, field
from pathlib import Path

from rich.console import Console
from rich.table import Table

console = Console()

# Rate limit: Semantic Scholar allows ~100 req/5min without API key
SEMANTIC_SCHOLAR_API = "https://api.semanticscholar.org/graph/v1/paper/search"
CROSSREF_API = "https://api.crossref.org/works"
REQUEST_DELAY = 0.5  # seconds between API calls


@dataclass
class ReferenceCheckResult:
    """Result of reference verification."""
    total: int
    verified: int
    unverified: int
    references: list[dict]  # each has: raw, title, authors, status, source, match_info

    @property
    def fake_rate(self) -> float:
        return self.unverified / self.total if self.total > 0 else 0.0


def check_references(paper_latex: str) -> ReferenceCheckResult:
    """Extract and verify all references from a LaTeX paper.

    Args:
        paper_latex: LaTeX source code

    Returns:
        ReferenceCheckResult with verification status per reference
    """
    raw_refs = _extract_references(paper_latex)
    console.print(f"  Found {len(raw_refs)} references to verify")

    if not raw_refs:
        return ReferenceCheckResult(total=0, verified=0, unverified=0, references=[])

    results = []
    verified = 0
    for i, ref in enumerate(raw_refs):
        title = ref.get("title", "")
        authors = ref.get("authors", "")
        year = ref.get("year", "")

        if not title:
            # Can't verify without a title — mark as parse_error, not unverified.
            # This avoids false rejections from parser bugs.
            results.append({**ref, "status": "parse_error", "reason": "no title extracted"})
            continue

        console.print(f"  [{i+1}/{len(raw_refs)}] Checking: {title[:60]}...")

        # Try Semantic Scholar first, then CrossRef
        match = None
        api_error = False
        try:
            match = _search_semantic_scholar(title, authors, year)
        except Exception:
            api_error = True

        if not match:
            try:
                match = _search_crossref(title, authors, year)
            except Exception:
                api_error = True

        if match:
            verified += 1
            results.append({
                **ref,
                "status": "verified",
                "source": match["source"],
                "matched_title": match["title"],
                "matched_authors": match.get("authors", ""),
                "matched_year": match.get("year", ""),
                "url": match.get("url", ""),
            })
        elif api_error:
            # Both APIs failed — don't penalize, mark as unchecked
            results.append({
                **ref,
                "status": "api_error",
                "reason": "both Semantic Scholar and CrossRef APIs failed",
            })
        else:
            results.append({
                **ref,
                "status": "unverified",
                "reason": "not found in Semantic Scholar or CrossRef",
            })

        time.sleep(REQUEST_DELAY)

    # Only count refs that were actually checked and found to be fake.
    # Exclude parse_error (parser bug) and api_error (API outage) —
    # these should not trigger false rejections.
    unverified = sum(1 for r in results if r["status"] == "unverified")

    result = ReferenceCheckResult(
        total=len(raw_refs),
        verified=verified,
        unverified=unverified,
        references=results,
    )

    _display_results(result)
    return result


# ── LaTeX reference extraction ───────────────────────────────────────────


def _extract_references(latex: str) -> list[dict]:
    """Extract individual references from LaTeX bibliography.

    Handles both \\bibitem and thebibliography environments, as well as
    raw bibliography entries.
    """
    refs = []

    # Method 1: \bibitem entries
    bibitem_pattern = r"\\bibitem\{([^}]*)\}\s*(.*?)(?=\\bibitem|\s*\\end\{thebibliography\}|$)"
    for match in re.finditer(bibitem_pattern, latex, re.DOTALL):
        key = match.group(1)
        body = _clean_latex(match.group(2).strip())
        parsed = _parse_bib_body(body)
        parsed["key"] = key
        parsed["raw"] = body[:300]
        refs.append(parsed)

    # Method 2: If no \bibitem found, try to parse a .bib-style bibliography
    if not refs:
        bib_pattern = r"@\w+\{([^,]+),\s*(.*?)\n\}"
        for match in re.finditer(bib_pattern, latex, re.DOTALL):
            key = match.group(1).strip()
            body = match.group(2)
            parsed = _parse_bibtex_entry(body)
            parsed["key"] = key
            parsed["raw"] = body[:300]
            refs.append(parsed)

    # Method 3: If still nothing, try to find references section and parse free-form
    if not refs:
        ref_section = re.search(
            r"\\section\*?\{References\}(.*?)(?=\\section|\\end\{document\}|$)",
            latex, re.DOTALL | re.IGNORECASE,
        )
        if ref_section:
            body = ref_section.group(1)
            # Split by common patterns: [1], [2] or numbered lines
            items = re.split(r"\n\s*\[?\d+\]?\s*", body)
            for item in items:
                item = _clean_latex(item.strip())
                if len(item) > 20:  # skip noise
                    parsed = _parse_bib_body(item)
                    parsed["raw"] = item[:300]
                    refs.append(parsed)

    return refs


def _parse_bib_body(body: str) -> dict:
    """Parse a free-form bibliography entry into title/authors/year."""
    result = {"title": "", "authors": "", "year": ""}

    # Extract year
    year_match = re.search(r"\b(19|20)\d{2}\b", body)
    if year_match:
        result["year"] = year_match.group(0)

    # Try to extract title — often in quotes or after author list
    # Pattern: authors. "Title" or authors. Title. venue
    title_match = re.search(r'[""](.*?)["""]', body)
    if title_match:
        result["title"] = title_match.group(1).strip()
    else:
        # Try: text after first period, before second period/venue
        parts = re.split(r"\.\s+", body, maxsplit=3)
        if len(parts) >= 2:
            # First part is usually authors, second is title
            result["authors"] = parts[0].strip()
            result["title"] = parts[1].strip().rstrip(".")
        elif parts:
            result["title"] = parts[0].strip()

    # Extract authors (text before the year or first period)
    if not result["authors"]:
        author_match = re.match(r"^(.*?)(?:\.|,\s*\(?\d{4})", body)
        if author_match:
            result["authors"] = author_match.group(1).strip()

    return result


def _parse_bibtex_entry(body: str) -> dict:
    """Parse a BibTeX entry body into title/authors/year."""
    result = {"title": "", "authors": "", "year": ""}

    title_match = re.search(r"title\s*=\s*\{(.*?)\}", body, re.IGNORECASE | re.DOTALL)
    if title_match:
        result["title"] = _clean_latex(title_match.group(1).strip())

    author_match = re.search(r"author\s*=\s*\{(.*?)\}", body, re.IGNORECASE | re.DOTALL)
    if author_match:
        result["authors"] = _clean_latex(author_match.group(1).strip())

    year_match = re.search(r"year\s*=\s*\{?(\d{4})\}?", body, re.IGNORECASE)
    if year_match:
        result["year"] = year_match.group(1)

    return result


def _clean_latex(text: str) -> str:
    """Remove common LaTeX commands from text."""
    text = re.sub(r"\\[a-zA-Z]+\{([^}]*)\}", r"\1", text)  # \cmd{arg} -> arg
    text = re.sub(r"[{}~\\]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ── API lookups ──────────────────────────────────────────────────────────


def _search_semantic_scholar(title: str, authors: str, year: str) -> dict | None:
    """Search Semantic Scholar for a paper by title.

    Returns match dict or None. Raises on API/network errors.
    """
    query = urllib.parse.quote(title[:200])
    url = f"{SEMANTIC_SCHOLAR_API}?query={query}&limit=3&fields=title,authors,year,externalIds,url"

    req = urllib.request.Request(url)
    req.add_header("User-Agent", "ResearchArena/1.0")

    with urllib.request.urlopen(req, timeout=10) as resp:
        data = json.loads(resp.read().decode())

    for paper in data.get("data", []):
        if _titles_match(title, paper.get("title", "")):
            authors_str = ", ".join(
                a.get("name", "") for a in paper.get("authors", [])[:5]
            )
            return {
                "source": "semantic_scholar",
                "title": paper.get("title", ""),
                "authors": authors_str,
                "year": str(paper.get("year", "")),
                "url": paper.get("url", ""),
            }

    return None


def _search_crossref(title: str, authors: str, year: str) -> dict | None:
    """Search CrossRef for a paper by title.

    Returns match dict or None. Raises on API/network errors.
    """
    query = urllib.parse.quote(title[:200])
    url = f"{CROSSREF_API}?query.title={query}&rows=3"

    req = urllib.request.Request(url)
    req.add_header("User-Agent", "ResearchArena/1.0 (mailto:researcharena@example.com)")

    with urllib.request.urlopen(req, timeout=10) as resp:
        data = json.loads(resp.read().decode())

    for item in data.get("message", {}).get("items", []):
        item_titles = item.get("title", [])
        if item_titles and _titles_match(title, item_titles[0]):
            authors_str = ", ".join(
                f"{a.get('given', '')} {a.get('family', '')}".strip()
                for a in item.get("author", [])[:5]
            )
            year_str = ""
            date_parts = item.get("published-print", {}).get("date-parts", [[]])
            if date_parts and date_parts[0]:
                year_str = str(date_parts[0][0])

            doi = item.get("DOI", "")
            return {
                "source": "crossref",
                "title": item_titles[0],
                "authors": authors_str,
                "year": year_str,
                "url": f"https://doi.org/{doi}" if doi else "",
            }

    return None


def _titles_match(query: str, candidate: str) -> bool:
    """Fuzzy title matching — checks if titles are similar enough."""
    def normalize(t: str) -> str:
        t = t.lower()
        t = re.sub(r"[^a-z0-9\s]", "", t)
        t = re.sub(r"\s+", " ", t).strip()
        return t

    q = normalize(query)
    c = normalize(candidate)

    if not q or not c:
        return False

    # Exact match after normalization
    if q == c:
        return True

    # Check if one contains most of the other (handle truncation)
    q_words = set(q.split())
    c_words = set(c.split())

    if not q_words or not c_words:
        return False

    overlap = len(q_words & c_words)
    max_len = max(len(q_words), len(c_words))

    return overlap / max_len >= 0.7


# ── Display ──────────────────────────────────────────────────────────────


def _display_results(result: ReferenceCheckResult):
    """Print reference check results as a table."""
    table = Table(title=f"Reference Check: {result.verified}/{result.total} verified")
    table.add_column("#", justify="center", width=3)
    table.add_column("Status", justify="center")
    table.add_column("Title")
    table.add_column("Source")

    for i, ref in enumerate(result.references):
        status = ref["status"]
        style = "green" if status == "verified" else "red"
        title = ref.get("title", ref.get("raw", "?"))[:60]
        source = ref.get("source", ref.get("reason", ""))

        table.add_row(str(i + 1), f"[{style}]{status}[/]", title, source)

    console.print(table)

    if result.fake_rate > 0:
        console.print(
            f"  [{'red' if result.fake_rate > 0.3 else 'yellow'}]"
            f"Unverified rate: {result.fake_rate:.0%} "
            f"({result.unverified}/{result.total})[/]"
        )


def save_reference_check(result: ReferenceCheckResult, output_dir: str | Path) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "reference_check.json"
    data = {
        "total": result.total,
        "verified": result.verified,
        "unverified": result.unverified,
        "fake_rate": result.fake_rate,
        "references": result.references,
    }
    path.write_text(json.dumps(data, indent=2))
    return path
