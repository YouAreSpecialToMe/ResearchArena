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
ARXIV_API = "http://export.arxiv.org/api/query"
REQUEST_DELAY = 1.0  # seconds between API calls
REQUEST_TIMEOUT = 30  # seconds per API request
MAX_RETRIES = 2       # retry count per API


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


def check_references(paper_latex: str, workspace: Path | None = None) -> ReferenceCheckResult:
    """Extract and verify all references from a LaTeX paper.

    Args:
        paper_latex: LaTeX source code
        workspace: workspace directory (to find .bib files if references
                   are in a separate file via \\bibliography{})

    Returns:
        ReferenceCheckResult with verification status per reference
    """
    # If the .tex uses \bibliography{name}, read the .bib file too
    combined = paper_latex
    if workspace:
        bib_match = re.search(r'\\bibliography\{(\w+)\}', paper_latex)
        if bib_match:
            bib_path = workspace / f"{bib_match.group(1)}.bib"
            if bib_path.exists():
                combined = paper_latex + "\n" + bib_path.read_text()

    raw_refs = _extract_references(combined)
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

        # Try Semantic Scholar, then CrossRef, then arXiv (with retries)
        match = None
        api_error_count = 0
        for search_fn in [_search_semantic_scholar, _search_crossref, _search_arxiv]:
            if match:
                break
            try:
                match = _search_with_retry(search_fn, title, authors, year)
            except Exception:
                api_error_count += 1
        api_error = (api_error_count == 3)  # all three APIs failed

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

    # Method 1: \bibitem entries (with optional [...] label)
    bibitem_pattern = r"\\bibitem(?:\[[^\]]*\])?\{([^}]*)\}\s*(.*?)(?=\\bibitem|\s*\\end\{thebibliography\}|$)"
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
        # Split on sentence boundaries: period+space, but skip single-letter
        # abbreviations (P. Srinivasan, T. Barron, etc.)
        # Try multiple candidate titles and pick the best one
        parts = re.split(r"(?<![A-Z])\.(?:\s+)(?=[A-Z])", body, maxsplit=5)
        if len(parts) >= 2:
            result["authors"] = parts[0].strip()
            # The title is usually the longest non-venue part after authors
            # Skip parts that look like venue names
            for part in parts[1:]:
                part = part.strip().rstrip(".")
                if len(part) > 10 and not re.match(r"(?:In |Proceedings|arXiv|https?://|ACM |IEEE )", part):
                    result["title"] = part
                    break
            if not result["title"] and len(parts) >= 2:
                result["title"] = parts[1].strip().rstrip(".")
        elif parts:
            result["title"] = parts[0].strip()

    # Extract authors (text before the year or first sentence boundary)
    if not result["authors"]:
        author_match = re.match(r"^(.*?)(?<![A-Z])\.(?:\s+)(?=[A-Z])", body)
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
    # Remove \newblock and other no-arg commands
    text = re.sub(r"\\(?:newblock|noindent|par|medskip|smallskip|bigskip)\b\s*", "", text)
    # \cmd{arg} -> arg (handles nested braces simply)
    text = re.sub(r"\\[a-zA-Z]+\{([^}]*)\}", r"\1", text)
    # {X} -> X (single-char brace groups like {G}aussian -> Gaussian)
    text = re.sub(r"\{([^}]*)\}", r"\1", text)
    # Remove remaining tildes and backslashes
    text = re.sub(r"[~\\]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ── API lookups ──────────────────────────────────────────────────────────


def _search_with_retry(search_fn, title: str, authors: str, year: str):
    """Call a search function with retries and exponential backoff."""
    last_error = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            return search_fn(title, authors, year)
        except Exception as e:
            last_error = e
            if attempt < MAX_RETRIES:
                time.sleep(2 ** attempt)  # 1s, 2s backoff
    raise last_error


def _search_semantic_scholar(title: str, authors: str, year: str) -> dict | None:
    """Search Semantic Scholar for a paper by title.

    Returns match dict or None. Raises on API/network errors.
    """
    query = urllib.parse.quote(title[:200])
    url = f"{SEMANTIC_SCHOLAR_API}?query={query}&limit=3&fields=title,authors,year,externalIds,url"

    req = urllib.request.Request(url)
    req.add_header("User-Agent", "ResearchArena/1.0")

    with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
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

    with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
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


def _search_arxiv(title: str, authors: str, year: str) -> dict | None:
    """Search arXiv for a paper by title.

    arXiv indexes preprints quickly, so this catches very recent papers
    that Semantic Scholar and CrossRef haven't indexed yet.

    Returns match dict or None. Raises on API/network errors.
    """
    # arXiv API uses Atom feed format
    query = urllib.parse.quote(f'ti:"{title[:150]}"')
    url = f"http://export.arxiv.org/api/query?search_query={query}&max_results=3"

    req = urllib.request.Request(url)
    req.add_header("User-Agent", "ResearchArena/1.0")

    with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
        data = resp.read().decode()

    # Parse Atom XML for title matches
    # arXiv returns XML, extract titles with regex (avoid xml dependency)
    titles_found = re.findall(r"<title>(.*?)</title>", data, re.DOTALL)
    arxiv_ids = re.findall(r"<id>http://arxiv.org/abs/([^<]+)</id>", data)
    author_names = re.findall(r"<name>(.*?)</name>", data)

    # Skip first title (it's the feed title, not a paper)
    for i, arxiv_title in enumerate(titles_found[1:]):
        clean_title = re.sub(r"\s+", " ", arxiv_title.strip())
        if _titles_match(title, clean_title):
            arxiv_id = arxiv_ids[i] if i < len(arxiv_ids) else ""
            return {
                "source": "arxiv",
                "title": clean_title,
                "authors": "",
                "year": year,
                "url": f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else "",
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
