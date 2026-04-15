import os
import re
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import fitz
import requests
import xml.etree.ElementTree as ET


ARXIV_NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
}


@dataclass
class RefSpec:
    arxiv_id: str
    venue: str
    year: int
    url: str
    pdf_url: str


REFS = [
    RefSpec(
        arxiv_id="2004.11362",
        venue="NeurIPS",
        year=2020,
        url="https://arxiv.org/abs/2004.11362",
        pdf_url="https://arxiv.org/pdf/2004.11362.pdf",
    ),
    RefSpec(
        arxiv_id="2204.07596",
        venue="ICML",
        year=2022,
        url="https://arxiv.org/abs/2204.07596",
        pdf_url="https://arxiv.org/pdf/2204.07596.pdf",
    ),
    RefSpec(
        arxiv_id="2207.07180",
        venue="ICML",
        year=2022,
        url="https://arxiv.org/abs/2207.07180",
        pdf_url="https://arxiv.org/pdf/2207.07180.pdf",
    ),
    RefSpec(
        arxiv_id="2306.15925",
        venue="ICCV",
        year=2023,
        url="https://arxiv.org/abs/2306.15925",
        pdf_url="https://arxiv.org/pdf/2306.15925.pdf",
    ),
    RefSpec(
        arxiv_id="2110.02473",
        venue="JMLR",
        year=2023,
        url="https://arxiv.org/abs/2110.02473",
        pdf_url="https://arxiv.org/pdf/2110.02473.pdf",
    ),
    RefSpec(
        arxiv_id="2103.00020",
        venue="ICML",
        year=2021,
        url="https://arxiv.org/abs/2103.00020",
        pdf_url="https://arxiv.org/pdf/2103.00020.pdf",
    ),
    RefSpec(
        arxiv_id="2405.03649",
        venue="IJCAI",
        year=2024,
        url="https://arxiv.org/abs/2405.03649",
        pdf_url="https://arxiv.org/pdf/2405.03649.pdf",
    ),
    RefSpec(
        arxiv_id="2410.12474",
        venue="NeurIPS",
        year=2024,
        url="https://arxiv.org/abs/2410.12474",
        pdf_url="https://arxiv.org/pdf/2410.12474.pdf",
    ),
    RefSpec(
        arxiv_id="1901.10514",
        venue="NeurIPS",
        year=2019,
        url="https://arxiv.org/abs/1901.10514",
        pdf_url="https://arxiv.org/pdf/1901.10514.pdf",
    ),
    RefSpec(
        arxiv_id="2208.10043",
        venue="ECCV",
        year=2022,
        url="https://arxiv.org/abs/2208.10043",
        pdf_url="https://arxiv.org/pdf/2208.10043.pdf",
    ),
]


def slugify(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9]+", "-", text).strip("-")
    return text[:100] or "paper"


def fetch_arxiv_metadata(arxiv_id: str) -> Dict[str, str]:
    api = f"http://export.arxiv.org/api/query?id_list={arxiv_id}"
    root = ET.fromstring(requests.get(api, timeout=30).text)
    entry = root.find("atom:entry", ARXIV_NS)
    if entry is None:
        raise RuntimeError(f"Missing arXiv metadata for {arxiv_id}")

    title = " ".join(entry.findtext("atom:title", default="", namespaces=ARXIV_NS).split())
    summary = " ".join(entry.findtext("atom:summary", default="", namespaces=ARXIV_NS).split())
    authors = [
        " ".join(author.findtext("atom:name", default="", namespaces=ARXIV_NS).split())
        for author in entry.findall("atom:author", ARXIV_NS)
    ]
    updated = entry.findtext("atom:updated", default="", namespaces=ARXIV_NS)
    published = entry.findtext("atom:published", default="", namespaces=ARXIV_NS)
    return {
        "title": title,
        "summary": summary,
        "authors": authors,
        "updated": updated,
        "published": published,
    }


def download_pdf(pdf_url: str, target: Path) -> None:
    response = requests.get(pdf_url, timeout=60)
    response.raise_for_status()
    target.write_bytes(response.content)


def extract_text(pdf_path: Path) -> str:
    doc = fitz.open(pdf_path)
    pages = []
    for page in doc:
        pages.append(page.get_text("text"))
    return "\n".join(pages)


def normalize_text(text: str) -> str:
    text = text.replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def extract_abstract(raw_text: str, fallback: str) -> str:
    match = re.search(
        r"\bAbstract\b[:\s]*\n?(.*?)(?=\n\s*(?:1[\.\s]+Introduction|1\s+Introduction|Introduction)\b)",
        raw_text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if match:
        return normalize_text(match.group(1)).strip()
    return fallback.strip()


def section_starts(raw_text: str) -> List[re.Match]:
    pattern = re.compile(
        r"(?m)^\s*((?:\d+|[A-Z])(?:\.\d+)*)[\.\s]+([A-Z][^\n]{2,120})\s*$"
    )
    matches = []
    for match in pattern.finditer(raw_text):
        heading = match.group(2).strip()
        if heading.lower() in {"abstract", "references"}:
            continue
        matches.append(match)
    return matches


def split_sections(raw_text: str) -> Dict[str, str]:
    sections: Dict[str, str] = {}
    matches = section_starts(raw_text)
    for idx, match in enumerate(matches):
        start = match.start()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(raw_text)
        heading_num = match.group(1).strip()
        heading_name = " ".join(match.group(2).split())
        body = raw_text[start:end].strip()
        sections[f"{heading_num} {heading_name}"] = normalize_text(body)
    return sections


def make_bibtex(meta: Dict[str, str], spec: RefSpec) -> str:
    first_author = meta["authors"][0].split()[-1].lower()
    key = f"{first_author}{spec.year}{slugify(meta['title']).split('-')[0]}"
    authors = " and ".join(meta["authors"])
    fields = [
        f"  title = {{{meta['title']}}}",
        f"  author = {{{authors}}}",
        f"  year = {{{spec.year}}}",
        f"  journal = {{{spec.venue}}}" if spec.venue == "JMLR" else f"  booktitle = {{{spec.venue}}}",
        f"  url = {{{spec.url}}}",
        f"  eprint = {{{spec.arxiv_id}}}",
        "  archivePrefix = {arXiv}",
    ]
    return "@article{" + key + ",\n" + ",\n".join(fields) + "\n}\n"


def write_reference(spec: RefSpec, out_root: Path) -> None:
    meta = fetch_arxiv_metadata(spec.arxiv_id)
    paper_dir = out_root / slugify(meta["title"])
    meta_dir = paper_dir / "meta"
    sec_dir = paper_dir / "sections"
    meta_dir.mkdir(parents=True, exist_ok=True)
    sec_dir.mkdir(parents=True, exist_ok=True)

    pdf_path = meta_dir / "paper.pdf"
    if not pdf_path.exists():
        download_pdf(spec.pdf_url, pdf_path)

    raw_text = extract_text(pdf_path)
    raw_text = normalize_text(raw_text)
    abstract = extract_abstract(raw_text, meta["summary"])
    sections = split_sections(raw_text)

    meta_info = textwrap.dedent(
        f"""\
        Title: {meta['title']}
        Authors: {", ".join(meta['authors'])}
        Venue: {spec.venue}
        Year: {spec.year}
        URL: {spec.url}
        arXiv: {spec.arxiv_id}
        Published: {meta['published']}
        Updated: {meta['updated']}
        """
    ).strip() + "\n"

    (meta_dir / "meta_info.txt").write_text(meta_info)
    (meta_dir / "bibtex.txt").write_text(make_bibtex(meta, spec))
    (sec_dir / "abstract.md").write_text(abstract + "\n")

    if not sections:
        sections = {"1 Full Text": raw_text}

    for section_name, content in sections.items():
        filename = slugify(section_name) + ".md"
        (sec_dir / filename).write_text(content + "\n")


def main() -> None:
    out_root = Path("references")
    out_root.mkdir(exist_ok=True)
    for spec in REFS:
        write_reference(spec, out_root)


if __name__ == "__main__":
    main()
