from __future__ import annotations

import re
import tarfile
import tempfile
from pathlib import Path

import requests
from pypdf import PdfReader


ROOT = Path(__file__).resolve().parent
REF_ROOT = ROOT / "references"

PAPERS = [
    {
        "slug": "Free-Record-Level-Privacy-Risk-Evaluation-Through-Artifact-Based-Methods",
        "title": "Free Record-Level Privacy Risk Evaluation Through Artifact-Based Methods",
        "authors": [
            "Joseph Pollock",
            "Igor Shilov",
            "Euodia Dodd",
            "Yves-Alexandre de Montjoye",
        ],
        "year": "2024",
        "venue": "arXiv preprint",
        "arxiv_id": "2411.05743v3",
    },
    {
        "slug": "Evaluating-the-Dynamics-of-Membership-Privacy-in-Deep-Learning",
        "title": "Evaluating the Dynamics of Membership Privacy in Deep Learning",
        "authors": [
            "Yuetian Chen",
            "Zhiqi Wang",
            "Nathalie Baracaldo",
            "Swanand Ravindra Kadhe",
            "Lei Yu",
        ],
        "year": "2025",
        "venue": "arXiv preprint",
        "arxiv_id": "2507.23291v2",
    },
    {
        "slug": "Membership-Inference-Attacks-as-Privacy-Tools-Reliability-Disparity-and-Ensemble",
        "title": "Membership Inference Attacks as Privacy Tools: Reliability, Disparity and Ensemble",
        "authors": [
            "Zhiqi Wang",
            "Chengyu Zhang",
            "Yuetian Chen",
            "Nathalie Baracaldo",
            "Swanand Kadhe",
            "Lei Yu",
        ],
        "year": "2025",
        "venue": "arXiv preprint",
        "arxiv_id": "2506.13972v2",
    },
    {
        "slug": "Membership-Inference-Attacks-From-First-Principles",
        "title": "Membership Inference Attacks From First Principles",
        "authors": [
            "Nicholas Carlini",
            "Steve Chien",
            "Milad Nasr",
            "Shuang Song",
            "Andreas Terzis",
            "Florian Tramèr",
        ],
        "year": "2021",
        "venue": "IEEE Symposium on Security and Privacy / arXiv preprint",
        "arxiv_id": "2112.03570v2",
    },
    {
        "slug": "Machine-Learning-with-Membership-Privacy-using-Adversarial-Regularization",
        "title": "Machine Learning with Membership Privacy using Adversarial Regularization",
        "authors": [
            "Milad Nasr",
            "Reza Shokri",
            "Amir Houmansadr",
        ],
        "year": "2018",
        "venue": "ACM CCS / arXiv preprint",
        "arxiv_id": "1807.05852v1",
    },
    {
        "slug": "MIST-Defending-Against-Membership-Inference-Attacks-Through-Membership-Invariant-Subspace-Training",
        "title": "MIST: Defending Against Membership Inference Attacks Through Membership-Invariant Subspace Training",
        "authors": [
            "Jiacheng Li",
            "Ninghui Li",
            "Bruno Ribeiro",
        ],
        "year": "2024",
        "venue": "arXiv preprint",
        "arxiv_id": "2311.00919v2",
    },
    {
        "slug": "Defending-Against-Membership-Inference-Attacks-on-Iteratively-Pruned-Deep-Neural-Networks",
        "title": "Defending Against Membership Inference Attacks on Iteratively Pruned Deep Neural Networks",
        "authors": [
            "Jing Shang",
            "Jian Wang",
            "Kailun Wang",
            "Jiqiang Liu",
            "Nan Jiang",
            "Md. Armanuzzaman",
            "Ziming Zhao",
        ],
        "year": "2025",
        "venue": "NDSS Symposium",
        "pdf_url": "https://www.ndss-symposium.org/wp-content/uploads/2025-90-paper.pdf",
        "abs_url": "https://www.ndss-symposium.org/ndss-paper/defending-against-membership-inference-attacks-on-iteratively-pruned-deep-neural-networks/",
        "source_url": "https://www.ndss-symposium.org/ndss-paper/defending-against-membership-inference-attacks-on-iteratively-pruned-deep-neural-networks/",
        "pdf_filename": "ndss2025-90-paper.pdf",
    },
    {
        "slug": "RelaxLoss-Defending-Membership-Inference-Attacks-without-Losing-Utility",
        "title": "RelaxLoss: Defending Membership Inference Attacks without Losing Utility",
        "authors": [
            "Dingfan Chen",
            "Ning Yu",
            "Mario Fritz",
        ],
        "year": "2022",
        "venue": "ICLR / arXiv preprint",
        "arxiv_id": "2207.05801v1",
    },
    {
        "slug": "AdaMixup-A-Dynamic-Defense-Framework-for-Membership-Inference-Attack-Mitigation",
        "title": "AdaMixup: A Dynamic Defense Framework for Membership Inference Attack Mitigation",
        "authors": [
            "Ying Chen",
            "Jiajing Chen",
            "Yijie Weng",
            "ChiaHua Chang",
            "Dezhi Yu",
            "Guanbiao Lin",
        ],
        "year": "2025",
        "venue": "arXiv preprint",
        "arxiv_id": "2501.02182v1",
    },
    {
        "slug": "Mitigating-Privacy-Risk-in-Membership-Inference-by-Convex-Concave-Loss",
        "title": "Mitigating Privacy Risk in Membership Inference by Convex-Concave Loss",
        "authors": [
            "Zhenlong Liu",
            "Lei Feng",
            "Huiping Zhuang",
            "Xiaofeng Cao",
            "Hongxin Wei",
        ],
        "year": "2024",
        "venue": "ICML / arXiv preprint",
        "arxiv_id": "2402.05453v3",
    },
    {
        "slug": "Overconfidence-is-a-Dangerous-Thing-Mitigating-Membership-Inference-Attacks-by-Enforcing-Less-Confident-Prediction",
        "title": "Overconfidence is a Dangerous Thing: Mitigating Membership Inference Attacks by Enforcing Less Confident Prediction",
        "authors": [
            "Zitao Chen",
            "Karthik Pattabiraman",
        ],
        "year": "2024",
        "venue": "NDSS Symposium / arXiv preprint",
        "arxiv_id": "2307.01610v1",
    },
    {
        "slug": "Mitigating-Membership-Inference-Attacks-by-Self-Distillation-Through-a-Novel-Ensemble-Architecture",
        "title": "Mitigating Membership Inference Attacks by Self-Distillation Through a Novel Ensemble Architecture",
        "authors": [
            "Xinyu Tang",
            "Saeed Mahloujifar",
            "Liwei Song",
            "Virat Shejwalkar",
            "Milad Nasr",
            "Amir Houmansadr",
            "Prateek Mittal",
        ],
        "year": "2021",
        "venue": "arXiv preprint",
        "arxiv_id": "2110.08324v1",
    },
    {
        "slug": "Mitigating-Disparate-Impact-of-Differentially-Private-Learning-through-Bounded-Adaptive-Clipping",
        "title": "Mitigating Disparate Impact of Differentially Private Learning through Bounded Adaptive Clipping",
        "authors": [
            "Linzh Zhao",
            "Aki Rehn",
            "Mikko A. Heikkilä",
            "Razane Tajeddine",
            "Antti Honkela",
        ],
        "year": "2025",
        "venue": "arXiv preprint",
        "arxiv_id": "2506.01396v1",
    },
]


def sanitize_heading(text: str) -> str:
    text = text.strip()
    text = re.sub(r"[\\/:*?\"<>|]+", "-", text)
    return text[:120] or "section"


def clean_text(text: str) -> str:
    text = text.replace("\x00", "")
    text = text.encode("utf-8", errors="ignore").decode("utf-8")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def normalize_heading_line(line: str) -> str:
    line = line.strip()
    while True:
        merged = re.sub(r"\b([A-Z])\s+(?=[A-Z]\b)", r"\1", line)
        if merged == line:
            break
        line = merged
    return line


def extract_text(pdf_path: Path) -> str:
    reader = PdfReader(str(pdf_path))
    parts = []
    for page in reader.pages:
        parts.append(page.extract_text() or "")
    return clean_text("\n".join(parts))


def strip_tex_comments(text: str) -> str:
    return re.sub(r"(?<!\\)%.*", "", text)


def load_main_tex(source_dir: Path) -> str | None:
    tex_files = list(source_dir.rglob("*.tex"))
    if not tex_files:
        return None

    scored = []
    for path in tex_files:
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if "\\begin{document}" in text:
            score = text.count("\\section") + text.count("\\subsection")
            scored.append((score, len(text), path))

    if not scored:
        return None

    _, _, main_path = max(scored)
    visited: set[Path] = set()

    def resolve(path: Path) -> str:
        path = path.resolve()
        if path in visited or not path.exists():
            return ""
        visited.add(path)
        text = strip_tex_comments(path.read_text(encoding="utf-8", errors="ignore"))

        def repl(match: re.Match[str]) -> str:
            name = match.group(2).strip()
            if not name:
                return ""
            candidate = (path.parent / name)
            if candidate.suffix != ".tex":
                candidate = candidate.with_suffix(".tex")
            return "\n" + resolve(candidate) + "\n"

        return re.sub(r"\\(input|include)\{([^}]+)\}", repl, text)

    return resolve(main_path)


def tex_to_text(value: str) -> str:
    value = re.sub(r"\\[a-zA-Z]+\*?(?:\[[^\]]*\])?\{([^{}]*)\}", r"\1", value)
    value = re.sub(r"\\[a-zA-Z]+\*?", "", value)
    value = value.replace("{", "").replace("}", "")
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def split_sections_from_tex(tex: str) -> list[tuple[str, str]]:
    tex = strip_tex_comments(tex)
    sections: list[tuple[str, str]] = []

    abstract_match = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", tex, flags=re.S)
    if abstract_match:
        abstract = tex_to_text(abstract_match.group(1))
        if abstract:
            sections.append(("abstract", abstract))

    pattern = re.compile(r"\\(section|subsection|subsubsection)\*?\{([^}]*)\}")
    matches = list(pattern.finditer(tex))
    for idx, match in enumerate(matches):
        title = tex_to_text(match.group(2))
        if not title:
            continue
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(tex)
        body = tex_to_text(tex[start:end])
        if body:
            level = match.group(1)
            prefix = {"section": "", "subsection": "subsection ", "subsubsection": "subsubsection "}[level]
            sections.append((f"{prefix}{title}".strip(), body))

    return sections


def split_sections(text: str) -> list[tuple[str, str]]:
    raw_lines = text.splitlines()
    lines = [normalize_heading_line(line) for line in raw_lines]
    heading_pattern = re.compile(
        r"^\s*((?:\d+(?:\.\d+)*)|(?:[IVXLC]+\.)?)\s*([A-Z][A-Za-z0-9 ,:()\-/]+)$"
    )
    common_headings = {
        "abstract",
        "introduction",
        "background",
        "related work",
        "method",
        "methods",
        "approach",
        "experiments",
        "evaluation",
        "discussion",
        "limitations",
        "conclusion",
        "conclusions",
        "appendix",
    }

    def is_heading(line: str) -> bool:
        candidate = line.strip()
        if not candidate:
            return False
        lower = candidate.lower()
        if lower in common_headings:
            return True
        match = heading_pattern.match(candidate)
        if not match:
            return False
        words = candidate.split()
        return 1 <= len(words) <= 12

    abstract_idx = next(
        (
            i
            for i, line in enumerate(lines)
            if line.strip().lower() == "abstract" or line.strip().lower().startswith("abstract-")
        ),
        None,
    )

    sections: list[tuple[str, str]] = []
    intro_idx = None
    for i, line in enumerate(lines):
        if abstract_idx is not None and i <= abstract_idx:
            continue
        if is_heading(line):
            intro_idx = i
            break

    if abstract_idx is not None:
        start = abstract_idx + 1
        end = intro_idx if intro_idx is not None else len(lines)
        abstract = clean_text("\n".join(lines[start:end]))
        if abstract:
            sections.append(("abstract", abstract))

    current_title = None
    current_lines: list[str] = []
    for line in lines[intro_idx or 0 :]:
        if is_heading(line):
            if current_title and current_lines:
                sections.append((current_title, clean_text("\n".join(current_lines))))
            match = heading_pattern.match(line.strip())
            if match and match.group(1):
                current_title = f"{match.group(1)} {match.group(2).strip()}".strip()
            else:
                current_title = line.strip()
            current_lines = []
        elif current_title:
            current_lines.append(line)

    if current_title and current_lines:
        sections.append((current_title, clean_text("\n".join(current_lines))))

    if not sections:
        sections.append(("full_text", text))

    return sections


def bibtex_key(paper: dict) -> str:
    surname = paper["authors"][0].split()[-1].lower()
    return f"{surname}{paper['year']}{paper['title'].split()[0].lower()}"


def write_paper(paper: dict) -> None:
    paper_dir = REF_ROOT / paper["slug"]
    meta_dir = paper_dir / "meta"
    sections_dir = paper_dir / "sections"
    meta_dir.mkdir(parents=True, exist_ok=True)
    sections_dir.mkdir(parents=True, exist_ok=True)

    if "arxiv_id" in paper:
        pdf_url = f"https://arxiv.org/pdf/{paper['arxiv_id']}.pdf"
        abs_url = f"https://arxiv.org/abs/{paper['arxiv_id']}"
        source_url = f"https://arxiv.org/e-print/{paper['arxiv_id'].split('v')[0]}"
        pdf_filename = f"{paper['arxiv_id']}.pdf"
    else:
        pdf_url = paper["pdf_url"]
        abs_url = paper["abs_url"]
        source_url = paper.get("source_url", abs_url)
        pdf_filename = paper.get("pdf_filename", "paper.pdf")

    pdf_path = paper_dir / pdf_filename
    if not pdf_path.exists():
        response = requests.get(pdf_url, timeout=60)
        response.raise_for_status()
        pdf_path.write_bytes(response.content)

    sections: list[tuple[str, str]] = []
    with tempfile.TemporaryDirectory() as tmpdir:
        archive_path = Path(tmpdir) / "source.tar"
        try:
            if "arxiv_id" in paper:
                response = requests.get(source_url, timeout=60)
                response.raise_for_status()
                archive_path.write_bytes(response.content)
                source_dir = Path(tmpdir) / "src"
                source_dir.mkdir()
                with tarfile.open(archive_path) as tar:
                    tar.extractall(source_dir, filter="data")
                tex = load_main_tex(source_dir)
                if tex:
                    sections = split_sections_from_tex(tex)
        except Exception:
            sections = []

    if not sections:
        text = extract_text(pdf_path)
        sections = split_sections(text)

    meta_info = "\n".join(
        [
            f"Title: {paper['title']}",
            f"Authors: {', '.join(paper['authors'])}",
            f"Venue: {paper['venue']}",
            f"Year: {paper['year']}",
            f"URL: {abs_url}",
            f"PDF: {pdf_url}",
            f"Source: {source_url}",
        ]
    )
    (meta_dir / "meta_info.txt").write_text(meta_info + "\n", encoding="utf-8")

    if "arxiv_id" in paper:
        bibtex_lines = [
            f"@article{{{bibtex_key(paper)},",
            f"  title={{ {paper['title']} }},",
            f"  author={{ {' and '.join(paper['authors'])} }},",
            f"  journal={{ {paper['venue']} }},",
            f"  year={{ {paper['year']} }},",
            f"  eprint={{ {paper['arxiv_id'].split('v')[0]} }},",
            f"  archivePrefix={{ arXiv }},",
            f"  url={{ {abs_url} }},",
            "}",
        ]
    else:
        bibtex_lines = [
            f"@inproceedings{{{bibtex_key(paper)},",
            f"  title={{ {paper['title']} }},",
            f"  author={{ {' and '.join(paper['authors'])} }},",
            f"  booktitle={{ {paper['venue']} }},",
            f"  year={{ {paper['year']} }},",
            f"  url={{ {abs_url} }},",
            "}",
        ]
    bibtex = "\n".join(bibtex_lines)
    (meta_dir / "bibtex.txt").write_text(bibtex + "\n", encoding="utf-8")

    for title, content in sections:
        filename = "abstract.md" if title == "abstract" else f"{sanitize_heading(title)}.md"
        section_text = f"# {title.replace('_', ' ').title()}\n\n{content}\n"
        (sections_dir / filename).write_text(section_text, encoding="utf-8")


def main() -> None:
    REF_ROOT.mkdir(exist_ok=True)
    for paper in PAPERS:
        write_paper(paper)


if __name__ == "__main__":
    main()
