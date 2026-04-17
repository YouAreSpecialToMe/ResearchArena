"""
Research Arena — Paper Viewer
Interactive Streamlit app to browse all 117 AI-generated papers with reviews.

Usage:
    streamlit run paper_viewer.py
"""

import json
import os
import streamlit as st
import pandas as pd

from datetime import datetime

ROOT = os.path.dirname(os.path.abspath(__file__))
COMMENTS_FILE = os.path.join(ROOT, "papers", "human_comments.json")


def load_comments():
    if os.path.exists(COMMENTS_FILE):
        with open(COMMENTS_FILE) as f:
            return json.load(f)
    return {}


def save_comments(comments):
    with open(COMMENTS_FILE, "w") as f:
        json.dump(comments, f, indent=2)


def render_comments(paper_key):
    """Render human comments section for a paper."""
    if "comments" not in st.session_state:
        st.session_state.comments = load_comments()

    comments = st.session_state.comments
    paper_comments = comments.get(paper_key, [])

    # Display existing comments
    if paper_comments:
        for i, c in enumerate(paper_comments):
            st.markdown(
                f'<div style="background:#161b22;border:1px solid #30363d;border-radius:6px;padding:0.8rem;margin-bottom:0.5rem">'
                f'<div style="font-size:0.75rem;color:#8b949e;margin-bottom:0.3rem">{c["timestamp"]}</div>'
                f'<div style="color:#c9d1d9">{c["text"]}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        if st.button("Clear all comments", key=f"clear_{paper_key}"):
            comments.pop(paper_key, None)
            save_comments(comments)
            st.session_state.comments = comments
            st.rerun()
    else:
        st.caption("No comments yet.")

    # Add new comment
    new_comment = st.text_area(
        "Add a comment",
        key=f"comment_input_{paper_key}",
        height=100,
        placeholder="Write your notes about this paper, its reviews, or experiments...",
    )
    if st.button("Save Comment", key=f"save_{paper_key}"):
        if new_comment.strip():
            if paper_key not in comments:
                comments[paper_key] = []
            comments[paper_key].append({
                "text": new_comment.strip(),
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
            })
            save_comments(comments)
            st.session_state.comments = comments
            st.rerun()

# ── Seed display names ──────────────────────────────────────────────
DOMAIN_NAMES = {
    "causal_learning": "Causal Learning",
    "compiler_optimization": "Compiler Optimization",
    "data_integration_cleaning": "Data Integration & Cleaning",
    "datasets_benchmarks": "Datasets & Benchmarks",
    "operating_system_design": "Operating System Design",
    "ai_for_biology": "AI for Biology",
    "computer_vision": "Computer Vision",
    "generative_models": "Generative Models",
    "interpretability_of_learned_repr": "Interpretability of Learned Repr.",
    "natural_language_processing": "Natural Language Processing",
    "privacy_in_ml": "Privacy in ML",
    "probabilistic_methods": "Probabilistic Methods",
    "supervised_repr_learning": "Supervised Repr. Learning",
}

AGENT_COLORS = {"claude": "#9C27B0", "codex": "#4CAF50", "kimi": "#FF9800"}
AGENT_LABELS = {"claude": "Claude Code", "codex": "Codex", "kimi": "Kimi"}
DIM_LABELS = {
    "novelty": "Novelty",
    "soundness": "Soundness",
    "significance": "Significance",
    "clarity": "Clarity",
    "reproducibility": "Reproducibility",
    "experimental_rigor": "Exp. Rigor",
    "references": "References",
    "reference_integrity": "Ref. Integrity",
    "results_integrity": "Results Integrity",
}


@st.cache_data
def load_data():
    manifest = json.load(open(os.path.join(ROOT, "papers/manifest.json")))

    papers = []
    for i, entry in enumerate(manifest):
        path = entry["path"]
        domain = DOMAIN_NAMES.get(entry["seed"], entry["seed"])

        # PR reviews
        pr_path = os.path.join(ROOT, path, "reviews.json")
        pr_data = None
        pr_avg = None
        pr_decision = None
        if os.path.exists(pr_path):
            try:
                pr_data = json.load(open(pr_path))
                pr_avg = pr_data.get("avg_score")
                pr_decision = pr_data.get("decision")
            except Exception:
                pass

        # SAR review
        sar_path = os.path.join(ROOT, path, "stanford_review.json")
        sar_data = None
        sar_score = None
        if os.path.exists(sar_path):
            try:
                sar_data = json.load(open(sar_path))
                sar_score = sar_data.get("overall_score")
            except Exception:
                pass

        papers.append({
            "index": i,
            "title": entry["title"],
            "agent": entry["agent"],
            "seed": entry["seed"],
            "domain": domain,
            "platform": entry["platform"],
            "trial": entry["trial"],
            "has_paper": entry.get("has_paper", False),
            "path": path,
            "pr_avg": round(pr_avg, 2) if pr_avg else None,
            "pr_decision": pr_decision,
            "pr_data": pr_data,
            "sar_score": sar_score,
            "sar_data": sar_data,
        })

    return papers


def score_color(score):
    if score is None:
        return "gray"
    if score >= 6:
        return "#3fb950"
    if score >= 4:
        return "#d29922"
    return "#f85149"


def agent_badge(agent):
    color = AGENT_COLORS.get(agent, "#888")
    label = AGENT_LABELS.get(agent, agent)
    return f'<span style="background:{color};color:#fff;padding:2px 8px;border-radius:4px;font-size:0.8rem;font-weight:600">{label}</span>'


def decision_badge(d):
    if not d:
        return ""
    colors = {"accept": "#3fb950", "revision": "#d29922", "reject": "#f85149"}
    text_color = "#000" if d == "revision" else "#fff"
    return f'<span style="background:{colors.get(d,"#888")};color:{text_color};padding:2px 8px;border-radius:4px;font-size:0.75rem;font-weight:600;text-transform:uppercase">{d}</span>'


def render_dimension_scores(scores):
    if not scores:
        st.caption("No dimension scores available")
        return
    dims = list(DIM_LABELS.keys())
    vals = [scores.get(d) for d in dims]
    labels = [DIM_LABELS[d] for d in dims]

    df = pd.DataFrame({"Dimension": labels, "Score": vals})
    df = df.dropna()
    if df.empty:
        st.caption("No dimension scores available")
        return

    st.bar_chart(df.set_index("Dimension"), height=250, color="#58a6ff")


def render_pr_reviews(pr_data):
    if not pr_data or "reviews" not in pr_data:
        st.info("No peer reviews available")
        return

    reviews = pr_data["reviews"]
    avg = pr_data.get("avg_score")
    decision = pr_data.get("decision")

    col1, col2 = st.columns(2)
    with col1:
        if avg is not None:
            st.metric("Average Score", f"{avg:.2f} / 10")
    with col2:
        if decision:
            st.metric("Decision", decision.upper())

    tabs = st.tabs([f"Reviewer {j+1}: {r.get('source', 'Unknown')}" for j, r in enumerate(reviews)])

    for j, (tab, review) in enumerate(zip(tabs, reviews)):
        with tab:
            # Header metrics
            c1, c2 = st.columns(2)
            with c1:
                os_score = review.get("overall_score")
                if os_score is not None:
                    st.metric("Overall Score", f"{os_score} / 10")
            with c2:
                rd = review.get("decision", "")
                if rd:
                    st.metric("Decision", rd.upper())

            # Dimension scores
            scores = review.get("scores", {})
            if scores:
                st.markdown("**Dimension Scores**")
                render_dimension_scores(scores)

            # Summary
            summary = review.get("summary", "")
            if summary:
                st.markdown("**Summary**")
                st.write(summary)

            # Strengths
            strengths = review.get("strengths", [])
            if strengths:
                st.markdown(f"**Strengths ({len(strengths)})**")
                for s in strengths:
                    st.markdown(f"- {s}")

            # Weaknesses
            weaknesses = review.get("weaknesses", [])
            if weaknesses:
                st.markdown(f"**Weaknesses ({len(weaknesses)})**")
                for w in weaknesses:
                    st.markdown(f"- {w}")

            # Detailed feedback
            feedback = review.get("detailed_feedback", "")
            if feedback:
                st.markdown("**Detailed Feedback**")
                st.write(feedback)

            # Novelty assessment
            novelty = review.get("novelty_assessment", "")
            if novelty:
                st.markdown("**Novelty Assessment**")
                st.write(novelty)

            # Questions
            questions = review.get("questions_for_authors", "")
            if questions:
                st.markdown("**Questions for Authors**")
                st.write(questions)

            # Integrity check
            integrity = review.get("integrity_check", "")
            if integrity:
                st.markdown("**Integrity Check**")
                st.write(integrity)


def render_sar_review(sar_data):
    if not sar_data:
        st.info("No Stanford AI Review available")
        return

    score = sar_data.get("overall_score")
    sections = sar_data.get("sections", {})

    if score is not None:
        st.markdown(
            f'<div style="text-align:center;padding:1rem 0">'
            f'<span style="font-size:3rem;font-weight:800;color:{score_color(score)}">{score}</span>'
            f'<span style="font-size:1.2rem;color:#8b949e"> / 10</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

    section_map = [
        ("Summary", "summary"),
        ("Strengths", "strengths"),
        ("Weaknesses", "weaknesses"),
        ("Detailed Comments", "detailed_comments"),
        ("Questions for Authors", "questions_for_authors"),
        ("Overall Assessment", "overall_assessment"),
    ]

    for title, key in section_map:
        text = sections.get(key, "")
        if text:
            # Clean boilerplate from overall_assessment
            if key == "overall_assessment":
                for marker in ["We Value Your Feedback", "Full Review", "Submit Your Review"]:
                    idx = text.find(marker)
                    if idx > 0:
                        text = text[:idx].strip()
            st.markdown(f"**{title}**")
            st.write(text)
            st.markdown("---")


def main():
    st.set_page_config(
        page_title="Research Arena — Paper Viewer",
        page_icon="📄",
        layout="wide",
    )

    st.title("Research Arena — Paper Viewer")
    st.caption("Browse all 117 AI-generated research papers with Stanford AI Review and Peer Review scores")

    papers = load_data()

    # ── Sidebar filters ──────────────────────────────────────────
    with st.sidebar:
        st.header("Filters")

        agents = st.multiselect(
            "Agent",
            options=["claude", "codex", "kimi"],
            default=["claude", "codex", "kimi"],
            format_func=lambda x: AGENT_LABELS[x],
        )

        domains = sorted(set(p["domain"] for p in papers))
        selected_domain = st.selectbox("Domain", ["All"] + domains)

        platforms = st.multiselect(
            "Platform",
            options=["gpu", "cpu"],
            default=["gpu", "cpu"],
            format_func=str.upper,
        )

        decisions = st.multiselect(
            "PR Decision",
            options=["accept", "revision", "reject"],
            default=["accept", "revision", "reject"],
            format_func=str.capitalize,
        )

        sort_by = st.selectbox(
            "Sort by",
            ["SAR Score ↓", "SAR Score ↑", "PR Score ↓", "PR Score ↑", "Agent + Domain", "Title A-Z"],
        )

        search = st.text_input("Search titles")

    # ── Filter papers ─────────────────────────────────────────────
    filtered = [
        p
        for p in papers
        if p["agent"] in agents
        and p["platform"] in platforms
        and (p["pr_decision"] or "reject") in decisions
        and (selected_domain == "All" or p["domain"] == selected_domain)
        and (not search or search.lower() in p["title"].lower())
    ]

    # ── Sort ──────────────────────────────────────────────────────
    if sort_by == "SAR Score ↓":
        filtered.sort(key=lambda p: -(p["sar_score"] or 0))
    elif sort_by == "SAR Score ↑":
        filtered.sort(key=lambda p: (p["sar_score"] or 0))
    elif sort_by == "PR Score ↓":
        filtered.sort(key=lambda p: -(p["pr_avg"] or 0))
    elif sort_by == "PR Score ↑":
        filtered.sort(key=lambda p: (p["pr_avg"] or 0))
    elif sort_by == "Agent + Domain":
        filtered.sort(key=lambda p: (p["agent"], p["domain"], p["trial"]))
    elif sort_by == "Title A-Z":
        filtered.sort(key=lambda p: p["title"])

    # ── Summary stats ─────────────────────────────────────────────
    st.markdown(f"**Showing {len(filtered)} of {len(papers)} papers**")

    cols = st.columns(3)
    for i, agent in enumerate(["claude", "codex", "kimi"]):
        agent_papers = [p for p in filtered if p["agent"] == agent]
        if agent_papers:
            sar_avg = sum(p["sar_score"] or 0 for p in agent_papers) / len(agent_papers)
            pr_papers = [p for p in agent_papers if p["pr_avg"] is not None]
            pr_avg = sum(p["pr_avg"] for p in pr_papers) / len(pr_papers) if pr_papers else 0
            with cols[i]:
                color = AGENT_COLORS[agent]
                st.markdown(
                    f'<div style="border-left:4px solid {color};padding:0.5rem 1rem;background:rgba(255,255,255,0.03);border-radius:4px">'
                    f'<b style="color:{color}">{AGENT_LABELS[agent]}</b> ({len(agent_papers)} papers)<br>'
                    f'SAR: <b>{sar_avg:.2f}</b> · PR: <b>{pr_avg:.2f}</b>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

    st.divider()

    # ── Paper list ────────────────────────────────────────────────
    for p in filtered:
        color = AGENT_COLORS[p["agent"]]
        sar = p["sar_score"]
        pr = p["pr_avg"]
        sar_str = f"{sar:.1f}" if sar else "N/A"
        pr_str = f"{pr:.1f}" if pr else "N/A"

        with st.expander(
            f"{agent_badge(p['agent'])}  **{p['title']}**  —  SAR: {sar_str}  ·  PR: {pr_str}",
        ):
            # Paper info row
            c1, c2, c3, c4, c5 = st.columns([2, 1.5, 1, 1, 1])
            with c1:
                st.markdown(f"**Domain:** {p['domain']}")
            with c2:
                st.markdown(f"**Platform:** {p['platform'].upper()} · Trial {p['trial']}")
            with c3:
                st.markdown(
                    f'**SAR:** <span style="color:{score_color(sar)};font-weight:700">{sar_str}</span>',
                    unsafe_allow_html=True,
                )
            with c4:
                st.markdown(
                    f'**PR:** <span style="color:{score_color(pr)};font-weight:700">{pr_str}</span>',
                    unsafe_allow_html=True,
                )
            with c5:
                if p["pr_decision"]:
                    st.markdown(decision_badge(p["pr_decision"]), unsafe_allow_html=True)

            # PDF link
            pdf_path = os.path.join(ROOT, p["path"], "paper.pdf")
            if os.path.exists(pdf_path):
                with open(pdf_path, "rb") as pdf_file:
                    st.download_button(
                        "Download PDF",
                        data=pdf_file.read(),
                        file_name=f"{p['agent']}_{p['seed']}_trial{p['trial']}.pdf",
                        mime="application/pdf",
                        key=f"pdf_{p['index']}",
                    )

            # Review tabs
            paper_key = f"{p['agent']}/{p['seed']}_trial{p['trial']}"
            comments = st.session_state.get("comments", load_comments())
            n_comments = len(comments.get(paper_key, []))
            comment_label = f"My Notes ({n_comments})" if n_comments else "My Notes"
            tab_pr, tab_sar, tab_idea, tab_comments = st.tabs(["Peer Reviews", "Stanford AI Review", "Paper Info", comment_label])

            with tab_pr:
                render_pr_reviews(p["pr_data"])

            with tab_sar:
                render_sar_review(p["sar_data"])

            with tab_idea:
                # Load idea.json
                idea_path = os.path.join(ROOT, p["path"], "idea.json")
                if os.path.exists(idea_path):
                    try:
                        idea = json.load(open(idea_path))
                        if idea.get("title"):
                            st.markdown(f"**Idea Title:** {idea['title']}")
                        if idea.get("description"):
                            st.write(idea["description"])
                        if idea.get("motivation"):
                            st.markdown("**Motivation**")
                            st.write(idea["motivation"])
                        if idea.get("proposed_approach"):
                            st.markdown("**Proposed Approach**")
                            st.write(idea["proposed_approach"])
                    except Exception:
                        st.info("Could not load idea.json")
                else:
                    st.info("No idea.json found")

                # Show experiment files
                exp_dir = os.path.join(ROOT, p["path"], "exp")
                if os.path.isdir(exp_dir):
                    py_files = []
                    for root, dirs, files in os.walk(exp_dir):
                        dirs[:] = [d for d in dirs if d not in [".venv", "__pycache__"]]
                        for f in files:
                            if f.endswith(".py") or f.endswith(".sh"):
                                rel = os.path.relpath(os.path.join(root, f), exp_dir)
                                py_files.append(rel)
                    if py_files:
                        st.markdown(f"**Experiment Code ({len(py_files)} files)**")
                        st.code("\n".join(sorted(py_files)), language="text")

            with tab_comments:
                render_comments(paper_key)


if __name__ == "__main__":
    main()
