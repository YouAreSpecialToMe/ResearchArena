"""
SAR Ambiguous Review Annotator
Manually classify ambiguous Stanford Agentic Reviewer reviews as Accept/Reject/Unclear.

Usage:
    streamlit run sar_annotator.py
"""

import json
import re
import os
from pathlib import Path
import streamlit as st

ROOT = Path(__file__).parent
ANNOTATIONS_FILE = ROOT / "analysis" / "sar_annotations.json"


def classify(text):
    text_l = text.lower()
    has_accept = bool(re.search(r'\b(accept|acceptance)\b', text_l))
    has_reject = bool(re.search(r'\b(reject|rejection)\b', text_l))
    if has_accept and not has_reject:
        return 'accept'
    if has_reject and not has_accept:
        return 'reject'
    if has_accept and has_reject:
        ra = bool(re.search(r'(recommend.*accept|lean.*toward.*accept|borderline.*accept|could.*be accepted|worth.*accept|merits.*accept|suitable for.*accept)', text_l))
        rr = bool(re.search(r'(recommend.*reject|lean.*toward.*reject|borderline.*reject|should.*be rejected|not.*ready.*accept)', text_l))
        if ra and not rr:
            return 'accept'
        if rr and not ra:
            return 'reject'
    return 'ambiguous'


@st.cache_data
def load_all_data():
    """Load ICLR accepted/rejected + FARS papers (excludes ICLR random)."""
    items = []

    # ICLR baseline (300 papers across 6 workers) — skip random
    BASE = ROOT / "analysis" / "iclr2025_baseline" / "stanford_reviews"
    all_iclr = {}
    for i in range(6):
        f = BASE / f'assessments_w{i:02d}.json'
        if f.exists():
            all_iclr.update(json.load(open(f)))

    for key, v in all_iclr.items():
        category = key.rsplit('_', 1)[0]  # accepted/rejected/random
        if category == 'random':
            continue  # skip random
        text = v.get('assessment', '')
        items.append({
            'id': f'iclr_{key}',
            'source': f'ICLR {category}',
            'score': v.get('score'),
            'text': text,
            'auto_label': classify(text),
        })

    # FARS (102 papers)
    fars_file = ROOT / "analysis" / "stanford_reviews" / "fars_assessments.json"
    if fars_file.exists():
        fars = json.load(open(fars_file))
        for key, v in fars.items():
            text = v.get('assessment', '')
            items.append({
                'id': f'fars_{key}',
                'source': 'FARS (Analemma)',
                'score': v.get('score'),
                'text': text,
                'auto_label': classify(text),
            })

    return items


def load_annotations():
    if ANNOTATIONS_FILE.exists():
        return json.load(open(ANNOTATIONS_FILE))
    return {}


def save_annotations(ann):
    ANNOTATIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(ANNOTATIONS_FILE, 'w') as f:
        json.dump(ann, f, indent=2)


def set_label(item_id, label):
    if 'annotations' not in st.session_state:
        st.session_state.annotations = load_annotations()
    st.session_state.annotations[item_id] = label
    save_annotations(st.session_state.annotations)


def main():
    st.set_page_config(page_title="SAR Annotator", layout="wide")
    st.title("SAR Ambiguous Review Annotator")
    st.caption("Classify ambiguous Stanford AI Reviewer assessments as Accept / Reject / Unclear")

    items = load_all_data()

    if 'annotations' not in st.session_state:
        st.session_state.annotations = load_annotations()
    ann = st.session_state.annotations

    # Sidebar: filters & progress
    with st.sidebar:
        st.header("Progress")
        total = len(items)
        done = sum(1 for it in items if it['id'] in ann)
        st.metric("Annotated", f"{done} / {total}")
        st.progress(done / total if total else 0)

        counts = {'accept': 0, 'reject': 0, 'unclear': 0}
        for it in items:
            label = ann.get(it['id'])
            if label in counts:
                counts[label] += 1
        st.write(f"**Accept**: {counts['accept']}")
        st.write(f"**Reject**: {counts['reject']}")
        st.write(f"**Unclear**: {counts['unclear']}")

        st.divider()

        st.header("Filter")
        filter_source = st.selectbox(
            "Source",
            ["All", "ICLR accepted", "ICLR rejected", "FARS (Analemma)"],
        )
        filter_auto = st.multiselect(
            "Auto-classification",
            ["accept", "reject", "ambiguous"],
            default=["accept", "reject", "ambiguous"],
        )
        filter_status = st.radio(
            "Annotation status",
            ["All", "Unannotated only", "Annotated only"],
        )

        st.divider()

        if st.button("Export results"):
            out = {}
            for it in items:
                out[it['id']] = {
                    'source': it['source'],
                    'score': it['score'],
                    'label': ann.get(it['id'], 'unannotated'),
                }
            export_path = ROOT / "analysis" / "sar_annotations_export.json"
            with open(export_path, 'w') as f:
                json.dump(out, f, indent=2)
            st.success(f"Exported to {export_path}")

    # Filter items
    filtered = items
    if filter_source != "All":
        filtered = [it for it in filtered if it['source'] == filter_source]
    filtered = [it for it in filtered if it['auto_label'] in filter_auto]
    if filter_status == "Unannotated only":
        filtered = [it for it in filtered if it['id'] not in ann]
    elif filter_status == "Annotated only":
        filtered = [it for it in filtered if it['id'] in ann]

    if not filtered:
        st.success("All done in this filter! Switch to 'All' to re-review.")
        return

    # Navigation
    if 'idx' not in st.session_state:
        st.session_state.idx = 0
    st.session_state.idx = min(st.session_state.idx, len(filtered) - 1)

    idx = st.session_state.idx
    item = filtered[idx]

    # Header
    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
    with col1:
        st.subheader(f"{idx + 1} / {len(filtered)} — {item['source']}")
    with col2:
        score = item['score']
        color = "#3fb950" if score and score >= 5 else "#f85149"
        st.markdown(
            f'<div style="text-align:center;padding:0.5rem;background:#161b22;border-radius:6px">'
            f'<span style="font-size:0.75rem;color:#8b949e">SAR Score</span><br>'
            f'<span style="font-size:1.8rem;font-weight:700;color:{color}">{score if score else "N/A"}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
    with col3:
        auto = item['auto_label']
        auto_colors = {'accept': '#3fb950', 'reject': '#f85149', 'ambiguous': '#d29922'}
        st.markdown(
            f'<div style="text-align:center;padding:0.5rem;background:#161b22;border-radius:6px">'
            f'<span style="font-size:0.75rem;color:#8b949e">Auto-class</span><br>'
            f'<span style="font-size:1.1rem;font-weight:700;color:{auto_colors.get(auto,"#888")}">{auto.upper()}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
    with col4:
        current = ann.get(item['id'], 'unannotated')
        current_colors = {'accept': '#3fb950', 'reject': '#f85149', 'unclear': '#d29922', 'unannotated': '#8b949e'}
        st.markdown(
            f'<div style="text-align:center;padding:0.5rem;background:#161b22;border-radius:6px">'
            f'<span style="font-size:0.75rem;color:#8b949e">Your Label</span><br>'
            f'<span style="font-size:1.1rem;font-weight:700;color:{current_colors.get(current,"#888")}">{current.upper()}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown(f"**ID:** `{item['id']}`")

    # Review text
    st.markdown("### Overall Assessment Text")
    st.markdown(
        f'<div style="background:#0d1117;border:1px solid #30363d;border-radius:6px;padding:1rem;max-height:500px;overflow-y:auto;white-space:pre-wrap;font-family:monospace;font-size:0.9rem;color:#c9d1d9">'
        f'{item["text"]}'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Labeling buttons
    st.markdown("### Classify this review")
    b1, b2, b3, b4, b5, b6 = st.columns(6)
    with b1:
        if st.button("✅ ACCEPT", use_container_width=True, type="primary" if current == 'accept' else "secondary"):
            set_label(item['id'], 'accept')
            st.session_state.idx = min(idx + 1, len(filtered) - 1)
            st.rerun()
    with b2:
        if st.button("❌ REJECT", use_container_width=True, type="primary" if current == 'reject' else "secondary"):
            set_label(item['id'], 'reject')
            st.session_state.idx = min(idx + 1, len(filtered) - 1)
            st.rerun()
    with b3:
        if st.button("❓ UNCLEAR", use_container_width=True, type="primary" if current == 'unclear' else "secondary"):
            set_label(item['id'], 'unclear')
            st.session_state.idx = min(idx + 1, len(filtered) - 1)
            st.rerun()
    with b4:
        if st.button("↶ Prev", use_container_width=True):
            st.session_state.idx = max(0, idx - 1)
            st.rerun()
    with b5:
        if st.button("Next ↷", use_container_width=True):
            st.session_state.idx = min(len(filtered) - 1, idx + 1)
            st.rerun()
    with b6:
        if st.button("🗑️ Clear", use_container_width=True):
            if item['id'] in ann:
                del ann[item['id']]
                save_annotations(ann)
                st.rerun()

    # Jump to
    st.markdown("### Jump to item")
    new_idx = st.number_input("Go to #", min_value=1, max_value=len(filtered), value=idx + 1, step=1)
    if new_idx != idx + 1:
        st.session_state.idx = new_idx - 1
        st.rerun()


if __name__ == "__main__":
    main()
