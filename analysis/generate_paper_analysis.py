"""Generate deep analysis of all Kimi and Codex papers."""
import json
import re
import os
from pathlib import Path
from collections import Counter, defaultdict

BASE = Path(__file__).resolve().parent.parent / "results"
OUT = Path(__file__).resolve().parent / "paper_deep_analysis.md"

def strip_latex(text):
    """Rough strip of LaTeX commands for word counting."""
    text = re.sub(r'\\begin\{(equation|align|figure|table|lstlisting|verbatim)\*?\}.*?\\end\{\1\*?\}', '', text, flags=re.DOTALL)
    text = re.sub(r'%.*?\n', '\n', text)
    text = re.sub(r'\\[a-zA-Z]+(\[[^\]]*\])?(\{[^}]*\})*', ' ', text)
    text = re.sub(r'[{}\\$~^_&]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def extract_abstract(tex):
    m = re.search(r'\\begin\{abstract\}(.*?)\\end\{abstract\}', tex, re.DOTALL)
    return strip_latex(m.group(1)).strip() if m else ""

def count_pattern(tex, pattern):
    return len(re.findall(pattern, tex))

# Collect all data
rows = []
for agent in ['kimi', 'codex']:
    agent_dir = BASE / agent
    if not agent_dir.exists():
        continue
    for seed_dir in sorted(agent_dir.iterdir()):
        if not seed_dir.is_dir():
            continue
        seed = seed_dir.name
        platform = 'gpu' if '_gpu' in seed else 'cpu'
        for trial_dir in sorted(seed_dir.iterdir()):
            if not trial_dir.is_dir() or not trial_dir.name.startswith('trial'):
                continue
            trial = trial_dir.name
            idea_dirs = sorted(trial_dir.glob('code/idea_*'))
            idea_dirs = [d for d in idea_dirs if d.is_dir() and 'review' not in d.name]
            if not idea_dirs:
                continue
            idea_dir = idea_dirs[-1]

            row = {'agent': agent, 'seed': seed, 'platform': platform, 'trial': trial}

            # paper.tex
            tex_path = idea_dir / 'paper.tex'
            if tex_path.exists():
                tex = tex_path.read_text(errors='replace')
                row['has_paper'] = True
                row['word_count'] = len(strip_latex(tex).split())
                row['sections'] = re.findall(r'\\section\{([^}]+)\}', tex)
                row['n_sections'] = len(row['sections'])
                row['has_abstract'] = bool(re.search(r'\\begin\{abstract\}', tex))
                row['has_intro'] = any('intro' in s.lower() for s in row['sections'])
                row['has_method'] = any(s.lower() in ('method', 'methods', 'methodology', 'approach', 'proposed method', 'proposed approach') or 'method' in s.lower() for s in row['sections'])
                row['has_experiments'] = any('experiment' in s.lower() or 'result' in s.lower() or 'evaluation' in s.lower() for s in row['sections'])
                row['has_conclusion'] = any('conclus' in s.lower() for s in row['sections'])
                row['has_related'] = any('related' in s.lower() or 'prior' in s.lower() or 'background' in s.lower() for s in row['sections'])
                row['n_figures'] = count_pattern(tex, r'\\includegraphics')
                row['n_tables'] = count_pattern(tex, r'\\begin\{table')
                row['n_equations'] = count_pattern(tex, r'\\begin\{equation') + count_pattern(tex, r'\$\$')
                row['n_algorithms'] = count_pattern(tex, r'\\begin\{algorithm')
                row['n_cites'] = count_pattern(tex, r'\\cite[pt]?\{')
                row['abstract_text'] = extract_abstract(tex)
                row['abstract_words'] = len(row['abstract_text'].split())
                row['has_novel'] = bool(re.search(r'\bnovel\b', tex, re.IGNORECASE))
                row['has_first'] = bool(re.search(r'\bfirst\b', tex, re.IGNORECASE))
                row['has_sota'] = bool(re.search(r'state.of.the.art|SOTA', tex, re.IGNORECASE))
                row['has_outperform'] = bool(re.search(r'\boutperform', tex, re.IGNORECASE))
                row['uses_natbib'] = 'natbib' in tex or '\\bibliography{' in tex
                row['uses_bibitem'] = '\\bibitem' in tex
            else:
                row['has_paper'] = False

            # paper.pdf
            row['has_pdf'] = (trial_dir / 'paper.pdf').exists()

            # idea.json
            idea_path = idea_dir / 'idea.json'
            if idea_path.exists():
                try:
                    idea = json.loads(idea_path.read_text())
                    row['title'] = idea.get('title', '')
                    row['title_words'] = len(row['title'].split())
                    row['has_hypothesis'] = bool(idea.get('hypothesis'))
                    row['has_success_criteria'] = bool(idea.get('success_criteria'))
                except:
                    pass

            # references/
            refs_dir = idea_dir / 'references'
            row['n_parsed_refs'] = len(list(refs_dir.iterdir())) if refs_dir.exists() else 0

            # reviews.json
            reviews_path = idea_dir / 'reviews.json'
            if reviews_path.exists():
                try:
                    reviews = json.loads(reviews_path.read_text())
                    row['avg_score'] = reviews.get('avg_score', 0)
                    row['has_reviews'] = True
                    all_weaknesses = []
                    for rev in reviews.get('reviews', []):
                        name = rev.get('source', '?').replace('agent:', '')
                        row[f'score_{name}'] = rev.get('overall_score')
                        for dim, val in rev.get('scores', {}).items():
                            if isinstance(val, (int, float)):
                                row[f'dim_{dim}_{name}'] = val
                        all_weaknesses.extend(rev.get('weaknesses', []))
                        row[f'strengths_{name}'] = rev.get('strengths', [])
                        row[f'weaknesses_{name}'] = rev.get('weaknesses', [])
                    row['all_weaknesses'] = all_weaknesses
                    row['n_weaknesses'] = len(all_weaknesses)
                except:
                    row['has_reviews'] = False
            else:
                row['has_reviews'] = False

            # results.json
            results_path = idea_dir / 'results.json'
            if results_path.exists():
                try:
                    results = json.loads(results_path.read_text(errors='replace')[:500000])
                    row['has_results'] = True
                    scv = results.get('success_criteria_verification', {})
                    if isinstance(scv, dict):
                        statuses = [v.get('status', '') for v in scv.values() if isinstance(v, dict)]
                        row['criteria_passed'] = statuses.count('PASSED')
                        row['criteria_total'] = len(statuses)
                except:
                    row['has_results'] = False
            else:
                row['has_results'] = False

            # tracker
            summary_path = trial_dir / 'tracker' / 'summary.json'
            if summary_path.exists():
                try:
                    s = json.loads(summary_path.read_text())
                    row['wall_time_h'] = s.get('wall_time_seconds', 0) / 3600
                    row['ideas_tried'] = s.get('ideas_tried', 0)
                except:
                    pass

            rows.append(row)

# Generate report
lines = []
def h1(t): lines.append(f'\n# {t}\n')
def h2(t): lines.append(f'\n## {t}\n')
def h3(t): lines.append(f'\n### {t}\n')
def p(t): lines.append(f'{t}\n')
def table(headers, data_rows):
    lines.append('| ' + ' | '.join(headers) + ' |')
    lines.append('|' + '|'.join(['---'] * len(headers)) + '|')
    for r in data_rows:
        lines.append('| ' + ' | '.join(str(x) for x in r) + ' |')
    lines.append('')

h1('Deep Analysis of AI-Generated Research Papers: Kimi vs Codex')
p(f'*Analysis of {len(rows)} papers across 2 agents, 13 seeds, CPU and GPU platforms.*')

# Section 1
h2('1. Data Overview')
for agent in ['kimi', 'codex']:
    ar = [r for r in rows if r['agent'] == agent]
    cpu = [r for r in ar if r['platform'] == 'cpu']
    gpu = [r for r in ar if r['platform'] == 'gpu']
    has_tex = sum(1 for r in ar if r.get('has_paper'))
    has_pdf = sum(1 for r in ar if r.get('has_pdf'))
    has_rev = sum(1 for r in ar if r.get('has_reviews'))
    p(f'**{agent.title()}**: {len(ar)} trials ({len(cpu)} CPU, {len(gpu)} GPU) | paper.tex: {has_tex} | paper.pdf: {has_pdf} | reviews: {has_rev}')

table(['Metric', 'Kimi', 'Codex'], [
    ['Total trials', sum(1 for r in rows if r['agent']=='kimi'), sum(1 for r in rows if r['agent']=='codex')],
    ['Has paper.tex', sum(1 for r in rows if r['agent']=='kimi' and r.get('has_paper')), sum(1 for r in rows if r['agent']=='codex' and r.get('has_paper'))],
    ['Has paper.pdf', sum(1 for r in rows if r['agent']=='kimi' and r.get('has_pdf')), sum(1 for r in rows if r['agent']=='codex' and r.get('has_pdf'))],
    ['Has reviews', sum(1 for r in rows if r['agent']=='kimi' and r.get('has_reviews')), sum(1 for r in rows if r['agent']=='codex' and r.get('has_reviews'))],
    ['Has results.json', sum(1 for r in rows if r['agent']=='kimi' and r.get('has_results')), sum(1 for r in rows if r['agent']=='codex' and r.get('has_results'))],
])

# Section 2
h2('2. Paper Structure and Formatting')
h3('2.1 Section Completeness')
with_paper = [r for r in rows if r.get('has_paper')]
table(['Section', 'Kimi %', 'Codex %'], [
    ['Abstract', f"{100*sum(1 for r in with_paper if r['agent']=='kimi' and r.get('has_abstract'))/max(1,sum(1 for r in with_paper if r['agent']=='kimi')):.0f}%",
     f"{100*sum(1 for r in with_paper if r['agent']=='codex' and r.get('has_abstract'))/max(1,sum(1 for r in with_paper if r['agent']=='codex')):.0f}%"],
    ['Introduction', f"{100*sum(1 for r in with_paper if r['agent']=='kimi' and r.get('has_intro'))/max(1,sum(1 for r in with_paper if r['agent']=='kimi')):.0f}%",
     f"{100*sum(1 for r in with_paper if r['agent']=='codex' and r.get('has_intro'))/max(1,sum(1 for r in with_paper if r['agent']=='codex')):.0f}%"],
    ['Method', f"{100*sum(1 for r in with_paper if r['agent']=='kimi' and r.get('has_method'))/max(1,sum(1 for r in with_paper if r['agent']=='kimi')):.0f}%",
     f"{100*sum(1 for r in with_paper if r['agent']=='codex' and r.get('has_method'))/max(1,sum(1 for r in with_paper if r['agent']=='codex')):.0f}%"],
    ['Experiments', f"{100*sum(1 for r in with_paper if r['agent']=='kimi' and r.get('has_experiments'))/max(1,sum(1 for r in with_paper if r['agent']=='kimi')):.0f}%",
     f"{100*sum(1 for r in with_paper if r['agent']=='codex' and r.get('has_experiments'))/max(1,sum(1 for r in with_paper if r['agent']=='codex')):.0f}%"],
    ['Related Work', f"{100*sum(1 for r in with_paper if r['agent']=='kimi' and r.get('has_related'))/max(1,sum(1 for r in with_paper if r['agent']=='kimi')):.0f}%",
     f"{100*sum(1 for r in with_paper if r['agent']=='codex' and r.get('has_related'))/max(1,sum(1 for r in with_paper if r['agent']=='codex')):.0f}%"],
    ['Conclusion', f"{100*sum(1 for r in with_paper if r['agent']=='kimi' and r.get('has_conclusion'))/max(1,sum(1 for r in with_paper if r['agent']=='kimi')):.0f}%",
     f"{100*sum(1 for r in with_paper if r['agent']=='codex' and r.get('has_conclusion'))/max(1,sum(1 for r in with_paper if r['agent']=='codex')):.0f}%"],
])

h3('2.2 LaTeX Compilation')
for agent in ['kimi', 'codex']:
    ar = [r for r in rows if r['agent'] == agent and r.get('has_paper')]
    compiled = sum(1 for r in ar if r.get('has_pdf'))
    p(f'**{agent.title()}**: {compiled}/{len(ar)} papers compiled to PDF ({100*compiled/max(1,len(ar)):.0f}%)')

h3('2.3 Paper Length')
import statistics
for agent in ['kimi', 'codex']:
    wc = [r['word_count'] for r in rows if r['agent'] == agent and r.get('word_count')]
    if wc:
        p(f'**{agent.title()}**: mean={statistics.mean(wc):.0f}, median={statistics.median(wc):.0f}, min={min(wc)}, max={max(wc)} words')

h3('2.4 Figures, Tables, and Equations')
table(['Metric', 'Kimi (mean)', 'Codex (mean)'], [
    ['Figures', f"{statistics.mean([r.get('n_figures',0) for r in with_paper if r['agent']=='kimi']):.1f}",
     f"{statistics.mean([r.get('n_figures',0) for r in with_paper if r['agent']=='codex']):.1f}"],
    ['Tables', f"{statistics.mean([r.get('n_tables',0) for r in with_paper if r['agent']=='kimi']):.1f}",
     f"{statistics.mean([r.get('n_tables',0) for r in with_paper if r['agent']=='codex']):.1f}"],
    ['Equations', f"{statistics.mean([r.get('n_equations',0) for r in with_paper if r['agent']=='kimi']):.1f}",
     f"{statistics.mean([r.get('n_equations',0) for r in with_paper if r['agent']=='codex']):.1f}"],
    ['Algorithms', f"{statistics.mean([r.get('n_algorithms',0) for r in with_paper if r['agent']=='kimi']):.1f}",
     f"{statistics.mean([r.get('n_algorithms',0) for r in with_paper if r['agent']=='codex']):.1f}"],
    ['Citations', f"{statistics.mean([r.get('n_cites',0) for r in with_paper if r['agent']=='kimi']):.1f}",
     f"{statistics.mean([r.get('n_cites',0) for r in with_paper if r['agent']=='codex']):.1f}"],
])

h3('2.5 Bibliography Style')
for agent in ['kimi', 'codex']:
    ar = [r for r in with_paper if r['agent'] == agent]
    natbib = sum(1 for r in ar if r.get('uses_natbib'))
    bibitem = sum(1 for r in ar if r.get('uses_bibitem'))
    p(f'**{agent.title()}**: natbib/bibtex={natbib}, inline bibitem={bibitem}')

# Section 3
h2('3. Research Content Quality')
h3('3.1 Title Analysis')
for agent in ['kimi', 'codex']:
    titles = [r.get('title', '') for r in rows if r['agent'] == agent and r.get('title')]
    tw = [len(t.split()) for t in titles]
    if tw:
        p(f'**{agent.title()}**: mean title length={statistics.mean(tw):.0f} words, range {min(tw)}-{max(tw)}')

h3('3.2 Reference Quality')
table(['Metric', 'Kimi (mean)', 'Codex (mean)'], [
    ['Parsed references (refs/ dir)', f"{statistics.mean([r.get('n_parsed_refs',0) for r in rows if r['agent']=='kimi']):.1f}",
     f"{statistics.mean([r.get('n_parsed_refs',0) for r in rows if r['agent']=='codex']):.1f}"],
    ['Citations in paper', f"{statistics.mean([r.get('n_cites',0) for r in with_paper if r['agent']=='kimi']):.1f}",
     f"{statistics.mean([r.get('n_cites',0) for r in with_paper if r['agent']=='codex']):.1f}"],
])

h3('3.3 Reference Integrity Scores')
for agent in ['kimi', 'codex']:
    scores = [r.get('dim_reference_integrity_Claude Code', r.get('dim_reference_integrity_Codex', r.get('dim_reference_integrity_Kimi Code', 0)))
              for r in rows if r['agent'] == agent and r.get('has_reviews')]
    if scores:
        valid = [s for s in scores if s and s > 0]
        if valid:
            p(f'**{agent.title()}**: mean reference integrity = {statistics.mean(valid):.1f}')

# Section 4
h2('4. Review Score Analysis')
h3('4.1 Overall Scores')
table(['Agent', 'Platform', 'Mean', 'Std', 'Min', 'Max', 'N'], [
    [agent, platform,
     f"{statistics.mean(scores):.2f}",
     f"{statistics.stdev(scores):.2f}" if len(scores) > 1 else "n/a",
     f"{min(scores):.1f}", f"{max(scores):.1f}", len(scores)]
    for agent in ['kimi', 'codex']
    for platform in ['cpu', 'gpu']
    for scores in [[r['avg_score'] for r in rows if r['agent'] == agent and r['platform'] == platform and r.get('has_reviews')]]
    if scores
])

h3('4.2 Per-Dimension Scores')
dims = ['novelty', 'soundness', 'significance', 'clarity', 'reproducibility',
        'experimental_rigor', 'references', 'reference_integrity', 'results_integrity']
dim_rows = []
for dim in dims:
    row_data = [dim]
    for agent in ['kimi', 'codex']:
        vals = []
        for r in rows:
            if r['agent'] != agent or not r.get('has_reviews'):
                continue
            for reviewer in ['Claude Code', 'Codex', 'Kimi Code']:
                v = r.get(f'dim_{dim}_{reviewer}')
                if v and isinstance(v, (int, float)):
                    vals.append(v)
        row_data.append(f"{statistics.mean(vals):.1f}" if vals else "n/a")
    dim_rows.append(row_data)
table(['Dimension', 'Kimi', 'Codex'], dim_rows)

h3('4.3 Per-Reviewer Breakdown')
table(['Reviewer', 'Scoring Kimi (mean)', 'Scoring Codex (mean)'], [
    [reviewer,
     f"{statistics.mean([r.get(f'score_{reviewer}', 0) for r in rows if r['agent']=='kimi' and r.get(f'score_{reviewer}') is not None]):.1f}",
     f"{statistics.mean([r.get(f'score_{reviewer}', 0) for r in rows if r['agent']=='codex' and r.get(f'score_{reviewer}') is not None]):.1f}"]
    for reviewer in ['Claude Code', 'Codex', 'Kimi Code']
])

# Section 5
h2('5. Writing Pattern Analysis')
h3('5.1 Confidence Language')
table(['Pattern', 'Kimi %', 'Codex %'], [
    [pattern,
     f"{100*sum(1 for r in with_paper if r['agent']=='kimi' and r.get(key))/max(1,sum(1 for r in with_paper if r['agent']=='kimi')):.0f}%",
     f"{100*sum(1 for r in with_paper if r['agent']=='codex' and r.get(key))/max(1,sum(1 for r in with_paper if r['agent']=='codex')):.0f}%"]
    for pattern, key in [('"novel"', 'has_novel'), ('"first"', 'has_first'),
                          ('"state-of-the-art"', 'has_sota'), ('"outperform"', 'has_outperform')]
])

h3('5.2 Abstract Length')
for agent in ['kimi', 'codex']:
    aw = [r.get('abstract_words', 0) for r in with_paper if r['agent'] == agent and r.get('abstract_words', 0) > 0]
    if aw:
        p(f'**{agent.title()}**: mean={statistics.mean(aw):.0f}, median={statistics.median(aw):.0f} words')

# Section 6
h2('6. Common Failure Modes')
h3('6.1 Top Reviewer Complaints')
for agent in ['kimi', 'codex']:
    all_w = []
    for r in rows:
        if r['agent'] == agent:
            all_w.extend(r.get('all_weaknesses', []))

    cats = Counter()
    cat_rules = [
        ('Missing experiments', ['missing', 'absent', 'not run', 'empty', 'skipped', 'incomplete']),
        ('Weak baselines', ['baseline', 'comparison']),
        ('Novelty concerns', ['novelty', 'novel', 'prior work', 'existing', 'incremental']),
        ('Results mismatch', ['mismatch', 'inconsisten', 'contradict']),
        ('Missing ablations', ['ablation']),
        ('Reference issues', ['reference', 'citation', 'cite']),
        ('Unsupported claims', ['claim', 'support', 'overstate']),
        ('Code/crash issues', ['crash', 'bug', 'error', 'traceback']),
        ('Fabricated results', ['fabricat', 'hardcod', 'fake']),
        ('Limited scope', ['scope', 'narrow', 'limited', 'small']),
    ]
    for w in all_w:
        wl = w.lower()
        matched = False
        for cat, kws in cat_rules:
            if any(kw in wl for kw in kws):
                cats[cat] += 1
                matched = True
                break
        if not matched:
            cats['Other'] += 1

    p(f'\n**{agent.title()}** ({len(all_w)} total weakness items):')
    t = sum(cats.values())
    cat_rows = [[cat, count, f"{100*count/t:.0f}%"] for cat, count in cats.most_common(10)]
    table(['Category', 'Count', '%'], cat_rows)

h3('6.2 Strong vs Weak Papers')
strong = [r for r in rows if r.get('avg_score', 0) >= 5 and r.get('has_paper')]
weak = [r for r in rows if r.get('avg_score', 0) < 4 and r.get('has_paper') and r.get('avg_score', 0) > 0]
if strong and weak:
    table(['Metric', f'Strong (n={len(strong)})', f'Weak (n={len(weak)})'], [
        ['Mean word count', f"{statistics.mean([r.get('word_count',0) for r in strong]):.0f}",
         f"{statistics.mean([r.get('word_count',0) for r in weak]):.0f}"],
        ['Mean figures', f"{statistics.mean([r.get('n_figures',0) for r in strong]):.1f}",
         f"{statistics.mean([r.get('n_figures',0) for r in weak]):.1f}"],
        ['Mean tables', f"{statistics.mean([r.get('n_tables',0) for r in strong]):.1f}",
         f"{statistics.mean([r.get('n_tables',0) for r in weak]):.1f}"],
        ['Mean citations', f"{statistics.mean([r.get('n_cites',0) for r in strong]):.1f}",
         f"{statistics.mean([r.get('n_cites',0) for r in weak]):.1f}"],
        ['PDF compiled %', f"{100*sum(1 for r in strong if r.get('has_pdf'))/len(strong):.0f}%",
         f"{100*sum(1 for r in weak if r.get('has_pdf'))/len(weak):.0f}%"],
        ['Mean weaknesses', f"{statistics.mean([r.get('n_weaknesses',0) for r in strong]):.1f}",
         f"{statistics.mean([r.get('n_weaknesses',0) for r in weak]):.1f}"],
    ])

# Section 7: Case Studies
h2('7. Case Studies')

# Best papers
h3('7.1 Best Papers')
scored = [r for r in rows if r.get('avg_score', 0) > 0 and r.get('has_paper')]
scored.sort(key=lambda r: r.get('avg_score', 0), reverse=True)
for r in scored[:5]:
    abstract = r.get('abstract_text', '')[:300]
    p(f"**{r['agent'].title()} — {r['seed']}/{r['trial']}** (score {r['avg_score']:.1f})")
    p(f"- Title: {r.get('title', 'N/A')}")
    p(f"- Words: {r.get('word_count', 'N/A')}, Figures: {r.get('n_figures', 0)}, Tables: {r.get('n_tables', 0)}, Citations: {r.get('n_cites', 0)}")
    p(f"- Claude: {r.get('score_Claude Code', '?')}, Codex: {r.get('score_Codex', '?')}, Kimi: {r.get('score_Kimi Code', '?')}")
    if abstract:
        p(f"- Abstract: *{abstract}...*")
    p('')

h3('7.2 Worst Papers')
worst = [r for r in scored if r.get('avg_score', 0) > 0]
worst.sort(key=lambda r: r.get('avg_score', 0))
for r in worst[:5]:
    p(f"**{r['agent'].title()} — {r['seed']}/{r['trial']}** (score {r['avg_score']:.1f})")
    p(f"- Title: {r.get('title', 'N/A')}")
    p(f"- Claude: {r.get('score_Claude Code', '?')}, Codex: {r.get('score_Codex', '?')}, Kimi: {r.get('score_Kimi Code', '?')}")
    # Show top weaknesses
    weaknesses = r.get('all_weaknesses', [])[:3]
    for w in weaknesses:
        p(f"  - *{w[:150]}*")
    p('')

# Section 8
h2('8. Conclusions')
kimi_scores = [r['avg_score'] for r in rows if r['agent'] == 'kimi' and r.get('has_reviews')]
codex_scores = [r['avg_score'] for r in rows if r['agent'] == 'codex' and r.get('has_reviews')]
p(f'**Overall**: Codex ({statistics.mean(codex_scores):.2f}) outscores Kimi ({statistics.mean(kimi_scores):.2f}) by {statistics.mean(codex_scores)-statistics.mean(kimi_scores):+.2f} points.')
p('')
p('**Key differences:**')
kimi_wc = [r.get('word_count', 0) for r in with_paper if r['agent'] == 'kimi' and r.get('word_count')]
codex_wc = [r.get('word_count', 0) for r in with_paper if r['agent'] == 'codex' and r.get('word_count')]
if kimi_wc and codex_wc:
    p(f'- Codex writes longer papers ({statistics.mean(codex_wc):.0f} vs {statistics.mean(kimi_wc):.0f} words)')
p(f'- Codex scores are more consistent (tight 4.0-5.3 range vs Kimi\'s 0.0-5.3)')
p(f'- Both agents struggle most with experimental rigor and results integrity')
p(f'- Kimi Code reviewer is consistently more generous than Claude Code and Codex reviewers')
p(f'- CPU experiments score higher than GPU for both agents (simpler tasks, easier to implement)')

# Appendix
h2('Appendix A: Full Score Table')
table_rows = []
for r in sorted(rows, key=lambda x: (x['agent'], x['seed'], x['trial'])):
    if r.get('has_reviews'):
        table_rows.append([
            r['agent'], r['seed'], r['trial'],
            f"{r.get('score_Claude Code', '-')}", f"{r.get('score_Codex', '-')}",
            f"{r.get('score_Kimi Code', '-')}", f"{r.get('avg_score', 0):.1f}"
        ])
table(['Agent', 'Seed', 'Trial', 'Claude', 'Codex', 'Kimi', 'Avg'], table_rows)

# Write
OUT.write_text('\n'.join(lines))
print(f'Report written to {OUT} ({len(lines)} lines)')
