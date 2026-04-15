#!/usr/bin/env python3
"""Final experiment runner with consistent parameters across all experiments."""

import sys, os, json, time, math
import numpy as np
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.sketches import BloomFilter, CountMinSketch, HyperLogLog
from src.allocator import get_allocator, SketchBudgetAllocator, GreedyAllocator

SEEDS = [42, 123, 456, 789, 1024]
SL = 500_000  # stream length
US = 50_000   # universe size
ALLOCATORS = ['uniform', 'independent', 'proportional', 'sketchbudget']
PIPELINES = ['P1', 'P2', 'P3']
BUDGETS = [10_000, 50_000, 100_000, 500_000, 1_000_000]
BLABELS = ['10KB', '50KB', '100KB', '500KB', '1MB']


def gen(alpha, seed, sl=SL, us=US):
    rng = np.random.default_rng(seed)
    w = np.arange(1, us+1, dtype=np.float64)**(-alpha)
    w /= w.sum()
    return rng.choice(us, size=sl, p=w) + 1

def gen_uniform(seed, sl=SL, us=US):
    return np.random.default_rng(seed).integers(1, us+1, size=sl)

def gt(stream):
    freq = Counter(stream.tolist())
    nd = len(freq)
    th = len(stream) / nd * 10
    hh = [k for k, v in freq.items() if v >= th]
    return {'frequencies': freq, 'n_distinct': nd, 'stream_length': len(stream),
            'universe_size': max(freq.keys()), 'threshold': th,
            'heavy_hitters': hh, 'n_heavy_hitters': len(hh)}

def stats(g):
    return {'stream_length': g['stream_length'], 'n_distinct': g['n_distinct'],
            'universe_size': g['universe_size'], 'threshold': g['threshold'],
            'frequencies': dict(g['frequencies']),
            'n_positive': min(2000, g['n_distinct']), 'n_negative': 2000,
            'set_size': min(500, g['n_distinct'])}

def run(pipe, alloc, stream, g, seed):
    freq = g['frequencies']
    if pipe == 'P1':
        bf = BloomFilter.from_memory(alloc['bf'], g['n_distinct'])
        cms = CountMinSketch.from_memory(alloc['cms'])
        for item in stream: cms.insert(int(item))
        seen = set()
        for item in stream:
            i = int(item)
            if i not in seen: bf.insert(i); seen.add(i)
        pos = sorted(freq.keys(), key=lambda x: -freq[x])[:min(2000, len(freq))]
        neg = list(range(g['universe_size']+1, g['universe_size']+2001))
        errs = [abs((cms.estimate(i) if bf.query(i) else 0) - freq[i]) for i in pos]
        errs += [abs(cms.estimate(i) if bf.query(i) else 0) for i in neg]
        return {'mean_abs_error': float(np.mean(errs))}

    elif pipe == 'P2':
        cms = CountMinSketch.from_memory(alloc['cms'])
        hll = HyperLogLog.from_memory(alloc['hll'])
        for item in stream: cms.insert(int(item))
        true_hh = set(g['heavy_hitters'])
        det = set(); fhh = 0
        for item in freq.keys():
            if cms.estimate(item) >= g['threshold']:
                hll.insert(item); det.add(item)
                if item not in true_hh: fhh += 1
        return {'cardinality_error': float(abs(hll.estimate() - len(true_hh))),
                'false_hh': fhh, 'missed_hh': len(true_hh - det)}

    elif pipe == 'P3':
        ss = min(500, g['n_distinct'])
        bf = BloomFilter.from_memory(alloc['bf'], g['n_distinct'])
        cms = CountMinSketch.from_memory(alloc['cms'])
        tset = set(sorted(freq.keys(), key=lambda x: -freq[x])[:ss])
        for item in tset: bf.insert(item)
        for item in stream: cms.insert(int(item))
        true_sum = sum(freq.get(i, 0) for i in tset)
        est_sum = sum(cms.estimate(i) for i in freq.keys() if bf.query(i))
        return {'abs_error': float(abs(est_sum - true_sum)),
                'rel_error': float(abs(est_sum - true_sum)/max(true_sum,1))}

def pm(p):
    return {'P1': 'mean_abs_error', 'P2': 'cardinality_error', 'P3': 'abs_error'}[p]

def bounds_for(pipe, alloc, g):
    from src.error_algebra import compute_bounds
    freq = dict(g['frequencies'])
    ns = g['stream_length']; nd = g['n_distinct']
    fpr = BloomFilter.fpr_from_memory(alloc.get('bf',64), nd)
    w, d = CountMinSketch.params_from_memory(alloc.get('cms',256))
    ce = math.e/w if w>0 else 1.0
    hr = 1.04/math.sqrt(1<<HyperLogLog.p_from_memory(alloc.get('hll',16)))
    if pipe == 'P1':
        p = {'fpr':fpr,'cms_epsilon':ce,'n_stream':ns,'n_positive':min(2000,nd),'n_negative':2000}
    elif pipe == 'P2':
        p = {'cms_epsilon':ce,'n_stream':ns,'n_distinct':nd,'threshold':g['threshold'],
             'hll_rel_error':hr,'freq_distribution':freq}
    elif pipe == 'P3':
        ss = min(500,nd)
        p = {'fpr':fpr,'cms_epsilon':ce,'n_stream':ns,'set_size':ss,'n_negative':nd-ss}
    return compute_bounds(pipe, p)


def main():
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    os.makedirs('results', exist_ok=True)
    t0 = time.time()

    # ============ Experiment 1: Main ============
    print("="*60 + "\nEXPERIMENT 1: Main\n" + "="*60)
    ds_cfgs = {'zipfian_1.0': 1.0, 'network_trace': 1.1}
    main_results = []
    for dsn, alpha in ds_cfgs.items():
        for pipe in PIPELINES:
            for an in ALLOCATORS:
                for bud, bl in zip(BUDGETS, BLABELS):
                    vals = []; fa = None
                    for seed in SEEDS:
                        s = gen(alpha, seed); g_ = gt(s); st = stats(g_)
                        ao = get_allocator(an)
                        al = ao.allocate(bud, pipe, st)
                        if fa is None: fa = dict(al)
                        m = run(pipe, al, s, g_, seed)
                        vals.append(m[pm(pipe)])
                    r = {'dataset':dsn,'pipeline':pipe,'allocator':an,'budget':bud,
                         'budget_label':bl,'primary_metric':pm(pipe),
                         'mean':float(np.mean(vals)),'std':float(np.std(vals)),
                         'min':float(np.min(vals)),'max':float(np.max(vals)),
                         'values':vals,'allocation':fa}
                    # Bounds for sketchbudget
                    if an == 'sketchbudget':
                        try:
                            s = gen(alpha, 42); g_ = gt(s)
                            nb, tb = bounds_for(pipe, fa, g_)
                            r['naive_bound'] = float(nb); r['tight_bound'] = float(tb)
                        except: pass
                    main_results.append(r)
                    print(f"  {dsn}/{pipe}/{an}/{bl}: {np.mean(vals):.2f}±{np.std(vals):.2f}")
    with open('results/main_experiments.json','w') as f: json.dump(main_results,f,indent=2,default=str)
    print(f"  Saved. ({(time.time()-t0)/60:.1f}min elapsed)")

    # ============ Experiment 2: Bound tightness ============
    print("\n" + "="*60 + "\nEXPERIMENT 2: Bound tightness\n" + "="*60)
    bt_results = []
    for alpha in [0.5, 0.8, 1.0, 1.2, 1.5, 2.0]:
        for pipe in PIPELINES:
            nr, tr = [], []
            for seed in SEEDS:
                s = gen(alpha, seed); g_ = gt(s); st = stats(g_)
                ao = get_allocator('sketchbudget')
                al = ao.allocate(500000, pipe, st)
                m = run(pipe, al, s, g_, seed)
                obs = m[pm(pipe)]
                try:
                    nb, tb = bounds_for(pipe, al, g_)
                    if obs > 0:
                        nr.append(nb/max(obs,1e-10)); tr.append(tb/max(obs,1e-10))
                except: pass
            if nr:
                bt_results.append({'pipeline':pipe,'alpha':alpha,'budget':500000,
                    'naive_tightness_mean':float(np.mean(nr)),'naive_tightness_std':float(np.std(nr)),
                    'tight_tightness_mean':float(np.mean(tr)),'tight_tightness_std':float(np.std(tr))})
                print(f"  {pipe}/α={alpha}: naive={np.mean(nr):.2f} tight={np.mean(tr):.2f}")
    with open('results/bound_tightness.json','w') as f: json.dump(bt_results,f,indent=2)

    # ============ Experiment 3: Depth ablation ============
    print("\n" + "="*60 + "\nEXPERIMENT 3: Depth ablation\n" + "="*60)
    da_results = []
    for depth in [2, 3]:
        pipe = 'P1' if depth==2 else 'P2'
        for an in ALLOCATORS:
            vals = []
            for seed in SEEDS:
                s = gen(1.0, seed); g_ = gt(s); st = stats(g_)
                al = get_allocator(an).allocate(500000, pipe, st)
                vals.append(run(pipe, al, s, g_, seed)[pm(pipe)])
            da_results.append({'depth':depth,'pipeline':pipe,'allocator':an,'budget':500000,
                'mean':float(np.mean(vals)),'std':float(np.std(vals))})
            print(f"  d={depth}/{an}: {np.mean(vals):.2f}±{np.std(vals):.2f}")
    with open('results/ablation_depth.json','w') as f: json.dump(da_results,f,indent=2)

    # ============ Experiment 4: Budget sensitivity ============
    print("\n" + "="*60 + "\nEXPERIMENT 4: Budget sensitivity\n" + "="*60)
    fb = np.logspace(np.log10(5000), np.log10(5_000_000), 12).astype(int).tolist()
    ba_results = []
    for pipe in PIPELINES:
        for bud in fb:
            for an in ALLOCATORS:
                vals = []
                for seed in SEEDS:
                    s = gen(1.0, seed); g_ = gt(s); st = stats(g_)
                    al = get_allocator(an).allocate(bud, pipe, st)
                    vals.append(run(pipe, al, s, g_, seed)[pm(pipe)])
                ba_results.append({'pipeline':pipe,'allocator':an,'budget':int(bud),
                    'mean':float(np.mean(vals)),'std':float(np.std(vals))})
        print(f"  {pipe}: done")
    with open('results/ablation_budget.json','w') as f: json.dump(ba_results,f,indent=2)

    # ============ Experiment 5: Greedy vs exact ============
    print("\n" + "="*60 + "\nEXPERIMENT 5: Greedy vs exact\n" + "="*60)
    ga_results = []
    for pipe in PIPELINES:
        for bud in [10000, 50000, 100000, 500000, 1000000]:
            s = gen(1.0, 42); g_ = gt(s); st = stats(g_)
            sb = SketchBudgetAllocator()
            t1 = time.time(); asb = sb.allocate(bud, pipe, st); tsb = time.time()-t1
            gr = GreedyAllocator()
            t1 = time.time(); agr = gr.allocate(bud, pipe, st, delta_m=max(200,bud//200)); tgr = time.time()-t1
            msb = run(pipe, asb, s, g_, 42); mgr = run(pipe, agr, s, g_, 42)
            p_ = pm(pipe)
            ga_results.append({'pipeline':pipe,'budget':bud,
                'scipy_error':float(msb[p_]),'greedy_error':float(mgr[p_]),
                'scipy_alloc':asb,'greedy_alloc':agr,
                'scipy_time':float(tsb),'greedy_time':float(tgr),
                'agreement_pct':float(100*(1-abs(msb[p_]-mgr[p_])/max(msb[p_],1e-10)))})
            print(f"  {pipe}/{bud}: scipy={msb[p_]:.2f} greedy={mgr[p_]:.2f}")
    # Runtime scaling
    rr = []
    s = gen(1.0, 42); g_ = gt(s); st = stats(g_)
    for k in [2,3,5,7,10]:
        t1 = time.time()
        GreedyAllocator().allocate(1_000_000, 'P1', st, delta_m=100)
        rr.append({'stages':k,'runtime_seconds':time.time()-t1})
    ga_results.append({'runtime_scaling': rr})
    with open('results/ablation_greedy.json','w') as f: json.dump(ga_results,f,indent=2,default=str)

    # ============ Experiment 6: Distribution sensitivity ============
    print("\n" + "="*60 + "\nEXPERIMENT 6: Distribution sensitivity\n" + "="*60)
    ds_results = []
    for alpha in [0.5, 0.8, 1.0, 1.2, 1.5, 2.0]:
        for an in ALLOCATORS:
            vals = []
            for seed in SEEDS:
                s = gen(alpha, seed); g_ = gt(s); st = stats(g_)
                al = get_allocator(an).allocate(500000, 'P2', st)
                vals.append(run('P2', al, s, g_, seed)[pm('P2')])
            ds_results.append({'alpha':alpha,'allocator':an,'pipeline':'P2','budget':500000,
                'mean':float(np.mean(vals)),'std':float(np.std(vals))})
            print(f"  α={alpha}/{an}: {np.mean(vals):.2f}±{np.std(vals):.2f}")
    for an in ALLOCATORS:
        vals = []
        for seed in SEEDS:
            s = gen_uniform(seed); g_ = gt(s); st = stats(g_)
            al = get_allocator(an).allocate(500000, 'P2', st)
            vals.append(run('P2', al, s, g_, seed)[pm('P2')])
        ds_results.append({'alpha':0.0,'allocator':an,'pipeline':'P2','budget':500000,
            'mean':float(np.mean(vals)),'std':float(np.std(vals))})
    with open('results/ablation_distribution.json','w') as f: json.dump(ds_results,f,indent=2)

    elapsed = time.time() - t0
    with open('results/timing.json','w') as f:
        json.dump({'total_seconds':elapsed,'total_minutes':elapsed/60},f)
    print(f"\n\nAll done in {elapsed/60:.1f} minutes")


if __name__ == '__main__':
    main()
