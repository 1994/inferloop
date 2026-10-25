import argparse
import hashlib
import itertools
import json
import math
import random
import statistics
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--root', type=Path, required=True)
parser.add_argument('--tag', default='final')
parser.add_argument('--pairs', type=int, default=5)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--profiles', nargs='+', default=['2b-mtp0', '27b-mtp0', '27b-mtp2'])
args = parser.parse_args()
rows = []
provenance = []
pair_decisions = []
unavailable = []
profiles = args.profiles

def interval(values):
    size = len(values)
    if size <= 5:
        samples = itertools.product(range(size), repeat=size)
    else:
        rng = random.Random(20261025)
        samples = ([rng.randrange(size) for _ in range(size)] for _ in range(100000))
    estimates = sorted(statistics.median(values[index] for index in sample) for sample in samples)
    return [estimates[math.floor(.025 * (len(estimates) - 1))], estimates[math.ceil(.975 * (len(estimates) - 1))]]

def report_memory(report):
    path = Path(report['telemetry']['log'])
    if not path.is_absolute():
        path = args.root / "reports" / path
    records = [json.loads(line) for line in path.read_text().splitlines()]
    samples = [row for row in records if row.get('type') == 'sample']
    rss = [sum(process['rss_bytes'] for process in row['processes'].values()) for row in samples]
    peak = [int(row['cgroup']['memory.peak']) for row in samples if row.get('cgroup', {}).get('memory.peak')]
    return {'sampled_server_tree_rss_peak_bytes': max(rss, default=0), 'cgroup_peak_bytes': max(peak, default=0),
        'gpu_peak_mib': report['telemetry']['active_window']['memory_used_max_mib'],
        'sampled_server_cpu_mean_percent_one_core': statistics.mean(
            sum(row['cpu_percent_one_core'].values()) for row in samples if row.get('cpu_percent_one_core'))}

for profile in profiles:
    grouped = {}
    side_reports = {'baseline': [], 'candidate': []}
    for pair in range(args.pairs):
        path = args.root / f'reports/{profile}-{args.tag}-p{pair}-comparison.json'
        comparison = json.loads(path.read_text())
        assert comparison['numeric']['identical'] and comparison['token_mismatches'] == 0
        pair_decisions.append({'profile': profile, 'pair': pair, 'point_estimate_gate_passed': comparison['passed'],
            'comparison_file': str(path), 'comparison_sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
        for metric in comparison['metrics']:
            if metric['available']:
                grouped.setdefault((metric['case'], metric['metric']), []).append(metric)
            else:
                unavailable.append({'profile': profile, 'pair': pair, **metric})
        model = 'qwen3vl-2b' if profile.startswith('2b-') else 'Qwen3.8-27B-NVFP4'
        depth = profile.rsplit('mtp', 1)[1].split('-')[0]
        for side in side_reports:
            report_path = args.root / f'reports/{model}-mtp{depth}-native-{args.tag}-p{pair}-{side}.json'
            report = json.loads(report_path.read_text())
            assert report['completed'] and report['checklist']['compliant']
            assert report['identity']['source']['dirty'] is False
            side_reports[side].append(report)
            provenance.append({'file': str(report_path), 'sha256': hashlib.sha256(report_path.read_bytes()).hexdigest(),
                'source': report['identity']['source'], 'binary_sha256': report['binary_sha256'],
                'memory': report_memory(report)})
    for (case, metric), measurements in sorted(grouped.items()):
        ratios = [value['ratio'] for value in measurements]
        bounds = interval(ratios)
        rows.append({'profile': profile, 'case': case, 'metric': metric, 'paired_units': len(ratios),
            'baseline_median': statistics.median(value['baseline'] for value in measurements),
            'candidate_median': statistics.median(value['candidate'] for value in measurements),
            'ratios': ratios, 'paired_median_ratio': statistics.median(ratios), 'bootstrap_95_percent_ci': bounds,
            'observed_ratio_drift_from_first_pair': max(abs(value / ratios[0] - 1) for value in ratios),
            'regression_decision': 'pass' if bounds[1] <= 1.10 else 'fail' if bounds[0] > 1.10 else 'inconclusive'})
        if metric == 'wall_seconds':
            rows.append({'profile': profile, 'case': case, 'metric': 'output_tokens_per_second_ratio',
                'paired_units': len(ratios), 'ratios': [1 / value for value in ratios],
                'paired_median_ratio': statistics.median(1 / value for value in ratios),
                'bootstrap_95_percent_ci': [1 / bounds[1], 1 / bounds[0]]})
    for case in sorted({trial['case'] for report in side_reports['baseline'] for trial in report['trials']}):
        tail = {}
        for side, reports in side_reports.items():
            values = sorted(result['ttft_seconds'] for report in reports for trial in report['trials']
                if trial['case'] == case and not trial['warmup'] for result in trial['results'])
            tail[side] = {'requests': len(values), 'observed_p95_ttft_seconds': values[math.ceil(.95 * len(values)) - 1],
                'max_ttft_seconds': max(values)}
        rows.append({'profile': profile, 'case': case, 'metric': 'finite_sample_ttft_tail', **tail})
    rows.append({'profile': profile, 'metric': 'startup_seconds', **{
        side: [report['startup_seconds'] for report in reports] for side, reports in side_reports.items()}})

summary = {'schema': 1, 'method': 'Process pairs are independent clusters. Exact empirical median-ratio bootstrap for N<=5; 100000 deterministic draws (seed 20261025) for larger N. Pointwise 95% intervals, not a simultaneous SLO guarantee.',
    'metrics': rows, 'provenance': provenance, 'pair_decisions': pair_decisions, 'unavailable': unavailable,
    'stable_baseline_certified': False,
    'stability_note': 'This summary does not certify a new stable baseline. Observed ratio drift from the first pair is reported per metric against the declared 10% tolerance; historical baselines are unchanged.',
    'all_primary_latency_intervals_pass': all(row['regression_decision'] == 'pass' for row in rows if 'regression_decision' in row)}
args.output.write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps({'all_primary_latency_intervals_pass': summary['all_primary_latency_intervals_pass'],
    'latency_cases': sum('regression_decision' in row for row in rows)}, indent=2))
