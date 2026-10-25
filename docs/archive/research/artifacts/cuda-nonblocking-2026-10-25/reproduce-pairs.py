import argparse
import hashlib
import json
import subprocess
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--pairs', type=int, default=5)
parser.add_argument('--start', type=int, default=0)
parser.add_argument('--profiles', nargs='+', default=['2b-mtp0', '27b-mtp0', '27b-mtp2'])
parser.add_argument('--tag', default='final')
args = parser.parse_args()
root = Path('/home/r/benchmarks/inferloop/cuda-nonblocking-v1')
tuning = root / 'inputs/tuning.json'
tuning_hash = hashlib.sha256(tuning.read_bytes()).hexdigest()
for profile in args.profiles:
    small = profile.startswith('2b-')
    depth = int(profile.rsplit('mtp', 1)[1].split('-')[0])
    suffix = '2b' if small else profile
    checkpoint = '/home/r/models/qwen3vl-2b' if small else '/home/r/models/Qwen3.8-27B-NVFP4'
    context = json.loads((root / f'inputs/profile-{suffix}-async.json').read_text())['resources']['max_model_len']
    name = Path(checkpoint).name
    for pair in range(args.start, args.start + args.pairs):
        sides = ['baseline', 'candidate'] if pair % 2 == 0 else ['candidate', 'baseline']
        reports = {}
        for side in sides:
            repo = '/home/r/code/inference-cuda-default-baseline' if side == 'baseline' else '/home/r/code/inference-cuda-nonblocking'
            binary = 'infer-baseline' if side == 'baseline' else 'infer-candidate'
            run_id = f'{args.tag}-p{pair}-{side}'
            command = ['bash', f'{repo}/tools/bench/safe-run.sh', '--memory-gib', '16' if small else '32',
                'env', f'INFER_CUDA_TUNING_TABLE={tuning}',
                'python3', f'{repo}/tools/bench/serve-compare.py', '--engine', 'native',
                '--model', checkpoint, '--executable', f'{repo}/artifacts/{binary}',
                '--inputs', str(root / f'inputs/inputs-{suffix}-async.json'),
                '--checklist', str(root / f'inputs/profile-{suffix}-async.json'),
                '--profile-id', f'{profile}-async-execution', '--output-dir', str(root / 'reports'),
                '--run-id', run_id, '--mtp', str(depth), '--tokens', '64', '--repeats', '3',
                '--gpu-memory-utilization', '0.88', '--max-model-len', str(context),
                '--max-num-seqs', '16', '--extra', '--no-autotune']
            print(f'START {profile} pair={pair} side={side}', flush=True)
            subprocess.run(command, cwd=repo, check=True)
            if hashlib.sha256(tuning.read_bytes()).hexdigest() != tuning_hash:
                raise RuntimeError('frozen tuning table changed')
            reports[side] = root / f'reports/{name}-mtp{depth}-native-{run_id}.json'
        baseline_report = json.loads(reports['baseline'].read_text())
        candidate_report = json.loads(reports['candidate'].read_text())
        assert baseline_report['config_readback'] == candidate_report['config_readback'], 'effective configuration differs'
        for before, after in zip(baseline_report['trials'], candidate_report['trials'], strict=True):
            assert (before['case'], before['repeat']) == (after['case'], after['repeat'])
            assert before['prefix_tokens_reused'] == after['prefix_tokens_reused'], 'cache workload differs'
            original = {row['slot']: row for row in before['results']}
            revised = {row['slot']: row for row in after['results']}
            assert original.keys() == revised.keys()
            assert all(original[slot]['finish_reason'] == revised[slot]['finish_reason'] for slot in original)
        comparison_path = root / f'reports/{profile}-{args.tag}-p{pair}-comparison.json'
        with comparison_path.open('x') as output:
            checked = subprocess.run(['python3', '/home/r/code/inference-cuda-nonblocking/tools/bench/compare-results.py',
                str(reports['baseline']), str(reports['candidate']), '--reference-engine', 'native',
                '--require-identical-tokens'], stdout=output)
        comparison = json.loads(comparison_path.read_text())
        if comparison.get('token_mismatches') or not comparison.get('numeric', {}).get('identical'):
            raise RuntimeError(f'output mismatch: {comparison_path}')
        print(f'PAIR {profile} p{pair} gate={checked.returncode}', flush=True)
        print(json.dumps([{k: m.get(k) for k in ['case', 'metric', 'ratio', 'available']} for m in comparison['metrics']]), flush=True)
