"""Check difficult training cells separately from held-out arm validation."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing
from pathlib import Path

import torch

from myo.arm_workload import evaluate_workload, scenarios
from myo.arm_workload_evaluation import evaluate_parallel
from myo.train_vertical_ppo import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--controllers', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, choices=(1, 2, 4), default=2)
    args = parser.parse_args()
    torch.set_num_threads(1)
    conditions = scenarios(((120, 120), (135, 135)), masses=(.5, 1., 2.), tilts=(120., 150.))
    report = {'schema': 'myo.arm_workload_coverage_probe.v1', 'status': 'running',
              'scope': 'twelve predefined training cells; diagnostic, not held-out test',
              'probe_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'conditions': conditions, 'models': {}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context('spawn')) as pool:
        for path in args.controllers:
            meta = json.loads(Path(str(path)+'.metadata.json').read_text())
            assert hashlib.sha256(path.read_bytes()).hexdigest() == meta['checkpoint_sha256']
            spec = meta['training']['spec']
            assert all(list(c['case']) in spec['cases'] and c['payload_mass'] in spec['masses']
                       and c['tilt_deg'] in spec['tilts'] and c['capacity'] in spec['capacities']
                       for c in conditions), 'These diagnostic cells must belong to training'
            value = evaluate_parallel(pool, path, conditions, args.workers)
            report['models'][str(path)] = {'checkpoint_sha256': meta['checkpoint_sha256'],
                                         'steps': meta['steps'], 'result': value}
            write_json(args.output, report)
            print(json.dumps({'controller': str(path), 'score_deg': value['score_deg'],
                              'passed': value['passed'], 'total': value['total']}), flush=True)
    report['heuristic'] = evaluate_workload('heuristic', conditions)
    report['status'] = 'complete'
    write_json(args.output, report)
    print(json.dumps({'controller': 'heuristic', 'score_deg': report['heuristic']['score_deg'],
                      'passed': report['heuristic']['passed'], 'total': report['heuristic']['total']}), flush=True)


if __name__ == '__main__':
    main()
