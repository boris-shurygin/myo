"""Assess selected/last arm policies without stochastic exploration."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
from pathlib import Path

import torch

from myo.arm_exploration import evaluation_job, load_exploration
from myo.arm_workload import scenarios
from myo.arm_workload_evaluation import merge_results
from myo.train_vertical_ppo import write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--roots', type=Path, nargs='+', required=True)
    p.add_argument('--test', type=Path, default=Path('runs/arm_reward_curriculum/test.json'))
    p.add_argument('--output', type=Path, default=Path('runs/arm_exploration/comparison.json'))
    args = p.parse_args()
    torch.set_num_threads(1)
    comparison = {'evaluation': 'deterministic mean; no added exploration noise', 'roots': []}
    with ProcessPoolExecutor(max_workers=2, mp_context=multiprocessing.get_context('spawn')) as pool:
        for root in args.roots:
            summary = json.loads((root/'summary.json').read_text())
            meta = json.loads((root/'best.zip.metadata.json').read_text())
            spec = meta['contract']['spec']
            train = scenarios(spec['cases'], spec['masses'], spec['tilts'], spec['capacities'], spec['seconds'])
            suites = {'train': train, 'validation': summary['training']['validation'],
                      'test': json.loads(args.test.read_text())}
            row = {'root': str(root), 'best_added': summary['best_added'], 'checkpoints': {}}
            for checkpoint in ('best', 'last'):
                if checkpoint == 'last' and summary['best_added'] == summary['added_transitions']:
                    best, checked, best_meta = load_exploration(root/'best.zip')
                    checked.close()
                    last, checked, last_meta = load_exploration(root/'last.zip')
                    checked.close()
                    identical = (best_meta['contract'] == last_meta['contract']
                        and best.num_timesteps == last.num_timesteps
                        and all(torch.equal(v, last.policy.state_dict()[k])
                                for k,v in best.policy.state_dict().items()))
                    if identical:
                        row['checkpoints']['last'] = row['checkpoints']['best']
                        row['last_equivalent_to_best'] = 'same contract, timestep and all policy tensors verified'
                        print(json.dumps({'root': str(root), 'last': 'identical to best; assessment reused'}), flush=True)
                        continue
                results = {}
                for suite, conditions in suites.items():
                    jobs = [pool.submit(evaluation_job, root/f'{checkpoint}.zip', conditions[i::2]) for i in range(2)]
                    results[suite] = merge_results([job.result() for job in jobs])
                row['checkpoints'][checkpoint] = results
                print(json.dumps({'root': str(root), 'checkpoint': checkpoint,
                    **{suite: {'passed': r['passed'], 'total': r['total'], 'error_deg': r['score_deg'],
                               'failures': r['failures']} for suite,r in results.items()}}), flush=True)
            row['exploration'] = summary['exploration']
            row['added_optimizer_steps'] = summary['added_optimizer_steps']
            comparison['roots'].append(row)
            write_json(args.output, comparison)
    comparison['status'] = 'complete'
    write_json(args.output, comparison)


if __name__ == '__main__':
    main()
