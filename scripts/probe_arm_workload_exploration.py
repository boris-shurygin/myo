"""Compare deterministic Beta means and sampled commands in native arm physics."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from myo.arm_workload import evaluate_workload, load_workload, scenarios
from myo.train_vertical_ppo import write_json


# Diagnostic validation cells, chosen before looking at stochastic outcomes.
CONDITIONS = (
    *scenarios(((125, 125),), masses=(1.5, 2.), tilts=(135.,)),
    *scenarios(((40, 40),), masses=(2.,), tilts=(195., 255.)),
    *scenarios(((70, 70),), masses=(.75, 2.), tilts=(15.,)),
)


class CommandProbe:
    def __init__(self, model, stochastic):
        self.model = model
        self.observation_space = model.observation_space
        self.stochastic = stochastic
        self.distributions = []

    def predict(self, observation, deterministic=True):
        with torch.no_grad():
            tensor, _ = self.model.policy.obs_to_tensor(observation)
            distribution = self.model.policy.get_distribution(tensor)
            beta = distribution.distribution
            self.distributions.append(np.concatenate((
                beta.mean.cpu().numpy()[0], beta.stddev.cpu().numpy()[0])))
            action = distribution.get_actions(deterministic=not self.stochastic)
            return action.cpu().numpy().reshape(self.model.action_space.shape), None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--controller', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seeds', type=int, nargs='+', default=(0, 1, 2))
    args = parser.parse_args()
    torch.set_num_threads(1)
    model, checked, meta = load_workload(args.controller)
    checked.close()
    summary_path = args.controller.parent/'summary.json'
    history = json.loads(summary_path.read_text())['history'] if summary_path.exists() else []
    matching = [h['validation']['cases'] for h in history if h['steps'] == meta['steps']]
    report = {'status': 'running', 'checkpoint_sha256': meta['checkpoint_sha256'],
              'probe_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'selection': 'six predefined validation cells; not test or new training data',
              'seeds': args.seeds, 'cases': []}
    for condition in CONDITIONS:
        for stochastic in (False, True):
            for seed in args.seeds if stochastic else (0,):
                probe = CommandProbe(model, stochastic)
                # Keep sampled evaluation reproducible without changing other RNG users.
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(seed)
                    value = evaluate_workload(probe, (condition,), samples=True)
                row = value['cases'][0]
                if not stochastic and matching:
                    expected = [r for r in matching[-1] if all(
                        json.dumps(r[k]) == json.dumps(v) for k, v in condition.items())]
                    if expected:
                        actual = {k: v for k, v in row.items() if k != 'samples'}
                        assert json.dumps(actual, sort_keys=True) == json.dumps(expected[0], sort_keys=True)
                        row['deterministic_validation_replay'] = True
                trace = np.asarray(row['samples'])
                tail = trace[:, 0] > condition['seconds']-1.-1e-8
                if not tail.any():
                    tail = np.ones(len(trace), dtype=bool)
                distributions = np.asarray(probe.distributions)
                row.update(stochastic=stochastic, seed=seed,
                    tail_bias_deg=float(np.mean(trace[tail, 1]-trace[tail, 3])),
                    tail_velocity_rms_deg_s=float(np.sqrt(np.mean(trace[tail, 2]**2))),
                    tail_command_distribution_mean=distributions[tail, :6].mean(0).tolist(),
                    tail_command_distribution_std=distributions[tail, 6:].mean(0).tolist(),
                    tail_issued_command_mean=trace[tail, 4:10].mean(0).tolist())
                report['cases'].append(row)
                args.output.parent.mkdir(parents=True, exist_ok=True)
                write_json(args.output, report)
                print(json.dumps({k: row[k] for k in (
                    'case', 'payload_mass', 'tilt_deg', 'stochastic', 'seed',
                    'tail_rmse_deg', 'passed', 'failure', 'activation_mean')}), flush=True)
    report['status'] = 'complete'
    write_json(args.output, report)


if __name__ == '__main__':
    main()
