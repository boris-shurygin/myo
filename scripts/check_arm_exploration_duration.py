"""Diagnostic command holds with a frozen initial policy; no PPO learning."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from myo.arm_learning import load_arm
from myo.train_vertical_architecture import parameter_hash


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=Path('runs/arm_hypotheses/exploration_duration.json'))
    p.add_argument('--concentration-multiplier', type=float, default=1.)
    args = p.parse_args()
    if args.output.exists():
        p.error('Output already exists')
    if not np.isfinite(args.concentration_multiplier) or args.concentration_multiplier <= 0:
        p.error('Concentration multiplier must be positive')
    torch.set_num_threads(1)
    path = Path('runs/arm_preacher/none/initial.zip')
    model, env, meta = load_arm(path)
    before = parameter_hash(model)
    rows = []
    for hold_steps in (1, 5, 10):
        for seed in range(4):
            torch.manual_seed(seed)
            obs, _ = env.reset(seed=0, options={'case': (20, 75)})
            trace = []
            for step in range(200):
                if step % hold_steps == 0:
                    if args.concentration_multiplier == 1.:
                        action = model.predict(obs, deterministic=False)[0]
                    else:
                        with torch.no_grad():
                            d = model.policy.get_distribution(torch.as_tensor(obs[None])).distribution
                            wide = torch.distributions.Beta(d.concentration1*args.concentration_multiplier,
                                                            d.concentration0*args.concentration_multiplier)
                            torch.testing.assert_close(d.mean, wide.mean, rtol=1e-6, atol=1e-7)
                            action = wide.sample()[0].numpy()
                obs, reward, failed, truncated, info = env.step(action)
                assert info['trigger'] == 1.
                trace.append([float(np.rad2deg(info['q'])), reward, *action])
                if failed or truncated:
                    break
            a = np.asarray(trace)
            row = {'hold_ms': hold_steps*20, 'seed': seed, 'failure': bool(failed),
                   'max_angle_deg': float(a[:, 0].max()), 'final_angle_deg': float(a[-1, 0]),
                   'frames_above_initial_plus_1deg': int(np.sum(a[:, 0] > 21)),
                   'return': float(a[:, 1].sum()),
                   'tail_rmse_deg': float(np.sqrt(np.mean((a[-50:, 0]-75)**2))),
                   'trace': a.tolist()}
            rows.append(row)
            print(json.dumps({k: v for k, v in row.items() if k != 'trace'}), flush=True)
    assert parameter_hash(model) == before
    assert hashlib.sha256(path.read_bytes()).hexdigest() == meta['checkpoint_sha256']
    env.close()
    result = {'schema': 'myo.arm_exploration_duration.v1', 'status': 'complete',
              'concentration_multiplier': args.concentration_multiplier,
              'checkpoint_sha256': meta['checkpoint_sha256'], 'results': rows,
              'scope': 'Frozen initial actor; command sampled at block start, then held. '
                       'This changes execution frequency, not a valid drop-in PPO noise algorithm.',
              'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
