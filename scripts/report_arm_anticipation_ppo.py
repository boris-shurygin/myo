"""Post-selection feature interventions and movement diagnostics for three PPO arms."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from myo.arm_feature_comparison import KINDS, load_comparison, frozen_digest
from myo.arm_precision import ArmPrecisionEnv, evaluate_arm
from myo.train_vertical_ppo import write_json


def trainable_digest(model):
    digest = hashlib.sha256()
    for name, parameter in model.policy.named_parameters():
        if parameter.requires_grad:
            digest.update(name.encode())
            digest.update(parameter.detach().numpy().tobytes())
    return digest.hexdigest()


def common_initial_audit(root):
    models = []
    for kind in KINDS:
        model, env, _ = load_comparison(root/kind/'initial.zip')
        env.close()
        models.append(model)
    assert len({trainable_digest(m) for m in models}) == 1
    # Check every optimizer accumulator, including counters, by parameter name.
    reference = models[0]
    for other in models[1:]:
        assert reference.policy.optimizer.param_groups[0]['lr'] == other.policy.optimizer.param_groups[0]['lr']
        old = dict(reference.policy.named_parameters())
        for name, parameter in other.policy.named_parameters():
            if parameter.requires_grad:
                state = other.policy.optimizer.state[parameter]
                expected = reference.policy.optimizer.state[old[name]]
                assert state.keys() == expected.keys()
                assert all(torch.equal(state[k], v) for k, v in expected.items())
    return {'common_trainable_weights_sha256': trainable_digest(reference),
        'all_trainable_weights_equal': True, 'all_optimizer_states_equal': True,
        'frozen_encoder_sha256': {kind: frozen_digest(m.policy) for kind, m in zip(KINDS, models, strict=True)}}


@torch.no_grad()
def feature_probes(model, observations, expected_validation):
    extractor = model.policy.mlp_extractor
    features = extractor.features(observations)
    native_actions = model.policy.get_distribution(observations).distribution.mean.clone()
    mean = features.mean(0, keepdim=True)
    rows = {'scaled_feature_rms': float(features.square().mean().sqrt()),
        'raw_feature_rms': float(features.square().mean().sqrt()/extractor.feature_gain),
        'feature_temporal_std_rms': float(features.std(0, unbiased=False).square().mean().sqrt()),
        'interventions': {}}
    original = extractor.features
    for label in ('mean', 'zero', 'shuffle'):
        if label == 'shuffle':
            permutation = torch.randperm(len(features), generator=torch.Generator().manual_seed(0))
            replacement = features[permutation]
        else:
            replacement = mean.expand_as(features) if label == 'mean' else torch.zeros_like(features)
        extractor.features = lambda x, replacement=replacement: replacement
        try:
            changed = model.policy.get_distribution(observations).distribution.mean
            rows['interventions'][label] = {
                'mean_abs_excitation_change': float((native_actions-changed).abs().mean()),
                'max_abs_excitation_change': float((native_actions-changed).abs().max())}
        finally:
            extractor.features = original
    # Strict evaluation already replayed native validation before opening test.
    rows['native_validation'] = expected_validation
    zero_actor_readout = not bool(extractor.policy_net[0].weight[:, :128].count_nonzero())
    for label in ('mean', 'zero'):
        replacement = mean if label == 'mean' else torch.zeros_like(mean)
        extractor.features = lambda x, replacement=replacement: replacement.expand(len(x), -1)
        try:
            if model.policy.feature_mode == 'none' or zero_actor_readout:
                # Exact actor-function equivalence: avoid repeating identical rollouts.
                rows[label+'_validation'] = expected_validation
            else:
                rows[label+'_validation'] = evaluate_arm(model)
        finally:
            extractor.features = original
    rows['actor_readout_feature_weight_norm'] = float(extractor.policy_net[0].weight[:, :128].norm())
    rows['critic_readout_feature_weight_norm'] = float(extractor.value_net[0].weight[:, :128].norm())
    return rows


def continuous_demo(model):
    env = ArmPrecisionEnv()
    base = env.unwrapped
    env.reset(seed=0, options={'case': (20, 75)})
    trace, failed = [], False
    for step in range(800):
        target = (75, 20, 105, 45)[step//200]
        base.set_command(float(np.deg2rad(target)), 1.)
        action = model.predict(env.observation(), deterministic=True)[0]
        _, _, failed, truncated, info = env.step(action)
        trace.append([base.time, float(np.rad2deg(info['q'])), float(np.rad2deg(info['qvel'])), target,
                      *np.asarray(action).tolist()])
        if failed:
            break
        if truncated:
            base._episode_done = False
    env.close()
    a = np.asarray(trace)
    return {'failure': bool(failed), 'seconds': len(a)*.02, 'samples': trace,
        'tail_rmse_deg_by_segment': [float(np.sqrt(np.mean((a[start+150:min(start+200, len(a)), 1]-target)**2)))
            if len(a) >= start+200 else None for start, target in zip(range(0, 800, 200), (75, 20, 105, 45), strict=True)]}


def diagnostics(root):
    assert (root/'frozen_selection.json').exists()
    summaries = {kind: json.loads((root/kind/'summary.json').read_text()) for kind in KINDS}
    assert all(s['status'] == 'complete' for s in summaries.values())
    audit = common_initial_audit(root)
    # Same historical training states for all probes; no post-selection retraining.
    source = Path('runs/arm_preacher_balanced/body')
    meta = json.loads((source/'dataset.json').read_text())
    data = np.load(source/'dataset.npz')
    ids = np.concatenate([np.arange(e['start'], e['stop']) for e in meta['episodes'] if e['split'] == 'train'])
    ids = ids[np.linspace(0, len(ids)-1, 5400, dtype=int)]
    observations = torch.as_tensor(data['observations'][ids])
    results = {'initial_audit': audit, 'feature_observations': '5400 evenly indexed historical train-only states',
               'trials': {}, 'last_trials': {}, 'continuous': {}, 'optimization': {}}
    for kind, summary in summaries.items():
        model, env, _ = load_comparison(root/kind/'best.zip')
        env.close()
        before = trainable_digest(model), frozen_digest(model.policy)
        results['trials'][kind] = feature_probes(model, observations, summary['best_validation'])
        assert results['trials'][kind]['native_validation'] == summary['best_validation']
        results['continuous'][kind] = continuous_demo(model)
        assert before == (trainable_digest(model), frozen_digest(model.policy))
        last, env, _ = load_comparison(root/kind/'last.zip')
        env.close()
        if (trainable_digest(last), frozen_digest(last.policy)) == before:
            results['last_trials'][kind] = results['trials'][kind]
        else:
            results['last_trials'][kind] = feature_probes(last, observations, summary['history'][-1]['validation'])
        assert results['last_trials'][kind]['native_validation'] == summary['history'][-1]['validation']
        updates = summary['updates']
        results['optimization'][kind] = {'rollouts': len(updates),
            'optimizer_steps': sum(r['optimizer_steps'] for r in updates),
            'kl_stopped_rollouts': sum(r['optimizer_steps'] < 80 for r in updates),
            'last_explained_variance': updates[-1]['explained_variance'],
            'mean_explained_variance': float(np.mean([r['explained_variance'] for r in updates])),
            'maximum_approx_kl': max(r['approx_kl'] for r in updates)}
    write_json(root/'diagnostics.json', results)
    lines = ['# Использование frozen-признаков после выбора моделей', '',
        '| Ветка | Raw RMS h | Scaled RMS h | Δu при h→mean | Native val, °;pass | Mean val | Zero val |',
        '| --- | ---: | ---: | ---: | --- | --- | --- |']
    for kind, r in results['trials'].items():
        values = [f"{r[label+'_validation']['score_deg']:.3f};{r[label+'_validation']['passed']}/9"
                  for label in ('native', 'mean', 'zero')]
        lines.append(f"| {kind} | {r['raw_feature_rms']:.4f} | {r['scaled_feature_rms']:.4f} | "
            f"{r['interventions']['mean']['mean_abs_excitation_change']:.5f} | "+' | '.join(values)+' |')
    lines += ['', '| Ветка | Tail RMSE при75→20→105→45°, ° | Failure |', '| --- | --- | --- |']
    for kind, r in results['continuous'].items():
        values = '/'.join(f'{v:.3f}' if v is not None else '—' for v in r['tail_rmse_deg_by_segment'])
        lines.append(f"| {kind} | {values} | {r['failure']} |")
    lines += ['', '## Последние checkpoints (без выбора по test)', '',
        '| Ветка | Δu при h→mean | Native val, °;pass | Mean val | Zero val | Actor feature weight norm |',
        '| --- | ---: | --- | --- | --- | ---: |']
    for kind, r in results['last_trials'].items():
        values = [f"{r[label+'_validation']['score_deg']:.3f};{r[label+'_validation']['passed']}/9"
                  for label in ('native', 'mean', 'zero')]
        lines.append(f"| {kind} | {r['interventions']['mean']['mean_abs_excitation_change']:.5f} | "+
            ' | '.join(values)+f" | {r['actor_readout_feature_weight_norm']:.5f} |")
    lines += ['', '| Ветка | SGD | KL-stop /rollouts | Last explained variance |',
              '| --- | ---: | --- | ---: |']
    for kind, r in results['optimization'].items():
        lines.append(f"| {kind} | {r['optimizer_steps']} | {r['kl_stopped_rollouts']}/{r['rollouts']} | "
                     f"{r['last_explained_variance']:.4f} |")
    (root/'diagnostics.md').write_text('\n'.join(lines)+'\n')
    print((root/'diagnostics.md').read_text(), flush=True)


def plot(root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    for kind in KINDS:
        s = json.loads((root/kind/'summary.json').read_text())
        axes[0].plot([r['added_transitions'] for r in s['history']],
                     [r['validation']['score_deg'] for r in s['history']], marker='.', label=kind)
        test = s['best_test']['cases']
        label = kind+' (initial, zero readout)' if s['best_steps'] == s['source_steps'] else kind
        axes[1].plot(range(11), [r['tail_rmse_deg'] for r in test], marker='.', label=label)
    axes[0].set(xlabel='Additional PPO transitions', ylabel='Mean tail RMSE (deg)', title='Validation / common parent')
    axes[1].set(xlabel='Test scenario', ylabel='Tail RMSE (deg)', title='Validation-selected policies')
    axes[1].set_xticks(range(11), ['20→75', '75→20', '30', '60', '90', '120', '130', '15', '25→105', '105→25', '60→125'], rotation=65)
    for axis in axes:
        axis.legend(); axis.grid(alpha=.25)
    for extension in ('png', 'svg'):
        fig.savefig(root/('comparison.'+extension), dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path('runs/arm_anticipation_ppo'))
    p.add_argument('--plot', action='store_true')
    args = p.parse_args()
    torch.set_num_threads(1)
    diagnostics(args.root)
    if args.plot:
        plot(args.root)


if __name__ == '__main__':
    main()
