"""Regressions for missing intermediate holds and confounded motor-probe parameters."""
import numpy as np

from myo.arm_balanced import specifications, collect_episode, sampling_weights
from myo.arm_precision import ArmPrecisionEnv


def test_balanced_probes_cover_hold_and_independent_capacity_amplitude():
    specs = specifications()
    for split in ('train', 'validation', 'test'):
        rows = [s for s in specs if s['split'] == split]
        for family in ('independent', 'reciprocal', 'common', 'multisine'):
            pairs = {(s['capacity'], s['limit']) for s in rows if s['family'] == family}
            assert pairs == {(c, u) for c in (1., .8) for u in (.12, .4, .8)}
    hold = next(s for s in specs if s['split'] == 'train' and s['family'] == 'hold'
                and s['position_deg'] == 80 and s['capacity'] == 1 and s['coactivation'] == 0)
    data, meta = collect_episode(hold)
    assert not meta['failure'] and data['valid'].all()
    np.testing.assert_array_equal(data['next_observations'][:-1], data['observations'][1:])
    q = data['observations'][:, 0]*90+75
    velocity = data['observations'][:, 1]*4*180/np.pi
    assert np.mean((abs(q[-50:]-80) <= 3) & (abs(velocity[-50:]) <= 5)) >= .9
    assert np.ptp(q[:20]) > .05  # Gravity/delay remain active; this is not a clamped pose.


def test_sampling_reduces_overrepresentation_of_lower_rest():
    env = ArmPrecisionEnv(); obs, _ = env.reset(options={'case': (15, 15)})
    data = np.repeat(obs[None], 110, axis=0)
    data[-10:, 0] = 0.  # Ten middle-angle states versus one hundred lower states.
    probabilities, info = sampling_weights(data, {'contract': env.contract()})
    assert len(info['counts']) == 2
    assert np.isclose(probabilities.sum(), 1)
    assert probabilities[-10:].sum() > 3*(10/110)
    assert probabilities[:100].sum() > .5  # The uniform-data half preserves common states.
    env.close()
