"""A changed curriculum must not masquerade as a compatible continuation."""
from copy import deepcopy
import json

import pytest

from myo.movement_curriculum import STAGE_NAMES, curriculum_metadata
from myo.train_movement import validate_source


def test_movement_source_distinguishes_fresh_initialization_and_exact_continuation():
    descriptor = curriculum_metadata(1.0)
    initial = {
        "reward_scale": 0.01,
        "action_persistence": 0.0,
        "bounded_distribution": "beta_alpha_beta_softplus_floor_0.05",
        "predictor_sha256": None,
        "trained_steps": 0,
    }
    validate_source(initial, descriptor, resume=False)
    trained = {
        **initial,
        "trained_steps": 1472,
        "movement_curriculum": json.loads(json.dumps(descriptor)),
        "movement_curriculum_stage": 0,
    }
    validate_source(trained, descriptor, resume=True)
    with pytest.raises(ValueError, match="untrained"):
        validate_source(trained, descriptor, resume=False)

    # The same action/observation spaces do not make changed exercises an
    # exact continuation: their reward outcomes and stopping gates differ.
    changed = deepcopy(descriptor)
    changed["stages"][0]["scenarios"][0]["commands"][0]["target_deg"] = 45.0
    with pytest.raises(ValueError, match="training contract"):
        validate_source(trained, changed, resume=True)

    # An out-of-range stage would skip the training loop and misleadingly
    # return a "running" summary without evaluating or updating the model.
    with pytest.raises(ValueError, match="valid saved curriculum stage"):
        validate_source({**trained, "movement_curriculum_stage": len(STAGE_NAMES)},
                        descriptor, resume=True)

    # The old prepared-hold experiment used persistent actions. It cannot be
    # silently imported as the initial state of this direct-excitation run.
    with pytest.raises(ValueError, match="direct Beta"):
        validate_source({**initial, "action_persistence": 0.9}, descriptor, resume=False)
