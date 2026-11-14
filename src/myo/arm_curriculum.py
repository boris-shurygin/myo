"""Task/LR curriculum for fresh 10/10-ms arm PPO; physics remains ArmTimingEnv."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

from stable_baselines3.common.utils import FloatSchedule

from .arm_precision import TRAIN_CASES, VALIDATION_CASES, TEST_CASES
from .arm_workload import WorkloadSpec, scenarios


@dataclass(frozen=True)
class CurriculumStage:
    name: str
    spec: WorkloadSpec
    learning_rate: float
    validation: tuple
    test: tuple
    minimum_steps: int = 65536
    maximum_steps: int = 262144

    def manifest(self):
        return asdict(self)


def conditions(cases, masses=(.5,), tilts=(30.,)):
    return scenarios(cases, masses, tilts, (1.,), 4.)


def representative(cases, masses, tilts):
    """Balanced small gate; full Cartesian training audit is required to advance."""
    return [{"case": case, "payload_mass": masses[i % len(masses)],
             "tilt_deg": tilts[(i // len(masses)) % len(tilts)],
             "capacity": 1., "seconds": 4.}
            for i, case in enumerate(cases)]


def curriculum():
    easy = tuple((goal + offset, goal) for goal in (45., 75., 105.) for offset in (-15., 0., 15.))
    validation = tuple(conditions(tuple((goal + offset, goal)
        for goal in (60., 90.) for offset in (-10., 0., 10.))))
    test = tuple(conditions(((40.,55.), (55.,55.), (70.,55.), (85.,100.), (100.,100.), (115.,100.))))
    simple = WorkloadSpec(masses=(.5,), tilts=(30.,), cases=easy)
    broad = WorkloadSpec(masses=(.5,), tilts=(30.,), cases=TRAIN_CASES)
    mass = WorkloadSpec(masses=(.5,1.,2.), tilts=(30.,), cases=TRAIN_CASES)
    nearby = WorkloadSpec(masses=(.5,1.,2.), tilts=(0.,30.,60.), cases=TRAIN_CASES)
    circle = WorkloadSpec(masses=(.5,1.,2.), tilts=tuple(float(x) for x in range(0,360,30)), cases=TRAIN_CASES)
    return (
        CurriculumStage('local_acquisition', simple, 3e-4, validation, test),
        # Same task, lower LR: explicitly separate acquisition from precision.
        CurriculumStage('local_precision', simple, 1e-4, validation, test, maximum_steps=131072),
        CurriculumStage('angle_range', broad, 1e-4,
            tuple(conditions(VALIDATION_CASES)), tuple(conditions(TEST_CASES))),
        CurriculumStage('payloads', mass, 3e-5,
            tuple(conditions(VALIDATION_CASES, (.75,1.5))),
            tuple(conditions(TEST_CASES, (.6,1.25,1.75)))),
        CurriculumStage('nearby_mounts', nearby, 3e-5,
            tuple(representative(VALIDATION_CASES*2, (.75,1.5), (15.,45.))),
            tuple(representative(TEST_CASES*2, (.6,1.25,1.75), (10.,50.)))),
        CurriculumStage('full_circle', circle, 1e-5,
            tuple(representative(VALIDATION_CASES*3, (.75,1.5), tuple(float(x) for x in range(15,360,30)))),
            tuple(representative(TEST_CASES*4, (.6,1.25,1.75), tuple(float(x) for x in range(10,360,30))))),
    )


def set_learning_rate(model, value):
    """Change both SB3's future updates and the current RMSprop groups, without reset."""
    model.learning_rate = float(value)
    model.lr_schedule = FloatSchedule(float(value))
    for group in model.policy.optimizer.param_groups:
        group['lr'] = float(value)


def passed_gate(result, *, strict=False):
    required = result['total'] if strict else math.ceil(.9*result['total'])
    return (result['total'] > 0 and result['failures'] == 0
            and result['passed'] >= required and result['score_deg'] <= 3.)


def improved(result, reference):
    """Accumulate small score changes until they reach 2%; prioritize successful cases."""
    return reference is None or (result['failures'], -result['passed']) < (
        reference['failures'], -reference['passed']) or (
        result['failures'] == reference['failures'] and result['passed'] >= reference['passed']
        and result['score_deg'] <= .98*reference['score_deg'])


def update_progress(validation, train, reference, train_reference, stale):
    validation_progress = improved(validation, reference)
    train_progress = train is not None and improved(train, train_reference)
    return (validation if validation_progress else reference,
            train if train_progress else train_reference,
            0 if validation_progress or train_progress else stale+1)


def unique_conditions(rows):
    result, seen = [], set()
    for row in rows:
        key = (tuple(row['case']), row['payload_mass'], row['tilt_deg'], row['capacity'], row['seconds'])
        if key not in seen:
            result.append(row)
            seen.add(key)
    return result
