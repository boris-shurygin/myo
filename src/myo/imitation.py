"""Command demonstrations and corrective actor rollouts for the D1 baseline.

The heuristic supplies labels only during training. Runtime remains a single
neural actor, with all excitation passing through the ordinary muscle dynamics.
"""
from __future__ import annotations

import numpy as np

TRAINING_TARGET_DEGREES = (15.0, 158.0)
TRAINING_STATE_UPPER_DEGREES = 160.0
ZERO_LABEL_TOLERANCE = 1e-5
ZERO_RAW_MEAN_TARGET = -0.03


def reset_training_capacity_profile(env, flexor_capacity: float, extensor_capacity: float) -> None:
    """Initialize asymmetric fatigue at t=0; never modify a running episode.

    Scalar environment resets cannot represent fatigued flexors with fresh
    extensors after a long hold. This data-collection initialization rebuilds
    muscle equilibrium and repeated initial sensor history consistently.
    """
    base = env.unwrapped
    if not base._has_reset or base.time != 0.0 or base._step_count != 0:
        raise RuntimeError("Training capacity profiles are only allowed immediately after reset.")
    base._muscle_outputs = [
        muscle.reset(float(length), activation=base.config.initial_activation,
                     capacity=extensor_capacity if name.lower().startswith("tri") else flexor_capacity)
        for name, muscle, length in zip(base.muscle_names, base.muscles, base.muscle_lengths())
    ]
    base._sensor_history.clear()
    if base.config.task_mode != "reference":
        for _ in range(base.config.history_steps):
            base._sensor_history.append(base._sensor_values())


def _observation(env):
    observation = env.unwrapped.observe()
    return env.observation(observation) if hasattr(env, "predictor") else observation


def _collect(model, env, steps, seed, teacher_probability, episode_offset=0, *, focused=False,
             curriculum="endpoints", vertical_teacher_noise=0.012):
    base = env.unwrapped
    transitions = curriculum == "endurance_transitions"
    reachable = curriculum in ("endurance_reachable", "endurance_transitions")
    random = np.random.default_rng(seed)
    observations, labels, episodes = [], [], []
    episode = 0
    stopped_for_coverage = 0
    physical_failures = 0
    warmup_steps = warmup_failures = 0
    family_transitions = {}
    while len(observations) < steps:
        # Different complete episodes cover long holds, both extremes, release
        # and command reversal. Future events never enter the actor's input.
        kind = episode % 7
        family = kind
        low_target, high_target = TRAINING_TARGET_DEGREES
        category = random.choice(3)
        target = float(random.uniform(140, high_target) if category == 0
                       else random.uniform(low_target, 35) if category == 1
                       else random.uniform(35, 140))
        if episode % 7 == 0:
            target = float(random.choice([75.0, 120.0, 140.0, 150.0, 155.0]))
        mass = float(random.uniform(0.5, 2.0))
        capacity = float(random.uniform(0.35, 1.0))
        if episode % 3 == 0:
            mass, capacity = 1.0, 1.0
        q = float(random.uniform(25, 155)) if episode % 3 == 1 else 75.0
        duration = (24.0, 16.0, 20.0, 10.0, 20.0, 18.0, 18.0)[kind]
        events = [(0.0, target, 1.0)]
        if kind == 1:
            events = [(0.0, 75.0, 1.0), (1.0, target, 1.0),
                      (9.0, float(random.uniform(low_target, high_target)), 1.0)]
        elif kind == 2:
            # Include settled passive rest, not just transient braking. Release
            # keeps the target in the input but removes its tracking objective.
            # The imported paths are not calibrated for passive falling beyond
            # the upper vertical pose; that remains a separate diagnostic.
            target = float(random.uniform(25, 130))
            q = min(q, 130.0)
            events = [(0.0, target, 1.0), (2.5, target, 0.0)]
        elif kind == 3:
            trigger = float(random.choice([0.0, 0.1, 0.4, 0.7]))
            target = min(target, 130.0)
            q = min(q, 130.0)
            events = [(0.0, 75.0, 1.0), (1.0, target, trigger),
                      (6.0, float(random.uniform(low_target, high_target)), 1.0)]
        elif kind == 4:
            q = float(random.uniform(15, 30))
            events = [(0.0, low_target, 0.0)]
        elif kind == 5:
            events = [(0.0, low_target, 1.0), (10.0, low_target, 0.0)]
        elif kind == 6:
            q = float(random.uniform(120, 140))
            events = [(0.0, 155.0, 1.0), (9.0, 130.0, 1.0)]
        velocity = float(random.uniform(-0.25, 0.25))
        profile = None
        warmup_goal = None
        passive_rest = False
        if focused:
            # Correct the observed endpoint/weak-trigger regressions while
            # retaining nominal holds and genuinely settled release states.
            focus_kind = episode % 8
            family = focus_kind
            mass, capacity, q, velocity, duration = 1.0, 1.0, 75.0, 0.0, 20.0
            target = (155.0, 15.0, 150.0, 105.0, 75.0,
                      float(random.choice([75.0, 120.0, 125.0])), 155.0, 15.0)[focus_kind]
            trigger = 0.1 if focus_kind == 3 else 0.0 if focus_kind in (4, 7) else 1.0
            events = [(0.0, 75.0, 1.0), (2.0, target, trigger)]
            if focus_kind in (0, 5):
                duration = 24.0
            if focus_kind == 5:
                capacity = float(random.choice([0.75, 1.0]))
            if focus_kind == 2:
                q = 140.0
                events[0] = (0.0, q, 1.0)
            elif focus_kind == 6:
                q = float(random.uniform(148, 157))
                mass, capacity = float(random.uniform(0.8, 1.2)), float(random.uniform(0.7, 1.0))
                events = [(0.0, target, 1.0)]
            elif focus_kind == 7:
                q = float(random.uniform(15, 24))
                events = [(0.0, target, 0.0)]
                passive_rest = True
        if focused and curriculum in ("endurance", "endurance_vertical", "endurance_reachable", "endurance_transitions"):
            family_count = 32 if transitions else 26 if reachable else 24 if curriculum == "endurance_vertical" else 16
            family = episode % family_count
            mass, capacity, q, velocity, duration = 1.0, 1.0, 75.0, 0.0, 12.0
            profile = (1.0, 1.0)
            passive_rest = family in (9, 10)
            if family >= 30:
                q = float(random.uniform(17, 20))
                profile = (float(random.uniform(0.90, 0.98)), 1.0)
                events = [(0.0, 125.0 if family == 30 else 155.0, 0.0)]
                passive_rest = True
            elif family >= 26:
                if family == 26:
                    events = [(0.0, 75.0, 1.0), (2.0, 45.0, 1.0)]
                else:
                    warmup_goal = (105.0, 155.0, 145.0)[family - 27]
                    events = [(0.0, (45.0, 145.0, 60.0)[family - 27], 1.0)]
            elif family >= 24:
                profile = (1.0 if family == 24 else 0.8, 1.0)
                events = [(0.0, 75.0, 1.0), (2.0, 75.0, 0.0)]
            elif family >= 16:
                if family < 22:
                    index = family - 16
                    q = float((148, 151, 153, 154.5, 156, 157)[index] + random.uniform(-0.35, 0.35))
                    target = (150, 154, 155, 155, 155, 155)[index]
                    profile = ((1, 1), (1, 0.85), (0.9, 0.9), (0.85, 0.85),
                               (1, 1), (1, 0.85))[index]
                    mass = float(random.uniform(0.95, 1.05))
                    velocity = float(random.uniform(-0.15, 0.15))
                    events = [(0.0, target, 1.0)]
                    if reachable:
                        q, velocity = 75.0, 0.0
                        warmup_goal = (148, 151, 153, 154.5, 155, 155)[index]
                        if index == 5:
                            profile = (1.0, 0.95)
                else:
                    q = float(random.uniform(17, 20))
                    events = [(0.0, 125.0 if family == 22 else 155.0, 0.0)]
                    passive_rest = True
            elif family <= 6:
                target, flexor_capacity = ((75, 1.0), (75, 0.75), (120, 1.0),
                                           (120, 0.75), (125, 1.0), (125, 0.75),
                                           (125, 0.85))[family]
                profile = (flexor_capacity, 1.0)
                # Both starting from the hold pose and reaching it matter.
                q = float(target) if episode // family_count % 2 == 0 else 75.0
                events = [(0.0, target, 1.0)]
            elif family in (7, 8):
                q = 125.0
                profile = (0.75 if family == 7 else 0.85, 1.0)
                events = [(0.0, 125.0, 1.0), (3.0, 125.0, 0.0)]
            elif family in (9, 10):
                q = float(random.uniform(17, 20))
                profile = (0.75 if family == 9 else 0.85, 1.0)
                events = [(0.0, 125.0 if family == 9 else 155.0, 0.0)]
            elif family in (11, 12):
                profile = (1.0, 1.0 if family == 11 else 0.85)
                events = [(0.0, 75.0, 1.0), (2.0, 155.0, 1.0)]
                if reachable:
                    profile = (1.0, 1.0 if family == 11 else 0.95)
                    duration = 30.0
            elif family == 13:
                profile = (0.85, 1.0)
                events = [(0.0, 75.0, 1.0), (2.0, 45.0, 1.0)]
            elif family == 14:
                events = [(0.0, 75.0, 1.0), (2.0, 105.0, 0.1)]
            else:
                events = [(0.0, 75.0, 1.0), (2.0, 15.0, 1.0), (8.0, 125.0, 0.0)]
            if family in (7, 8, 9, 10):
                # With trigger released, the remembered angle must not turn
                # back into a tracking objective. Future events stay hidden.
                events.extend((time, float(random.choice([15, 45, 75, 105, 125, 155])), 0.0)
                              for time in (6.0, 9.0))
        env.reset(seed=seed + episode, options={"q": float(np.deg2rad(q)),
                  "qvel": velocity,
                  **({"target_angle": float(np.deg2rad(q))} if focused else {}),
                  "payload_mass": mass, "capacity": capacity})
        if profile is not None:
            reset_training_capacity_profile(env, *profile)
        if warmup_goal is not None:
            # Reach the state through ordinary delayed muscles. These teacher
            # steps are initialization exposure, not supervised examples; their
            # complete sensor/action/command history remains for the handoff.
            warmup_failed = False
            for _ in range(int(round(8.0 / base.control_dt))):
                base.set_command(float(np.deg2rad(warmup_goal)), 1.0)
                _, _, terminated, truncated, _ = env.step(base.baseline_action())
                warmup_steps += 1
                if terminated or base.data.qpos[base.qpos_index] > np.deg2rad(TRAINING_STATE_UPPER_DEGREES):
                    physical_failures += int(terminated)
                    warmup_failed = True
                    break
                if truncated:
                    base._episode_done = False
            if warmup_failed:
                warmup_failures += 1
                episode += 1
                continue
        for step in range(int(round(duration / base.control_dt))):
            time = step * base.control_dt
            event = max((event for event in events if event[0] <= time + 1e-10), key=lambda event: event[0])
            base.set_command(float(np.deg2rad(event[1])), event[2])
            observation = _observation(env)
            teacher = base.baseline_action()
            observations.append(observation.copy())
            labels.append(teacher.copy())
            episodes.append(episode_offset + episode)
            family_transitions[str(family)] = family_transitions.get(str(family), 0) + 1
            if random.random() < teacher_probability:
                # Small excitation perturbations expose recovery states; the
                # unperturbed teacher remains the supervised target.
                # Keep some exact-zero rest episodes: constant positive noise
                # would prevent the demonstrated passive state from settling.
                noise_scale = 0.0 if passive_rest or (not focused and kind == 4) else 0.012
                if curriculum in ("endurance_vertical", "endurance_reachable", "endurance_transitions") and base.data.qpos[base.qpos_index] > np.deg2rad(140):
                    # Clipped noise on inactive strong flexors can overwhelm
                    # the much weaker extensor moment beyond vertical.
                    noise_scale = vertical_teacher_noise
                action = np.clip(teacher + random.normal(0.0, noise_scale, teacher.shape), 0.0, 1.0)
            else:
                action = model.predict(observation, deterministic=True)[0]
            _, _, terminated, truncated, _ = env.step(action)
            if terminated:
                physical_failures += 1
                break
            if base.data.qpos[base.qpos_index] > np.deg2rad(TRAINING_STATE_UPPER_DEGREES):
                # End only this training episode when it leaves the region in
                # which imported paths still have the intended muscle roles.
                # No torque, pose clamp, runtime limit or rescue is introduced.
                stopped_for_coverage += 1
                break
            if truncated:
                # Same continuous-state semantics as the interactive demo.
                base._episode_done = False
            if len(observations) >= steps:
                break
        episode += 1
    return (np.asarray(observations, dtype=np.float32), np.asarray(labels, dtype=np.float32),
            np.asarray(episodes)), episode, {"physical_failures": physical_failures,
                "left_training_angle_coverage": stopped_for_coverage, "warmup_control_steps": warmup_steps,
                "warmup_control_dt_s": base.control_dt,
                "warmup_failed_episodes": warmup_failures, "family_transitions": family_transitions}


def _closed_loop_validation(model, env, seed, *, endurance=False, vertical_retention=False,
                            required_behavior_priority=False, transition_retention=False):
    """Select refinement snapshots by held-out behavior as well as action fit."""
    from .evaluate import CommandEvent, Scenario, run_scenario

    diagnostics = (
        Scenario("vertical", 12.0, (CommandEvent(0, 75, 1), CommandEvent(2, 155, 1)),
                 1.05, capacity=0.95, initial_angle_deg=74),
        Scenario("extension", 12.0, (CommandEvent(0, 75, 1), CommandEvent(2, 15, 1)), 0.95),
        Scenario("release", 12.0, (CommandEvent(0, 75, 1), CommandEvent(2, 75, 0)), 1.1),
        Scenario("hold", 12.0, (CommandEvent(0, 75, 1),), 1.0, capacity=0.8),
        Scenario("weak_trigger", 12.0, (CommandEvent(0, 75, 1), CommandEvent(2, 105, 0.1)), 1.0),
    )
    profiles = {}
    if endurance:
        diagnostics += (
            Scenario("fatigued_hold125", 20.0, (CommandEvent(0, 125, 1),), 1.0,
                     initial_angle_deg=125),
            Scenario("fatigued_release125", 20.0,
                     (CommandEvent(0, 125, 1), CommandEvent(4, 125, 0)), 1.0,
                     initial_angle_deg=125),
            Scenario("rest_old_target125", 12.0, (CommandEvent(0, 125, 0),), 1.0,
                     initial_angle_deg=18),
        )
        profiles = {"hold": (0.8, 1.0), "fatigued_hold125": (0.8, 1.0),
                    "fatigued_release125": (0.8, 1.0), "rest_old_target125": (0.8, 1.0)}
    if vertical_retention:
        diagnostics += (Scenario("nominal_vertical", 30.0,
            (CommandEvent(0, 75, 1), CommandEvent(2, 155, 1)), 1.0, initial_angle_deg=76),)
    if transition_retention:
        diagnostics += (
            Scenario("lower75_to45", 12.0,
                     (CommandEvent(0, 75, 1), CommandEvent(2, 45, 1)), 1.0),
            Scenario("lower105_to45", 16.0,
                     (CommandEvent(0, 75, 1), CommandEvent(2, 105, 1), CommandEvent(6, 45, 1)), 1.0),
            Scenario("vertical_return", 30.0, (CommandEvent(0, 75, 1), CommandEvent(2, 155, 1),
                     CommandEvent(10, 145, 1), CommandEvent(18, 60, 1)), 1.0),
            Scenario("rest_recovered125", 12.0, (CommandEvent(0, 125, 0),), 1.0,
                     initial_angle_deg=18),
            Scenario("rest_recovered155", 12.0, (CommandEvent(0, 155, 0),), 1.0,
                     initial_angle_deg=18),
        )
        profiles.update({"rest_recovered125": (0.95, 1.0), "rest_recovered155": (0.95, 1.0)})

    import gymnasium as gym

    class InitialCapacityProfile(gym.ObservationWrapper):
        def __init__(self, wrapped, profile):
            super().__init__(wrapped)
            self.profile = profile

        def reset(self, **kwargs):
            self.env.reset(**kwargs)
            reset_training_capacity_profile(self.env, *self.profile)
            return _observation(self.env), self.env.unwrapped.telemetry()

        def observation(self, observation):
            return self.env.observation(observation) if hasattr(self.env, "predictor") else observation

        def step(self, action):
            # The wrapped environment already applies its optional predictor.
            # observation() is only needed for run_scenario's command refresh.
            return self.env.step(action)

    score, results = 0.0, []
    for index, scenario in enumerate(diagnostics):
        validation_env = InitialCapacityProfile(env, profiles[scenario.name]) if scenario.name in profiles else env
        result = run_scenario(validation_env, lambda obs: model.predict(obs, deterministic=True)[0],
                              scenario, seed + index, continue_after_time_limit=True)
        tail = result["hold_tail_last_5s"]
        error = tail.get("angle_rmse_deg")
        penalty = (0.2 if scenario.name == "weak_trigger" else 1.0) * (error or 0.0)
        if scenario.name in ("release", "fatigued_release125", "rest_old_target125", "rest_recovered125", "rest_recovered155"):
            penalty += 100 * tail.get("mean_excitation", 1.0) + max(0.0, result["final_angle_deg"] - 25)
            if endurance and tail.get("mean_excitation", 1.0) > (0.001 if required_behavior_priority else 0.005):
                penalty += 5000.0 if required_behavior_priority else 500.0
        if scenario.name == "fatigued_hold125" and (error is None or error > 5.0):
            penalty += 5000.0 if required_behavior_priority else 500.0
        if required_behavior_priority and scenario.name == "extension" and tail.get("mean_excitation", 1.0) > 0.1:
            penalty += 5000.0
        if vertical_retention and scenario.name == "weak_trigger" and (error is None or error > 15.0):
            penalty += 500.0
        if scenario.name == "nominal_vertical" and (error is None or error > 3.0):
            penalty += 3000.0 if required_behavior_priority else 1000.0
        if scenario.name in ("lower75_to45", "lower105_to45", "vertical_return") and (error is None or error > 5.0):
            penalty += 5000.0
        waypoint_errors = {}
        if scenario.name == "vertical_return":
            # Reaching the final low goal must not conceal failure to cross
            # vertical and reverse on the earlier commands.
            for start, end, goal in ((8, 10, 155), (16, 18, 145)):
                samples = [abs(row["angle_deg"] - goal) for row in result["trace_1hz"]
                           if start - 1e-8 <= row["time_s"] <= end + 1e-8]
                waypoint_error = float(np.mean(samples)) if samples else 180.0
                waypoint_errors[str(goal)] = waypoint_error
                penalty += waypoint_error + (5000.0 if waypoint_error > 5.0 else 0.0)
        if result["overall"].get("sampled_max_angle_deg", 180) > TRAINING_STATE_UPPER_DEGREES:
            penalty += 8000.0 if required_behavior_priority and scenario.name == "nominal_vertical" else 2000.0
        if result["physical_failure"] or result["numerical_failure"] or not result["completed_duration"]:
            penalty += 1000.0
        score += penalty
        results.append({"scenario": scenario.name, "final_angle_deg": result["final_angle_deg"],
                        "initial_capacity": result["initial_state"]["capacity"],
                        "tail_angle_rmse_deg": error,
                        "tail_mean_excitation": tail.get("mean_excitation"),
                        "waypoint_errors_deg": waypoint_errors, "penalty": penalty})
    return score, results


def train_command_actor(model, env, steps: int, rounds: int, seed: int, epochs: int = 180,
                        *, resume: bool = False, curriculum: str = "endpoints",
                        vertical_teacher_noise: float = 0.012,
                        inactive_flexor_margin: float = 0.0, weak_trigger_weight: float = 2.0):
    """Aggregate causal corrections on the actor's own visited states (DAgger).

    Normalization is fit on the first training collection then frozen. Validation
    uses entirely separate episodes; its data never enters optimizer batches.
    """
    import torch

    if env.unwrapped.config.task_mode == "reference":
        raise ValueError("Command imitation requires interactive/curriculum observations.")
    if rounds < 1 or steps < rounds * 1000:
        raise ValueError("Use at least 1000 command demonstration transitions per round.")
    random = np.random.default_rng(seed)
    train_x, train_y, groups = [], [], []
    counts, reports = 0, []
    per_round = steps // rounds
    # A separate teacher collection checks action error on unseen episodes.
    # Covers all seven episode families at the default 50 Hz control rate.
    endurance = curriculum in ("endurance", "endurance_vertical", "endurance_reachable", "endurance_transitions")
    vertical_retention = curriculum in ("endurance_vertical", "endurance_reachable", "endurance_transitions")
    reachable = curriculum in ("endurance_reachable", "endurance_transitions")
    transitions = curriculum == "endurance_transitions"
    validation, validation_episodes, validation_stops = _collect(
        model, env, 22_000 if transitions else 20_000 if reachable else 15_000 if vertical_retention else 10_000 if resume else 6300,
        seed + 100_000, 1.0, focused=resume,
        curriculum=curriculum, vertical_teacher_noise=vertical_teacher_noise)
    validation_x = torch.from_numpy(validation[0])
    validation_y = torch.from_numpy(validation[1])
    print({"validation_transitions": len(validation_x), "validation_episodes": validation_episodes}, flush=True)
    incumbent = None
    parent_validation = None
    if resume:
        model.policy.set_training_mode(False)
        with torch.no_grad():
            mean = model.policy.get_distribution(validation_x).distribution.mean
            parent_loss = float(torch.mean((mean.clamp(0.0, 1.0) - validation_y) ** 2))
        parent_score, parent_behavior = _closed_loop_validation(model, env, seed + 200_000,
            endurance=endurance, vertical_retention=vertical_retention,
            required_behavior_priority=reachable, transition_retention=transitions)
        parent_state = {key: value.detach().clone() for key, value in model.policy.state_dict().items()}
        incumbent = (parent_score, 0, parent_loss, parent_state, 0)
        parent_validation = {"score": parent_score, "scenarios": parent_behavior,
                             "validation_excitation_rmse": parent_loss ** 0.5}
        print({"parent_closed_loop_validation": parent_validation}, flush=True)
    for round_index in range(rounds):
        budget = per_round if round_index < rounds - 1 else steps - per_round * round_index
        beta = (0.7 if round_index == 0 else 0.0) if resume else (1.0 if round_index == 0 else 0.3 if round_index == 1 else 0.0)
        if vertical_retention and round_index == 0:
            beta = 1.0
        collection, episodes, collection_stops = _collect(
            model, env, budget, seed + 10_000 * round_index, beta, counts, focused=resume,
            curriculum=curriculum, vertical_teacher_noise=vertical_teacher_noise)
        counts += episodes
        train_x.append(collection[0])
        train_y.append(collection[1])
        groups.append(collection[2])
        x = torch.from_numpy(np.concatenate(train_x))
        y = torch.from_numpy(np.concatenate(train_y))
        # PPO already clips Gaussian means to the physical [0,1] action range.
        # A small negative target for zero labels lets this same actor learn
        # exact relaxation instead of hovering at a small positive excitation.
        # It is only a training target, never a runtime mask or trigger gate.
        zero_targets = (torch.all(y <= ZERO_LABEL_TOLERANCE, dim=1, keepdim=True)
                        if resume else y <= ZERO_LABEL_TOLERANCE)
        # During refinement, inactive antagonists retain zero targets while
        # another muscle is active. A negative margin there would create a
        # deadband at the delicate flexor/extensor switch near vertical.
        raw_targets = torch.where(zero_targets, ZERO_RAW_MEAN_TARGET, y)
        if inactive_flexor_margin > 0:
            flexors = [i for i, name in enumerate(env.unwrapped.muscle_names) if not name.lower().startswith("tri")]
            extensors = [i for i, name in enumerate(env.unwrapped.muscle_names) if name.lower().startswith("tri")]
            extension_level = y[:, extensors].amax(dim=1, keepdim=True)
            # Exact inactive flexors matter when their torque capacity greatly
            # exceeds the active extensors'. Fade the margin to zero at the
            # agonist switch; leave extensor onset targets unchanged.
            margin = inactive_flexor_margin * torch.clamp(extension_level / 0.05, max=1.0)
            inactive = (y[:, flexors] <= ZERO_LABEL_TOLERANCE) & (extension_level > ZERO_LABEL_TOLERANCE)
            raw_targets[:, flexors] = torch.where(inactive, -margin, raw_targets[:, flexors])
        weights = torch.ones(len(x))
        if vertical_retention:
            angles = x[:, env.unwrapped.observation_names.index("angle_centered_rad")] + env.unwrapped.config.reference_center
            triggers = x[:, env.unwrapped.observation_names.index("trigger")]
            weights[angles > np.deg2rad(140)] = 4.0
            weights[(triggers > 0) & (triggers <= 0.2)] = weak_trigger_weight
        if round_index == 0 and not resume:
            model.policy.features_extractor.set_normalization(x.mean(0), x.std(0, unbiased=False))
        optimizer = torch.optim.Adam(model.policy.parameters(), lr=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, eta_min=2e-5)
        best_loss, best_state, best_epoch = float("inf"), None, 0
        candidates = []
        print({"round": round_index + 1, "train_transitions": len(x), "teacher_probability": beta}, flush=True)
        model.policy.set_training_mode(True)
        for epoch in range(epochs):
            for indices in np.array_split(random.permutation(len(x)), max(1, len(x) // 512)):
                mean = model.policy.get_distribution(x[indices]).distribution.mean
                loss = torch.mean(weights[indices, None] * (mean - raw_targets[indices]) ** 2) / weights[indices].mean()
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.policy.parameters(), 1.0)
                optimizer.step()
            scheduler.step()
            with torch.no_grad():
                mean = model.policy.get_distribution(validation_x).distribution.mean
                validation_loss = float(torch.mean((mean.clamp(0.0, 1.0) - validation_y) ** 2))
            if validation_loss < best_loss:
                best_loss, best_epoch = validation_loss, epoch + 1
                best_state = {key: value.detach().clone() for key, value in model.policy.state_dict().items()}
            if resume and (epoch + 1) % (40 if endurance else 30) == 0:
                candidates.append((epoch + 1, validation_loss,
                                   {key: value.detach().clone() for key, value in model.policy.state_dict().items()}))
        closed_loop_reports = []
        selected_from_round = round_index + 1
        retained_previous_round = False
        if resume:
            if best_epoch not in [candidate[0] for candidate in candidates]:
                candidates.append((best_epoch, best_loss, best_state))
            best_score = float("inf")
            model.policy.set_training_mode(False)
            for candidate_epoch, candidate_loss, candidate_state in candidates:
                model.policy.load_state_dict(candidate_state)
                score, behavior = _closed_loop_validation(model, env, seed + 200_000,
                    endurance=endurance, vertical_retention=vertical_retention,
                    required_behavior_priority=reachable, transition_retention=transitions)
                closed_loop_reports.append({"epoch": candidate_epoch, "score": score, "scenarios": behavior})
                print({"round": round_index + 1, "closed_loop_epoch": candidate_epoch,
                       "score": score, "scenarios": behavior}, flush=True)
                if score < best_score:
                    best_score, best_epoch, best_loss, best_state = score, candidate_epoch, candidate_loss, candidate_state
            # DAgger can add useful states while its next fit makes control
            # worse. Compare rounds on the same held-out scenarios and retain
            # the previous winner instead of silently replacing it.
            if incumbent is not None and incumbent[0] <= best_score:
                best_score, best_epoch, best_loss, best_state, selected_from_round = incumbent
                retained_previous_round = True
            else:
                incumbent = (best_score, best_epoch, best_loss, best_state, selected_from_round)
        model.policy.load_state_dict(best_state)
        model.policy.set_training_mode(False)
        report = {"round": round_index + 1, "teacher_probability": beta, "training_transitions": len(x),
                  "training_episodes": counts, "validation_excitation_rmse": best_loss ** 0.5,
                  "collection_episode_stops": collection_stops,
                  "all_zero_label_fraction": float(torch.mean(torch.all(y <= ZERO_LABEL_TOLERANCE, dim=1).float())),
                  "closed_loop_validation": closed_loop_reports,
                  "selected_epoch": best_epoch,
                  "selected_from_round": selected_from_round,
                  "selected_closed_loop_score": best_score if resume else None,
                  "retained_previous_round": retained_previous_round}
        reports.append(report)
        print(report, flush=True)
    env.reset(seed=seed)
    return {"method": "causal_dataset_aggregation", "teacher": "diagnostic_heuristic",
            "demonstration_steps": steps, "seed": seed, "rounds": reports,
            "train_episodes": counts, "validation_episodes": validation_episodes,
            "validation_steps": len(validation_x), "validation_seed": seed + 100_000,
            "validation_episode_stops": validation_stops,
            "epochs_per_round": epochs,
            "normalization": ("preserved from parent checkpoint" if resume else "first training collection only; frozen; std floor 0.05"),
            "focused_endpoint_refinement": resume,
            "imitation_curriculum": curriculum if resume else "broad_commands",
            "selection": ("held-out closed-loop behavior score over parent, epoch snapshots and prior round winners"
                          if resume else "held-out episode action MSE, followed by separate closed-loop evaluation"),
            "parent_closed_loop_validation": parent_validation,
            "closed_loop_validation_seed": seed + 200_000 if resume else None,
            "command_range_deg": [15, 155] if resume else list(TRAINING_TARGET_DEGREES),
            "episode_duration_s": [12, 30] if reachable else [12] if endurance else [20, 24] if resume else [10, 16, 18, 20, 24],
            "near_vertical_warmup": ({"duration_s": 8, "controller": "diagnostic_heuristic",
                "reset_angle_deg": 75, "goals_deg": [148, 151, 153, 154.5, 155, 155] + ([105, 155, 145] if transitions else []),
                "counts_toward_label_budget": False, "history": "ordinary continuous physical history; no state reset at handoff"}
                if reachable else None),
            "required_behavior_priority": reachable,
            "transition_retention": transitions,
            "supervised_sample_weights": ({"angle_above_140_deg": 4, "trigger_in_0_to_02": weak_trigger_weight,
                                           "other": 1, "batch_normalized": True} if vertical_retention else "uniform"),
            "teacher_excitation_noise_std": {"usual": 0.012, "passive_rest": 0.0,
                "angle_above_140_deg": vertical_teacher_noise if vertical_retention else 0.012},
            "zero_excitation_labels": {"tolerance": ZERO_LABEL_TOLERANCE,
                                       "raw_gaussian_mean_target": ZERO_RAW_MEAN_TARGET,
                                       "negative_margin_scope": ("all-zero teacher rows plus inactive flexors with active extensors"
                                           if inactive_flexor_margin > 0 and resume else
                                           "all-zero teacher rows only" if resume else "zero components"),
                                       "inactive_flexor_margin": inactive_flexor_margin,
                                       "inactive_flexor_margin_scale": "min(teacher_max_extensor / 0.05, 1)",
                                       "runtime": "existing PPO action clipping only; no trigger mask"},
            "initial_capacity_range": [0.75, 1.0] if endurance else [0.7, 1.0] if resume else [0.35, 1.0],
            "initial_capacity_profiles": (([[1, 1], [0.75, 1], [0.85, 1], [1, 0.85]] +
                                           ([[0.9, 0.9], [0.85, 0.85]] if vertical_retention else []) +
                                           ([[1, 0.95], [0.8, 1]] if reachable else []))
                                          if endurance else "uniform at reset"),
            "capacity_profile_order": ["flexor_capacity", "extensor_capacity"],
            "initial_capacity_profile_distributions": ([{"families": [30, 31],
                "flexor_capacity_uniform": [0.90, 0.98], "extensor_capacity": 1.0}] if transitions else []),
            "initialization": ("training-only muscle reset and matching initial sensor history before the first step"
                               if endurance else "ordinary environment reset"),
            "payload_mass_range_kg": [0.95, 1.05] if vertical_retention else [1.0, 1.0] if endurance else [0.8, 1.2] if resume else [0.5, 2.0],
            "release_remembered_target_deg": [15, 45, 75, 105, 125, 155] if endurance else None,
            "systematic_families": ((["hold75_fresh", "hold75_flexors_075", "hold120_fresh", "hold120_flexors_075",
                "hold125_fresh", "hold125_flexors_075", "hold125_flexors_085", "release125_flexors_075",
                "release125_flexors_085", "rest_requested125_flexors_075", "rest_requested155_flexors_085",
                "hold155_fresh", "hold155_extensors_095" if reachable else "hold155_extensors_085", "lower45_flexors_085", "weak_trigger_raise",
                "lower15_then_release"] + (["near148_goal150_fresh", "near151_goal154_extensors085",
                "near153_goal155_uniform090", "near1545_goal155_uniform085",
                "prepared155_goal155_fresh" if reachable else "near156_goal155_fresh",
                "prepared155_goal155_extensors095" if reachable else "near157_goal155_extensors085",
                "rest_requested125_fresh", "rest_requested155_fresh"]
                if vertical_retention else []) + (["release75_fresh", "release75_flexors080"] if reachable else []) +
                (["lower75_to45_fresh", "prepared105_to45_fresh", "prepared155_to145_fresh", "prepared145_to60_fresh",
                  "rest_requested125_recovered_flexors", "rest_requested155_recovered_flexors"]
                 if transitions else [])) if endurance else None),
            "release_training_angle_range_deg": [15, 130],
            "training_state_upper_deg": TRAINING_STATE_UPPER_DEGREES,
            "training_coverage_stop": "Ends only data collection episodes above 160 deg; no runtime limit or state change.",
            "uncovered_conditions": ["Release beyond the upper vertical pose; imported muscle paths are not calibrated there."],
            "limitation": "Imitation, not independent PPO optimization; no runtime teacher or force override."}
