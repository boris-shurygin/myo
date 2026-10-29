# Working on myo

This repository is an experimental character-control project. Read README.md and
docs/roadmap.md before changing its scope. User instructions take precedence.

## Invariants

- MuJoCo owns the physical state. Visual deformation does not silently feed back
  into physics. Any coupling needs an explicit design and test.
- Keep excitation commands, delayed excitation, activation, tendon force, tendon
  strain and remaining capacity separate in code and telemetry.
- An activation time constant is not a transport delay. Preserve both.
- Tendon compliance requires an independent series-elastic state. Adding a
  parallel spring to a rigid-tendon muscle is not an equivalent implementation.
- Name pain, fatigue and joint-load approximations as proxies. Never describe the
  current numbers as medical thresholds, injury probabilities or a human model.
- Predictor input is causal. Future simulated outcomes belong only in labels.
  Split predictor validation by episode, not by neighboring time samples.
- Store timestep, observation order, action mapping, model configuration and
  predictor identity with learned policies. Fail on incompatible checkpoints.
- Keep training headless and local by default. Do not enable cloud tracking or
  download gated anatomy/motion/checkpoints without an explicit user request.

## Workflow

- Python code and identifiers are English; project documentation is Russian.
- Use the project virtual environment. Run `python -m pytest -q` for changes to
  dynamics, timing, observations, reward or serialization.
- Report actual validation commands and results in docs/validation.md. A short
  training smoke test demonstrates plumbing, not a successful motor policy.
- Preserve SI units internally. Convert coordinates/units at a named renderer
  boundary, not in the muscle or controller implementation.
- Do not commit generated rollouts, weights, access tokens, third-party motion
  datasets or anatomical assets. Check their individual licenses first.
- Do not assign a project license without an explicit owner decision.
- Keep tested versions explicit. Do not claim CPU/GPU backend equivalence or a
  functioning Unreal integration until the relevant checks have actually run.

