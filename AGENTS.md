# Working on myo

This repository is an experimental character-control project. Read README.md and
docs/roadmap.md before changing its scope. User instructions take precedence.
For the agreed next arm demo, also read docs/first-stage.md. It distinguishes
planned control/history/protection from the implemented V0; the visual milestones
are in integrations/unreal/README.md. Do not report a plan as working code.
Model sources and the laptop-first workflow are in docs/model-sources-and-workflow.md.

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

## Next arm demo decisions

- One muscle-control policy learns both user-command tracking and protective
  braking. Do not add a separate safety network by default. Trigger release is
  a request to relax, not a hard zero multiplier disabling protection.
- Keep motor delay and activation dynamics on protective commands too. Do not
  add an instant joint torque, reset velocity, or change the payload to rescue
  the policy. Learned protection is not a formal safety guarantee.
- Start with short sensor/action/command history and no separate predictor.
  Evaluate the predictor later within D1 against that baseline. Visual work
  need not wait for the predictor or final policy convergence.
- Preserve proportionate scope: an articulated 1-DOF arm can have detailed 3D
  visuals. UE is the presentation/input layer; Flesh is for tissue deformation,
  not a prerequisite for compliant tendons in headless physics.

## Model sources and development machines

- Use MyoArm from MyoSim for skeletal geometry, muscle paths and initial force
  parameters; reuse its bone meshes for rendering. Preserve series elasticity
  and delay when replacing native actuators; never apply both force paths.
- Use official BodyParts3D for muscle surfaces, aligned to the MyoArm skeleton.
  Do not substitute Z-Anatomy CC BY-SA assets into UE without suitable permission.
  Record per-asset provenance and licensing before importing or publishing.
  Meshes are not ready-made active Flesh assets; integration is still planned.
- Work locally on the Ubuntu laptop first. Move to the Windows RTX 3060 desktop
  when UE is needed or measured laptop performance becomes limiting. Do not add
  remote-desktop/streaming infrastructure or require WSL by default.
- Cloud GPU training is a later option after measuring a desktop bottleneck and
  verifying backend equivalence. Do not provision paid resources without a new
  explicit request. Keep training headless and portable between machines.

## Workflow

- Python code and identifiers are English; project documentation is Russian.
- Bootstrap the tested Python 3.12 environment with `bash scripts/setup.sh`.
  Do not assume the system `python3` meets the project's version requirements.
- Use `python3` explicitly, including in the project virtual environment.
  Run `python3 -m pytest -q` for changes to
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

## Proportionate testing

- This is a small hobby prototype, not a general-purpose public framework.
  Keep a compact suite of behavioral checks for the invariants above and
  regressions that have actually occurred. See tests/README.md for the scope.
- Add a test for a concrete failure mode, not automatically for every function,
  helper, branch, or rejected input. Reuse an existing scenario when appropriate.
- Avoid exhaustive invalid-input matrices and fragile assertions about private
  implementation details unless they protect a demonstrated, important failure.
- Do not merge unrelated tests or hide parameter cases merely to lower the count.
  Neither a test-count target nor a coverage percentage is a project goal.
- Passing tests do not establish anatomical accuracy, natural movement, or a
  usable GUI. Check relevant rendered poses and actual policy behavior as well.
