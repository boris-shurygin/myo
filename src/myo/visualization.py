"""Presentation-only diagnostic geometry, independent of the arm dynamics.

MyoArm uses the imported bone meshes and the same native spatial tendon paths
that determine mechanical lengths and moment arms. The older toy model retains
its independent schematic. Procedural tissue envelopes follow the separate
fiber/tendon lengths; they are diagnostic volume proxies, not anatomical meshes.

All geometry is appended to an existing ``MjvScene``. Model and simulation data
are read-only; refreshed kinematics are computed on a private copy when needed.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import TYPE_CHECKING

import mujoco
import numpy as np

if TYPE_CHECKING:
    from .env import PreacherCurlEnv


# A fixed perceptual scale, deliberately more distinct at low activations.
# These are absolute activation values, never per-frame minima or maxima.
# The same public stops/helper can be used by a viewer's color legend.
ACTIVATION_COLOR_STOPS = (
    (0.00, (0.28, 0.43, 0.57)),
    (0.08, (0.14, 0.52, 0.74)),
    (0.25, (0.95, 0.58, 0.10)),
    (0.60, (1.00, 0.36, 0.12)),
    (1.00, (0.90, 0.12, 0.055)),
)
TENDON_COLOR_STOPS = (
    (0.00, (0.82, 0.88, 0.87)),
    (0.15, (0.99, 0.81, 0.28)),
    (0.40, (1.00, 0.43, 0.10)),
    (1.00, (0.76, 0.06, 0.25)),
)

_BONE = (0.88, 0.86, 0.77, 1.0)
_BONE_END = (0.96, 0.94, 0.85, 1.0)
_ATTACHMENT = (1.0, 0.91, 0.63, 1.0)
_JOINT = (0.23, 0.30, 0.36, 1.0)
_JOINT_FACE = (0.70, 0.77, 0.79, 1.0)
_WEIGHT = (0.24, 0.29, 0.34, 1.0)
_HANDLE = (0.60, 0.66, 0.69, 1.0)
_FOREARM_BONES = {"radius": (-0.009, -0.014), "ulna": (0.009, 0.014)}
COLOR_MODES = ("tissue", "activation", "force", "strain")
COLOR_LABELS = {
    "tissue": "Muscle activation 0-100%\n\nTendon load 0-100% F0 (clipped)",
    "activation": "Activation 0-100%",
    "force": "Tendon force 0-100% F0 (clipped)",
    "strain": "Tendon strain 0-4% (clipped)",
}


def muscle_color_values(env: PreacherCurlEnv, color_mode: str) -> dict[str, float]:
    """Use fixed scales, never normalize against the current frame's range."""
    if color_mode not in COLOR_MODES:
        raise ValueError(f"Unknown muscle color mode: {color_mode}")
    state = env.telemetry()
    values = {}
    for index, muscle in enumerate(env.muscles):
        if color_mode in ("activation", "tissue"):
            value = muscle.activation
        elif color_mode == "force":
            value = state["tendon_force_N"][index] / muscle.params.max_isometric_force
        else:
            value = state["tendon_strain"][index] / 0.04
        values[muscle.params.name] = float(value)
    return values


@dataclass(frozen=True)
class SchematicMusclePath:
    """Four world-space cubic Bezier points, not a mechanical muscle path."""

    name: str
    control_points: np.ndarray
    belly_radius: float


@dataclass(frozen=True)
class ArmAnchors:
    """A read-only calculation from the model's fixed frames and current q."""

    shoulder: np.ndarray
    elbow: np.ndarray
    wrist: np.ndarray
    upper_rotation: np.ndarray
    forearm_rotation: np.ndarray
    forearm_origin: np.ndarray
    muscles: tuple[SchematicMusclePath, ...]


def activation_rgba(activation: float) -> np.ndarray:
    """Map absolute activation to RGBA; finite values are clipped to [0, 1]."""
    return _scale_rgba(activation, ACTIVATION_COLOR_STOPS)


def tendon_rgba(force_ratio: float) -> np.ndarray:
    """Fixed tendon load palette; one means the muscle's nominal F0, not a limit."""
    return _scale_rgba(force_ratio, TENDON_COLOR_STOPS)


def _scale_rgba(value: float, stops: tuple) -> np.ndarray:
    value = float(value)
    if not np.isfinite(value):
        raise ValueError("Activation color requires a finite value.")
    value = float(np.clip(value, 0.0, 1.0))
    levels = [stop[0] for stop in stops]
    colors = np.asarray([stop[1] for stop in stops])
    return np.asarray(
        [*(np.interp(value, levels, colors[:, channel]) for channel in range(3)), 1.0],
        dtype=np.float32,
    )


def _quaternion_matrix(quaternion: np.ndarray) -> np.ndarray:
    matrix = np.empty(9, dtype=np.float64)
    mujoco.mju_quat2Mat(matrix, np.asarray(quaternion, dtype=np.float64))
    return matrix.reshape(3, 3)


def _axis_rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    skew = np.array(((0.0, -z, y), (z, 0.0, -x), (-y, x, 0.0)))
    return np.eye(3) + np.sin(angle) * skew + (1.0 - np.cos(angle)) * (skew @ skew)


def forearm_bone_axis(name: str, length: float) -> tuple[np.ndarray, np.ndarray, float]:
    """Return one schematic shaft's endpoints/radius in the elbow-local frame.

    Both drawing and attachments use this geometry. The local frame has the
    forearm's orientation and points down the shaft along negative Z.
    """
    if not np.isfinite(length) or length <= 0.060:
        raise ValueError("The schematic forearm length must exceed 60 mm.")
    x, y = _FOREARM_BONES[name]
    return np.array((x, y, -0.030)), np.array((0.75 * x, 0.75 * y, -length + 0.030)), 0.0075


def forearm_bone_surface(name: str, length: float, toward_local: np.ndarray) -> np.ndarray:
    """Find the closest capsule surface toward an illustrative attachment point."""
    start, end, radius = forearm_bone_axis(name, length)
    target = np.asarray(toward_local, dtype=np.float64)
    if target.shape != (3,) or not np.isfinite(target).all():
        raise ValueError("The attachment reference must be a finite 3D point.")
    axis = end - start
    fraction = np.clip(np.dot(target - start, axis) / np.dot(axis, axis), 0.0, 1.0)
    nearest = start + fraction * axis
    direction = target - nearest
    distance = np.linalg.norm(direction)
    if distance < 1e-12:
        raise ValueError("The attachment reference must specify a side of the bone.")
    return nearest + radius * direction / distance


def schematic_anchors(model: mujoco.MjModel, elbow_angle: float) -> ArmAnchors:
    """Calculate V0 display anchors without changing ``model`` or ``MjData``.

    This reads the fixed upper-arm frame and rotates the forearm about its
    hinge. Using q directly avoids using potentially one-substep-old ``xpos``
    after ``mj_step2``. The attachment choices below are illustrative and
    deliberately separated so four muscle signals remain visible.
    """
    angle = float(elbow_angle)
    if not np.isfinite(angle):
        raise ValueError("The displayed elbow angle must be finite.")
    upper_id = model.body("upper_arm").id
    forearm_id = model.body("forearm").id
    payload_id = model.body("payload").id
    elbow_id = model.joint("elbow").id
    if int(model.body_parentid[upper_id]) != 0:
        raise ValueError("The V0 schematic expects a fixed upper arm attached to world.")

    shoulder = model.body_pos[upper_id].copy()
    upper_rotation = _quaternion_matrix(model.body_quat[upper_id])
    base_rotation = upper_rotation @ _quaternion_matrix(model.body_quat[forearm_id])
    base_origin = shoulder + upper_rotation @ model.body_pos[forearm_id]
    joint_position = model.jnt_pos[elbow_id]
    elbow = base_origin + base_rotation @ joint_position
    q_index = int(model.jnt_qposadr[elbow_id])
    forearm_rotation = base_rotation @ _axis_rotation(
        model.jnt_axis[elbow_id], angle - float(model.qpos0[q_index])
    )
    forearm_origin = elbow - forearm_rotation @ joint_position
    wrist = forearm_origin + forearm_rotation @ model.body_pos[payload_id]

    def upper(point):
        return shoulder + upper_rotation @ np.asarray(point, dtype=np.float64)

    def forearm(point):
        return forearm_origin + forearm_rotation @ np.asarray(point, dtype=np.float64)

    # Brachialis attaches to the displayed ulna surface. Intermediate controls
    # fan the schematic paths apart without moving that endpoint off its bone.
    brachialis_insertion = elbow + forearm_rotation @ forearm_bone_surface(
        "ulna", float(np.linalg.norm(wrist - elbow)), np.array((-0.008, 0.015, -0.060))
    )
    paths = (
        SchematicMusclePath("biceps_long", np.asarray((
            upper((-0.009, -0.012, -0.020)),
            upper((-0.100, -0.050, -0.110)),
            forearm((-0.060, -0.045, -0.055)),
            forearm((-0.010, -0.015, -0.120)),
        )), 0.011),
        SchematicMusclePath("biceps_short", np.asarray((
            upper((-0.012, 0.000, -0.065)),
            upper((-0.065, -0.005, -0.140)),
            forearm((-0.042, -0.003, -0.040)),
            forearm((-0.010, -0.005, -0.090)),
        )), 0.0095),
        SchematicMusclePath("brachialis", np.asarray((
            upper((-0.010, 0.009, -0.130)),
            upper((-0.036, 0.036, -0.205)),
            forearm((-0.026, 0.026, -0.025)),
            brachialis_insertion,
        )), 0.008),
        SchematicMusclePath("triceps", np.asarray((
            upper((0.012, 0.007, -0.040)),
            upper((0.066, 0.032, -0.140)),
            upper((0.062, 0.030, -0.305)),
            forearm((0.016, 0.014, -0.022)),
        )), 0.011),
    )
    return ArmAnchors(
        shoulder, elbow, wrist, upper_rotation, forearm_rotation,
        forearm_origin, paths,
    )


def sample_muscle_path(path: SchematicMusclePath, parameters: np.ndarray) -> np.ndarray:
    """Sample an illustrative curve at parameters in [0, 1]."""
    values = np.asarray(parameters, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError("Curve parameters must be a finite one-dimensional array in [0, 1].")
    t = values[:, None]
    p0, p1, p2, p3 = path.control_points
    return (1 - t)**3 * p0 + 3 * (1 - t)**2 * t * p1 + 3 * (1 - t) * t**2 * p2 + t**3 * p3


def configure_camera(model: mujoco.MjModel, camera: mujoco.MjvCamera) -> None:
    """Frame the arm's working range, independent of its world mount tilt."""
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.fixedcamid = -1
    camera.trackbodyid = -1
    if model.ntendon:
        data = mujoco.MjData(model)
        elbow_id = model.joint("elbow").id
        arm_bodies = {model.body("humerus_r").id}
        for body in range(model.nbody):
            if int(model.body_parentid[body]) in arm_bodies:
                arm_bodies.add(body)
        # Native mesh bounds avoid centering on wrap helpers or a large
        # clavicle/scapula. Only private forward kinematics are evaluated.
        bounds = []
        for geom in range(model.ngeom):
            if int(model.geom_bodyid[geom]) not in arm_bodies:
                continue
            if model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_MESH:
                mesh = int(model.geom_dataid[geom])
                start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
                vertices = model.mesh_vert[start:start + count]
                low, high = vertices.min(axis=0), vertices.max(axis=0)
            elif geom == model.geom("payload").id:
                low, high = -model.geom_size[geom, 0] * np.ones(3), model.geom_size[geom, 0] * np.ones(3)
            else:
                continue
            corners = np.array([
                (x, y, z) for x in (low[0], high[0])
                for y in (low[1], high[1]) for z in (low[2], high[2])
            ])
            bounds.append((geom, corners))
        points = []
        for angle in np.deg2rad((15.0, 75.0, 150.0)):
            data.qpos[model.jnt_qposadr[elbow_id]] = angle
            mujoco.mj_forward(model, data)
            points.extend(corners @ data.geom_xmat[geom].reshape(3, 3).T + data.geom_xpos[geom]
                          for geom, corners in bounds)
        points = np.concatenate(points)
        low, high = points.min(axis=0), points.max(axis=0)
        camera.lookat[:] = (low + high) / 2
        half_extent = 0.5 * np.linalg.norm(high - low)
        camera.distance = float(1.15 * half_extent / np.sin(np.deg2rad(model.vis.global_.fovy / 2)))
        camera.azimuth = 105.0
        camera.elevation = -12.0
        return
    reference = schematic_anchors(model, np.deg2rad(75.0))
    camera.lookat[:] = reference.elbow + np.array((-0.060, 0.0, 0.055))
    camera.distance = 0.85
    camera.azimuth = 105.0
    camera.elevation = -12.0


def configure_options(
    options: mujoco.MjvOption, model: mujoco.MjModel | None = None, *,
    show_contacts: bool = False,
) -> None:
    """Show native bone meshes; hide wrap geometry, markers, and debug overlays."""
    options.geomgroup[0] = bool(model is not None and model.ntendon)
    options.geomgroup[3] = 0
    options.geomgroup[5] = show_contacts
    options.sitegroup[:] = 0
    options.label = mujoco.mjtLabel.mjLABEL_NONE
    options.frame = mujoco.mjtFrame.mjFRAME_NONE
    for name in ("mjVIS_JOINT", "mjVIS_ACTUATOR", "mjVIS_COM", "mjVIS_CONTACTPOINT",
                 "mjVIS_CONTACTFORCE", "mjVIS_CONSTRAINT", "mjVIS_TENDON"):
        flag = getattr(mujoco.mjtVisFlag, name, None)
        if flag is not None:
            options.flags[flag] = 0
    # Full-length native lines are replaced by separately colored tissue parts.
    options.flags[mujoco.mjtVisFlag.mjVIS_TENDON] = 0


@dataclass(frozen=True)
class _Shape:
    kind: int
    first: np.ndarray
    second: np.ndarray | None
    radius: float
    rgba: tuple[float, ...] | np.ndarray


def _arm_shapes(anchors: ArmAnchors) -> list[_Shape]:
    shapes: list[_Shape] = []

    def sphere(point, radius, color):
        shapes.append(_Shape(mujoco.mjtGeom.mjGEOM_SPHERE, point, None, radius, color))

    def capsule(start, end, radius, color):
        shapes.append(_Shape(mujoco.mjtGeom.mjGEOM_CAPSULE, start, end, radius, color))

    def cylinder(start, end, radius, color):
        shapes.append(_Shape(mujoco.mjtGeom.mjGEOM_CYLINDER, start, end, radius, color))

    shoulder, elbow, wrist = anchors.shoulder, anchors.elbow, anchors.wrist
    upper_axis = (elbow - shoulder) / np.linalg.norm(elbow - shoulder)
    hinge_axis = anchors.upper_rotation[:, 1]
    forearm_axis = (wrist - elbow) / np.linalg.norm(wrist - elbow)
    capsule(shoulder + 0.015 * upper_axis, elbow - 0.018 * upper_axis, 0.012, _BONE)
    sphere(shoulder, 0.019, _BONE_END)
    sphere(elbow - 0.023 * upper_axis, 0.017, _BONE_END)

    # Radius and ulna are two separate thin shafts, rather than a solid arm tube.
    for name in _FOREARM_BONES:
        start, end, radius = forearm_bone_axis(name, float(np.linalg.norm(wrist - elbow)))
        proximal = elbow + anchors.forearm_rotation @ start
        distal = elbow + anchors.forearm_rotation @ end
        capsule(proximal, distal, radius, _BONE)
        sphere(proximal, 0.0105, _BONE_END)
        sphere(distal, 0.009, _BONE_END)
    capsule(wrist - 0.035 * forearm_axis, wrist, 0.010, _BONE_END)

    # A visible mechanical hinge identifies the single simulated degree of freedom.
    cylinder(elbow - 0.027 * hinge_axis, elbow + 0.027 * hinge_axis, 0.025, _JOINT)
    for side in (-1, 1):
        face = elbow + side * 0.028 * hinge_axis
        cylinder(face, face + side * 0.003 * hinge_axis, 0.020, _JOINT_FACE)
        sphere(face + side * 0.004 * hinge_axis, 0.0065, _JOINT)

    # A compact dumbbell symbol marks the payload; it does not change its inertia.
    weight_axis = anchors.forearm_rotation[:, 1]
    cylinder(wrist - 0.065 * weight_axis, wrist + 0.065 * weight_axis, 0.007, _HANDLE)
    for side in (-1, 1):
        center = wrist + side * 0.047 * weight_axis
        cylinder(center - 0.012 * weight_axis, center + 0.012 * weight_axis, 0.036, _WEIGHT)
        cylinder(center + side * 0.012 * weight_axis,
                 center + side * 0.014 * weight_axis, 0.012, _HANDLE)
    return shapes


@dataclass(frozen=True)
class TissuePath:
    """Display path in world metres with an arc coordinate in model metres.

    The lumped series tendon is split equally between the two attachments.
    This is an explicit visual assumption, not measured anatomical junctions.
    """

    name: str
    points: np.ndarray
    arc: np.ndarray
    fiber_length: float
    tendon_length: float
    belly_radius: float
    tendon_radius: float
    muscle_rgba: np.ndarray
    tendon_rgba: np.ndarray

    @property
    def belly_start(self) -> float:
        return self.tendon_length * 0.5

    @property
    def belly_end(self) -> float:
        return self.belly_start + self.fiber_length


def native_tissue_points(model, data, tendon_ids, scene=None) -> tuple[np.ndarray, ...]:
    """Reuse MuJoCo's tessellated wrap arcs, never join tangent points by chords."""
    scene = scene if scene is not None else mujoco.MjvScene(model, maxgeom=512)
    options = mujoco.MjvOption()
    options.geomgroup[:] = 0
    options.sitegroup[:] = 0
    options.flags[:] = 0
    options.flags[mujoco.mjtVisFlag.mjVIS_TENDON] = 1
    mujoco.mjv_updateScene(model, data, options, None, mujoco.MjvCamera(),
                          mujoco.mjtCatBit.mjCAT_ALL, scene)
    paths = {int(index): [] for index in tendon_ids}
    for i in range(scene.ngeom):
        geom = scene.geoms[i]
        if geom.objtype != mujoco.mjtObj.mjOBJ_TENDON or geom.objid not in paths:
            continue
        points = paths[geom.objid]
        axis = geom.mat.reshape(3, 3)[:, 2]
        start = (geom.pos - geom.size[2] * axis).astype(np.float64)
        end = (geom.pos + geom.size[2] * axis).astype(np.float64)
        if not points:
            points.append(start)
        # Native capsules are already in tendon order. Tiny float32 endpoint
        # discrepancies are omitted rather than creating microscopic segments.
        if np.linalg.norm(end - points[-1]) > 1e-8:
            points.append(end)
    return tuple(np.asarray(paths[int(index)]) for index in tendon_ids)


def tissue_paths(env, color_mode="tissue", *, kinematic_data=None, path_scene=None) -> tuple[TissuePath, ...]:
    """Map independent physical lengths to an approximately constant-volume view."""
    if color_mode not in COLOR_MODES:
        raise ValueError(f"Unknown muscle color mode: {color_mode}")
    state = env.telemetry()
    if env.geometry is not None:
        if kinematic_data is None:
            model = copy.copy(env.model)
            kinematic_data = mujoco.MjData(model)
            mujoco.mj_copyData(kinematic_data, model, env.data)
            mujoco.mj_forward(model, kinematic_data)
        points = native_tissue_points(env.model, kinematic_data, env.geometry.tendon_ids, path_scene)
        radii = [0.011 * np.sqrt(m.params.max_isometric_force / 500) for m in env.muscles]
    else:
        anchors = schematic_anchors(env.model, float(env.data.qpos[env.qpos_index]))
        points = tuple(sample_muscle_path(path, np.linspace(0, 1, 65)) for path in anchors.muscles)
        radii = [path.belly_radius for path in anchors.muscles]
    paths = []
    for index, (muscle, vertices, reference_radius) in enumerate(zip(env.muscles, points, radii, strict=True)):
        fiber = state["fiber_length_m"][index]
        tendon = state["tendon_length_m"][index]
        arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(vertices, axis=0), axis=1))]
        # For MyoArm this only corrects micron-scale arc tessellation error.
        # Toy paths are illustrative, so only their relative partition is physical.
        arc *= (fiber + tendon) / arc[-1]
        force = state["tendon_force_N"][index] / muscle.params.max_isometric_force
        belly_color = activation_rgba(state["activation"][index])
        tendon_color = tendon_rgba(force)
        if color_mode != "tissue":
            value = {"activation": state["activation"][index], "force": force,
                     "strain": state["tendon_strain"][index] / 0.04}[color_mode]
            belly_color = tendon_color = activation_rgba(value)
        paths.append(TissuePath(
            muscle.params.name, vertices, arc, fiber, tendon,
            float(reference_radius * np.sqrt(muscle.params.optimal_fiber_length / fiber)),
            # A slack or zero-length tendon must not inflate without bound.
            # Its display thickness stays nominal until it becomes taut.
            float(0.0025 * np.sqrt(muscle.params.tendon_slack_length /
                                  max(muscle.params.tendon_slack_length, tendon))),
            belly_color, tendon_color,
        ))
    return tuple(paths)


def _tissue_shapes(paths: tuple[TissuePath, ...]) -> list[_Shape]:
    shapes = []
    for path in paths:
        # Keep every native bend and insert the two exact tissue junctions.
        # Fine belly samples make the tapered capsule envelope visually smooth.
        positions = np.unique(np.r_[path.arc, path.belly_start, path.belly_end,
                                    np.linspace(path.belly_start, path.belly_end, 41)])
        points = np.column_stack([np.interp(positions, path.arc, path.points[:, axis]) for axis in range(3)])
        for index, (start, end) in enumerate(zip(points[:-1], points[1:], strict=True)):
            middle = (positions[index] + positions[index + 1]) * 0.5
            belly = path.belly_start < middle < path.belly_end
            radius = path.tendon_radius
            if belly:
                u = (middle - path.belly_start) / path.fiber_length
                radius = path.belly_radius * np.sqrt(np.sin(np.pi * u))
            shapes.append(_Shape(mujoco.mjtGeom.mjGEOM_CAPSULE, start, end, radius,
                                 path.muscle_rgba if belly else path.tendon_rgba))
        for endpoint in (points[0], points[-1]):
            shapes.append(_Shape(mujoco.mjtGeom.mjGEOM_SPHERE, endpoint, None, 0.0035, _ATTACHMENT))
    return shapes


def append_arm_geometry(
    scene: mujoco.MjvScene, env: PreacherCurlEnv, color_mode: str = "tissue", *,
    kinematic_data: mujoco.MjData | None = None, path_scene: mujoco.MjvScene | None = None,
) -> int:
    """Append diagnostic muscle paths and return the added geom count.

    Call after the scene's ordinary update, once per displayed frame. A caller
    owns scene clearing/lifetime. Insufficient capacity raises before appending
    anything, avoiding a misleading partial muscle drawing.
    """
    shapes = []
    if env.geometry is None:
        anchors = schematic_anchors(env.model, float(env.data.qpos[env.qpos_index]))
        shapes = _arm_shapes(anchors)
    shapes += _tissue_shapes(tissue_paths(env, color_mode, kinematic_data=kinematic_data, path_scene=path_scene))
    initial = scene.ngeom
    if initial + len(shapes) > scene.maxgeom:
        raise ValueError(f"Arm display needs {len(shapes)} free geoms; scene has {scene.maxgeom - initial}.")
    identity = np.eye(3, dtype=np.float64).ravel()
    for offset, shape in enumerate(shapes):
        geom = scene.geoms[initial + offset]
        size = np.full(3, shape.radius, dtype=np.float64)
        position = np.asarray(shape.first, dtype=np.float64)
        rotation = identity
        if shape.kind == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
            direction = shape.second - shape.first
            size[2] = 0.5 * np.linalg.norm(direction)
            position = 0.5 * (shape.first + shape.second)
            quaternion = np.empty(4, dtype=np.float64)
            mujoco.mju_quatZ2Vec(quaternion, direction)
            rotation = _quaternion_matrix(quaternion).ravel()
        mujoco.mjv_initGeom(
            geom, shape.kind, size, position, rotation,
            np.asarray(shape.rgba, dtype=np.float32),
        )
        if shape.second is not None and shape.kind != mujoco.mjtGeom.mjGEOM_ELLIPSOID:
            mujoco.mjv_connector(geom, shape.kind, shape.radius, shape.first, shape.second)
        geom.category = mujoco.mjtCatBit.mjCAT_DECOR
        geom.objtype = mujoco.mjtObj.mjOBJ_UNKNOWN
        geom.objid = -1
        geom.segid = -1
        geom.specular = 0.15
        geom.shininess = 0.25
    scene.ngeom = initial + len(shapes)
    return len(shapes)
