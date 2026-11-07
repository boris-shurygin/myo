"""Presentation-only cylinders for independent fiber and series-tendon lengths."""
from __future__ import annotations

import copy
from dataclasses import dataclass

import mujoco
import numpy as np

from .demo import legend_images
from .visualization import activation_rgba, tendon_rgba


@dataclass(frozen=True)
class TissueCylinder:
    muscle: str
    tissue: str
    start: np.ndarray
    end: np.ndarray
    length: float
    radius: float
    rgba: np.ndarray


def tissue_cylinders(env):
    """A muscle at each fixed anchor and its tendon toward the load.

    The material junction follows the independent fiber state, not a fixed
    fraction of the path. Thickness preserves an illustrative reference
    volume. A slack tendon's end-to-end length is not material compression.
    """
    h = env.config.half_span_m
    load = np.array([0.,0.,h+env.position])
    fibers,paths,forces = env.fiber_lengths(),env.path_lengths(),env.forces()
    result = []
    for i,(name,z) in enumerate((("upper",2*h),("lower",0.))):
        anchor = np.array([0.,0.,z])
        direction = (load-anchor)/paths[i]
        fiber = float(fibers[i])
        tendon = float(paths[i]-fiber)
        junction = anchor+direction*fiber
        for tissue,start,end,length,reference,radius,rgba in (
            ("muscle",anchor,junction,fiber,env.parameters.optimal_fiber_length,.024,
             activation_rgba(env.activation[i])),
            ("tendon",junction,load,tendon,env.parameters.tendon_slack_length,.008,
             tendon_rgba(forces[i]/env.parameters.max_isometric_force)),
        ):
            if length > 1e-9:
                result.append(TissueCylinder(name,tissue,start,end,length,
                    float(radius*np.sqrt(reference/length)),rgba))
    return result


def configure_camera(camera, env):
    camera.lookat[:] = [0,0,env.config.half_span_m]
    camera.distance = 1.85
    camera.azimuth = 90.
    camera.elevation = -5.


def configure_model(model):
    """Light only the renderer's copy so the fixed signal colors stay legible."""
    model.vis.headlight.ambient[:] = .4
    model.vis.headlight.diffuse[:] = .7
    model.vis.headlight.specular[:] = .15


def configure_options(options):
    # Native full-path lines would obscure the two independently drawn tissues.
    options.flags[mujoco.mjtVisFlag.mjVIS_TENDON] = 0
    options.label = mujoco.mjtLabel.mjLABEL_NONE


def append_tissues(scene, env):
    cylinders = tissue_cylinders(env)
    if scene.ngeom+len(cylinders) > scene.maxgeom:
        raise ValueError("Not enough free geometry slots for vertical tissues")
    for cylinder in cylinders:
        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(geom,mujoco.mjtGeom.mjGEOM_CYLINDER,np.zeros(3),
                           np.zeros(3),np.eye(3).ravel(),cylinder.rgba)
        mujoco.mjv_connector(geom,mujoco.mjtGeom.mjGEOM_CYLINDER,cylinder.radius,
                            cylinder.start,cylinder.end)
        geom.category = mujoco.mjtCatBit.mjCAT_DECOR
        geom.objtype = mujoco.mjtObj.mjOBJ_UNKNOWN
        geom.objid = geom.segid = -1
        geom.specular,geom.shininess = .15,.25
        scene.ngeom += 1
    return len(cylinders)


def text_overlays(env):
    font = mujoco.mjtFont.mjFONT_NORMAL
    grid = mujoco.mjtGridPos
    mechanics = ("active fibers, rigid tendons", "elastic fibers and tendons",
                 "elasticity + 40 ms transport", "elasticity + transport + activation")
    result = [(font,grid.mjGRID_TOPLEFT,
        f"VERTICAL MUSCLE STAND | variant {env.config.variant}\n"
        f"{mechanics[env.config.variant-1]}\n"
        f"Time {env.time:.2f} s | Load {env.config.mass_kg:g} kg\n"
        f"Position {1000*env.position:+.1f} mm | Speed {env.velocity:+.3f} m/s\n"
        "Mouse rotate/zoom | Esc close", ""),
        (font,grid.mjGRID_BOTTOMLEFT,
         "Muscle activation 0-100%\n\n"
         f"Tendon load 0-{env.parameters.max_isometric_force:g} N (color scale)", "")]
    fibers,paths,forces = env.fiber_lengths(),env.path_lengths(),env.forces()
    for i,(name,position) in enumerate((("UPPER",grid.mjGRID_TOPRIGHT),("LOWER",grid.mjGRID_BOTTOMRIGHT))):
        tendon = paths[i]-fibers[i]
        strain = max(0.,tendon/env.parameters.tendon_slack_length-1.) if env.config.elastic else 0.
        labels = (f"{name} MUSCLE\nExcitation\nDelivered excitation\nActivation\n"
                  "Tendon force\nFiber length\nTendon length\nTendon strain")
        values = (f"\n{100*env.excitation[i]:.1f}%\n{100*env.delivered[i]:.1f}%\n"
                  f"{100*env.activation[i]:.1f}%\n{forces[i]:.2f} N\n"
                  f"{1000*fibers[i]:.2f} mm\n{1000*tendon:.2f} mm\n{100*strain:.2f}%")
        result.append((font,position,labels,values))
    return result


def render_frame(env, width=1200, height=900):
    """Render the same cylinders, signals and legends for visual QA/export."""
    model = copy.copy(env.model)
    configure_model(model)
    data = mujoco.MjData(model)
    mujoco.mj_copyData(data,model,env.data)
    mujoco.mj_forward(model,data)
    model.vis.global_.offwidth,model.vis.global_.offheight = width,height
    camera,options = mujoco.MjvCamera(),mujoco.MjvOption()
    configure_camera(camera,env)
    configure_options(options)
    scene = mujoco.MjvScene(model,maxgeom=32)
    gl = mujoco.GLContext(width,height)
    context = None
    try:
        gl.make_current()
        context = mujoco.MjrContext(model,mujoco.mjtFontScale.mjFONTSCALE_150)
        mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN,context)
        viewport = mujoco.MjrRect(0,0,width,height)
        mujoco.mjv_updateScene(model,data,options,None,camera,mujoco.mjtCatBit.mjCAT_ALL,scene)
        append_tissues(scene,env)
        mujoco.mjr_render(viewport,scene,context)
        for font,position,text,values in text_overlays(env):
            mujoco.mjr_overlay(font,position,viewport,text,values,context)
        for rectangle,strip in legend_images("tissue"):
            mujoco.mjr_drawPixels(strip.ravel(),None,rectangle,context)
        pixels = np.empty((height,width,3),dtype=np.uint8)
        mujoco.mjr_readPixels(pixels,None,viewport,context)
        return np.flipud(pixels).copy()
    finally:
        if context is not None:
            context.free()
        gl.free()
