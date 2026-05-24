from __future__ import annotations

import math
from typing import Any, Callable

_REGISTERED = False


def register_height_field_bsdf(mi: Any | None = None) -> None:
    global _REGISTERED
    if _REGISTERED:
        return
    import drjit as dr
    import mitsuba as mitsuba

    mi = mitsuba if mi is None else mi

    class KokoroHeightFieldReflector(mi.BSDF):
        def __init__(self, props):
            mi.BSDF.__init__(self, props)
            self.height = _compile_drjit_height_source(str(props["height_source"]), dr)
            self.width_m = float(props.get("width_m", 0.10))
            self.depth_m = float(props.get("depth_m", 0.10))
            self.normal_step_m = float(props.get("normal_step_m", 25e-6))
            self.reflectance = mi.Color3f(props.get("reflectance", [0.86, 0.88, 0.92]))
            self.lobe_kappa = float(props.get("lobe_kappa", 4096.0))
            flags = mi.BSDFFlags.GlossyReflection | mi.BSDFFlags.FrontSide | mi.BSDFFlags.BackSide
            self.m_components = [flags]
            self.m_flags = flags

        def sample(self, ctx, si, sample1, sample2, active):
            del sample1
            target = self._target_lobe(si, dr)
            raw = mi.Frame3f(target).to_world(mi.warp.square_to_von_mises_fisher(sample2, self.lobe_kappa))
            norm = dr.rsqrt(dr.maximum(raw.x * raw.x + raw.y * raw.y + raw.z * raw.z, 1e-8))
            wo = mi.Vector3f(raw.x * norm, raw.y * norm, dr.abs(raw.z * norm))
            bs = mi.BSDFSample3f()
            bs.wo = wo
            bs.pdf = self.pdf(ctx, si, wo, active)
            bs.eta = 1.0
            bs.sampled_component = mi.UInt32(0)
            bs.sampled_type = mi.UInt32(+mi.BSDFFlags.GlossyReflection)
            value = self.eval(ctx, si, wo, active)
            return bs, dr.select(bs.pdf > 0.0, value / bs.pdf, mi.Color3f(0.0))

        def eval(self, ctx, si, wo, active):
            del ctx
            return self.reflectance * self._pdf_for_target(wo, self._target_lobe(si, dr), dr, active)

        def pdf(self, ctx, si, wo, active):
            del ctx
            return self._pdf_for_target(wo, self._target_lobe(si, dr), dr, active)

        def eval_pdf(self, ctx, si, wo, active):
            return self.eval(ctx, si, wo, active), self.pdf(ctx, si, wo, active)

        def _target_lobe(self, si, dr):
            normal = self._surface_normal(si, dr)
            wi = mi.Vector3f(_component(si.wi, 0), _component(si.wi, 1), _component(si.wi, 2))
            dot = dr.dot(wi, normal)
            raw = mi.Vector3f(
                2.0 * dot * normal.x - wi.x,
                2.0 * dot * normal.y - wi.y,
                2.0 * dot * normal.z - wi.z,
            )
            norm = dr.rsqrt(dr.maximum(raw.x * raw.x + raw.y * raw.y + raw.z * raw.z, 1e-8))
            return mi.Vector3f(raw.x * norm, raw.y * norm, dr.abs(raw.z * norm))

        def _surface_normal(self, si, dr):
            x = _component(si.p, 0)
            y = _component(si.p, 1)
            step = self.normal_step_m
            dzdx = (self.height(x + step, y) - self.height(x - step, y)) / (2.0 * step)
            dzdy = (self.height(x, y + step) - self.height(x, y - step)) / (2.0 * step)
            norm = dr.rsqrt(dr.maximum(dzdx * dzdx + dzdy * dzdy + 1.0, 1e-8))
            return mi.Vector3f(-dzdx * norm, -dzdy * norm, norm)

        def _pdf_for_target(self, wo, target, dr, active):
            wo_norm = dr.rsqrt(dr.maximum(wo.x * wo.x + wo.y * wo.y + wo.z * wo.z, 1e-8))
            wo_unit = mi.Vector3f(wo.x * wo_norm, wo.y * wo_norm, wo.z * wo_norm)
            mirrored = mi.Vector3f(wo_unit.x, wo_unit.y, -wo_unit.z)
            pdf = self._vmf_pdf(dr.dot(wo_unit, target), dr) + self._vmf_pdf(dr.dot(mirrored, target), dr)
            valid = active & (_component(wo_unit, 2) > 0.0)
            return dr.select(valid, pdf, 0.0)

        def _vmf_pdf(self, cos_angle, dr):
            denom = 2.0 * dr.pi * (1.0 - dr.exp(-2.0 * self.lobe_kappa))
            return (self.lobe_kappa / denom) * dr.exp(self.lobe_kappa * (dr.clip(cos_angle, -1.0, 1.0) - 1.0))

        def to_string(self):
            return f"KokoroHeightFieldReflector[lobe_kappa={self.lobe_kappa}]"

    mi.register_bsdf("kokoro_height_field_reflector", lambda props: KokoroHeightFieldReflector(props))
    _REGISTERED = True


def _compile_drjit_height_source(source: str, dr) -> Callable[[Any, Any], Any]:
    namespace: dict[str, Any] = {
        "math": math,
        "dr": dr,
        "pyramid_height": lambda x, y, **kwargs: _pyramid_height(dr, x, y, **kwargs),
        "radial_rotated_pyramid_height": lambda x, y, **kwargs: _radial_rotated_pyramid_height(dr, x, y, **kwargs),
    }
    exec(compile(source, "<kokoro-mitsuba-height>", "exec"), namespace)
    height = namespace.get("height")
    if not callable(height):
        raise ValueError("height source must define a callable height(x, y)")
    return height


def _pyramid_height(dr, x, y, *, period_m: float = 500e-6, amplitude_m: float = 150e-6):
    if period_m <= 0:
        raise ValueError("period_m must be positive")
    period = float(period_m)
    u = (x - period * dr.floor(x / period)) / period
    v = (y - period * dr.floor(y / period)) / period
    edge_distance = dr.maximum(dr.abs(u - 0.5), dr.abs(v - 0.5))
    return float(amplitude_m) * dr.maximum(1.0 - 2.0 * edge_distance, 0.0)


def _radial_rotated_pyramid_height(
    dr,
    x,
    y,
    *,
    period_m: float = 500e-6,
    amplitude_m: float = 150e-6,
    width_m: float = 0.10,
    depth_m: float = 0.10,
    max_rotation_rad: float = math.pi,
    radial_power: float = 1.0,
):
    if period_m <= 0:
        raise ValueError("period_m must be positive")
    if width_m <= 0 or depth_m <= 0:
        raise ValueError("width_m and depth_m must be positive")
    if radial_power <= 0:
        raise ValueError("radial_power must be positive")
    period = float(period_m)
    center_x = dr.floor(x / period + 0.5) * period
    center_y = dr.floor(y / period + 0.5) * period
    local_x = x - center_x
    local_y = y - center_y
    radius = dr.sqrt(center_x * center_x + center_y * center_y)
    max_radius = math.sqrt((float(width_m) * 0.5) ** 2 + (float(depth_m) * 0.5) ** 2)
    angle = float(max_rotation_rad) * radius / max_radius ** float(radial_power)
    cos_angle = dr.cos(angle)
    sin_angle = dr.sin(angle)
    rotated_x = cos_angle * local_x + sin_angle * local_y
    rotated_y = -sin_angle * local_x + cos_angle * local_y
    edge_distance = dr.maximum(dr.abs(rotated_x), dr.abs(rotated_y)) / (period * 0.5)
    return float(amplitude_m) * dr.maximum(1.0 - edge_distance, 0.0)


def _component(vector: Any, index: int) -> Any:
    try:
        return vector[index]
    except TypeError:
        return (vector.x, vector.y, vector.z)[index]
