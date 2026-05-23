from __future__ import annotations

from pathlib import Path
from typing import Any

from .brdf import load_npz_surrogate

_REGISTERED = False


def register_kokoro_bsdf(mi: Any | None = None) -> None:
    global _REGISTERED
    if _REGISTERED:
        return
    import drjit as dr
    import mitsuba as mitsuba

    mi = mitsuba if mi is None else mi

    class KokoroNeuralReflector(mi.BSDF):
        def __init__(self, props):
            mi.BSDF.__init__(self, props)
            checkpoint = Path(str(props["checkpoint"]))
            surrogate = load_npz_surrogate(checkpoint)
            self.weights = surrogate.weights
            self.biases = surrogate.biases
            self.width_m = float(surrogate.metadata["width_m"])
            self.depth_m = float(surrogate.metadata["depth_m"])
            period = surrogate.metadata.get("feature_period_m")
            self.feature_period_m = None if period is None else float(period)
            self.reflectance = mi.Color3f(props.get("reflectance", [0.86, 0.88, 0.92]))
            self.lobe_kappa = float(props.get("lobe_kappa", 96.0))
            flags = mi.BSDFFlags.GlossyReflection | mi.BSDFFlags.FrontSide | mi.BSDFFlags.BackSide
            self.m_components = [flags]
            self.m_flags = flags

        def sample(self, ctx, si, sample1, sample2, active):
            del sample1
            target = self._target_direction(si, dr)
            raw = mi.Frame3f(target).to_world(mi.warp.square_to_von_mises_fisher(sample2, self.lobe_kappa))
            norm = dr.rsqrt(dr.maximum(raw.x * raw.x + raw.y * raw.y + raw.z * raw.z, 1e-8))
            wo = mi.Vector3f(raw.x * norm, raw.y * norm, dr.abs(raw.z * norm))
            bs = mi.BSDFSample3f()
            bs.wo = wo
            bs.pdf = self._pdf_for_target(wo, target, dr, active)
            bs.eta = 1.0
            bs.sampled_component = mi.UInt32(0)
            bs.sampled_type = mi.UInt32(+mi.BSDFFlags.GlossyReflection)
            value = self.eval(ctx, si, wo, active)
            return bs, dr.select(bs.pdf > 0.0, value / bs.pdf, mi.Color3f(0.0))

        def eval(self, ctx, si, wo, active):
            del ctx
            target = self._target_direction(si, dr)
            return self.reflectance * self._pdf_for_target(wo, target, dr, active)

        def pdf(self, ctx, si, wo, active):
            del ctx
            return self._pdf_for_target(wo, self._target_direction(si, dr), dr, active)

        def eval_pdf(self, ctx, si, wo, active):
            return self.eval(ctx, si, wo, active), self.pdf(ctx, si, wo, active)

        def _eval_mlp(self, values, dr):
            x = values
            for layer_index, (weight, bias) in enumerate(zip(self.weights, self.biases)):
                out = []
                for row, b_value in zip(weight, bias):
                    value = float(b_value)
                    for item, w_value in zip(x, row):
                        value = value + item * float(w_value)
                    out.append(dr.tanh(value) if layer_index < len(self.weights) - 1 else value)
                x = out
            return x

        def _target_direction(self, si, dr):
            x_feature, y_feature = self._position_features(si, dr)
            features = [
                x_feature,
                y_feature,
                dr.clip(_component(si.wi, 2), -1.0, 1.0),
                _component(si.wi, 0),
                _component(si.wi, 1),
            ]
            raw = self._eval_mlp(features, dr)
            norm = dr.rsqrt(dr.maximum(raw[0] * raw[0] + raw[1] * raw[1] + raw[2] * raw[2], 1e-8))
            return mi.Vector3f(raw[0] * norm, raw[1] * norm, dr.abs(raw[2] * norm))

        def _position_features(self, si, dr):
            x = _component(si.p, 0)
            y = _component(si.p, 1)
            if self.feature_period_m is None:
                return x / (self.width_m * 0.5), y / (self.depth_m * 0.5)
            period = self.feature_period_m
            x_phase = x - period * dr.floor(x / period)
            y_phase = y - period * dr.floor(y / period)
            return x_phase / (period * 0.5) - 1.0, y_phase / (period * 0.5) - 1.0

        def _pdf_for_target(self, wo, target, dr, active):
            wo_norm = dr.rsqrt(dr.maximum(wo.x * wo.x + wo.y * wo.y + wo.z * wo.z, 1e-8))
            wo_unit = mi.Vector3f(wo.x * wo_norm, wo.y * wo_norm, wo.z * wo_norm)
            mirrored = mi.Vector3f(wo_unit.x, wo_unit.y, -wo_unit.z)
            pdf = (
                self._vmf_pdf(dr.dot(wo_unit, target), dr)
                + self._vmf_pdf(dr.dot(mirrored, target), dr)
            )
            valid = active & (_component(wo_unit, 2) > 0.0)
            return dr.select(valid, pdf, 0.0)

        def _vmf_pdf(self, cos_angle, dr):
            kappa = self.lobe_kappa
            denom = 2.0 * dr.pi * (1.0 - dr.exp(-2.0 * kappa))
            return (kappa / denom) * dr.exp(kappa * (dr.clip(cos_angle, -1.0, 1.0) - 1.0))

        def to_string(self):
            return f"KokoroNeuralReflector[lobe_kappa={self.lobe_kappa}]"

    mi.register_bsdf("kokoro_neural_reflector", lambda props: KokoroNeuralReflector(props))
    _REGISTERED = True


def _component(vector: Any, index: int) -> Any:
    try:
        return vector[index]
    except TypeError:
        return (vector.x, vector.y, vector.z)[index]
