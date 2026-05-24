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
            local_period = surrogate.metadata.get("local_feature_period_m")
            self.local_feature_period_m = None if local_period is None else float(local_period)
            self.activation = str(surrogate.metadata.get("activation", "tanh"))
            self.omega_0 = float(surrogate.metadata.get("omega_0", 12.0))
            self.position_frequency_count = int(surrogate.metadata.get("position_frequency_count", 0))
            include_position = surrogate.metadata.get("include_position_features")
            self.include_position_features = self.weights[0].shape[1] > 3 if include_position is None else bool(include_position)
            self.output_dim = int(self.weights[-1].shape[0])
            self.reflectance = mi.Color3f(props.get("reflectance", [0.86, 0.88, 0.92]))
            self.lobe_kappa = float(props.get("lobe_kappa", 96.0))
            default_ring_lobes = 4 if self.output_dim > 5 else (16 if self.output_dim > 3 else 1)
            self.ring_lobe_count = max(1, int(props.get("ring_lobe_count", default_ring_lobes)))
            flags = mi.BSDFFlags.GlossyReflection | mi.BSDFFlags.FrontSide | mi.BSDFFlags.BackSide
            self.m_components = [flags]
            self.m_flags = flags

        def sample(self, ctx, si, sample1, sample2, active):
            axis, cone_cos, phase = self._target_lobe(si, dr)
            target = self._sample_ring_target(axis, cone_cos, phase, sample1, dr)
            raw = mi.Frame3f(target).to_world(mi.warp.square_to_von_mises_fisher(sample2, self.lobe_kappa))
            norm = dr.rsqrt(dr.maximum(raw.x * raw.x + raw.y * raw.y + raw.z * raw.z, 1e-8))
            wo = mi.Vector3f(raw.x * norm, raw.y * norm, dr.abs(raw.z * norm))
            bs = mi.BSDFSample3f()
            bs.wo = wo
            bs.pdf = self._pdf_for_lobe(wo, axis, cone_cos, phase, dr, active)
            bs.eta = 1.0
            bs.sampled_component = mi.UInt32(0)
            bs.sampled_type = mi.UInt32(+mi.BSDFFlags.GlossyReflection)
            value = self.eval(ctx, si, wo, active)
            return bs, dr.select(bs.pdf > 0.0, value / bs.pdf, mi.Color3f(0.0))

        def eval(self, ctx, si, wo, active):
            del ctx
            axis, cone_cos, phase = self._target_lobe(si, dr)
            return self.reflectance * self._pdf_for_lobe(wo, axis, cone_cos, phase, dr, active)

        def pdf(self, ctx, si, wo, active):
            del ctx
            axis, cone_cos, phase = self._target_lobe(si, dr)
            return self._pdf_for_lobe(wo, axis, cone_cos, phase, dr, active)

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
                    out.append(self._activate(value, dr) if layer_index < len(self.weights) - 1 else value)
                x = out
            return x

        def _activate(self, value, dr):
            if self.activation == "sine":
                return dr.sin(self.omega_0 * value)
            return dr.tanh(value)

        def _target_lobe(self, si, dr):
            position_features = self._position_features(si, dr)
            features = self._features(si, position_features, dr)
            raw = self._eval_mlp(features, dr)
            norm = dr.rsqrt(dr.maximum(raw[0] * raw[0] + raw[1] * raw[1] + raw[2] * raw[2], 1e-8))
            axis = mi.Vector3f(raw[0] * norm, raw[1] * norm, dr.abs(raw[2] * norm))
            cone_cos = 1.0
            phase = 0.0
            if self.output_dim > 3:
                cone_cos = dr.clip(1.0 / (1.0 + dr.exp(-raw[3])), 0.0, 1.0)
            if self.output_dim > 5:
                phase = 0.25 * dr.atan2(raw[5], raw[4])
            return axis, cone_cos, phase

        def _position_features(self, si, dr):
            x = _component(si.p, 0)
            y = _component(si.p, 1)
            if self.feature_period_m is None:
                features = [x / (self.width_m * 0.5), y / (self.depth_m * 0.5)]
            else:
                period = self.feature_period_m
                x_phase = x - period * dr.floor(x / period)
                y_phase = y - period * dr.floor(y / period)
                features = [x_phase / (period * 0.5) - 1.0, y_phase / (period * 0.5) - 1.0]
            if self.local_feature_period_m is not None:
                period = self.local_feature_period_m
                x_phase = x - period * dr.floor(x / period)
                y_phase = y - period * dr.floor(y / period)
                features.extend([x_phase / (period * 0.5) - 1.0, y_phase / (period * 0.5) - 1.0])
            return features

        def _features(self, si, position_features, dr):
            incident = [
                dr.clip(_component(si.wi, 2), -1.0, 1.0),
                _component(si.wi, 0),
                _component(si.wi, 1),
            ]
            if not self.include_position_features:
                return incident
            x_feature = position_features[0]
            y_feature = position_features[1]
            if self.position_frequency_count <= 0:
                return [*position_features, *incident]
            encoded = [*position_features]
            for index in range(self.position_frequency_count):
                frequency = float(2 ** index) * dr.pi
                encoded.extend([
                    dr.sin(frequency * x_feature),
                    dr.cos(frequency * x_feature),
                    dr.sin(frequency * y_feature),
                    dr.cos(frequency * y_feature),
                ])
            return [*encoded, *incident]

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

        def _pdf_for_lobe(self, wo, axis, cone_cos, phase, dr, active):
            if self.ring_lobe_count <= 1:
                return self._pdf_for_target(wo, axis, dr, active)
            pdf = 0.0
            for index in range(self.ring_lobe_count):
                phi = phase + 2.0 * dr.pi * float(index) / float(self.ring_lobe_count)
                pdf = pdf + self._pdf_for_target(wo, self._ring_target(axis, cone_cos, phi, dr), dr, active)
            return pdf / float(self.ring_lobe_count)

        def _sample_ring_target(self, axis, cone_cos, phase, sample1, dr):
            if self.ring_lobe_count <= 1:
                return axis
            scaled = sample1 * float(self.ring_lobe_count)
            lobe_index = mi.UInt32(dr.minimum(dr.floor(scaled), float(self.ring_lobe_count - 1)))
            target = axis
            for index in range(self.ring_lobe_count):
                phi = phase + 2.0 * dr.pi * float(index) / float(self.ring_lobe_count)
                candidate = self._ring_target(axis, cone_cos, phi, dr)
                selected = lobe_index == mi.UInt32(index)
                target = mi.Vector3f(
                    dr.select(selected, candidate.x, target.x),
                    dr.select(selected, candidate.y, target.y),
                    dr.select(selected, candidate.z, target.z),
                )
            return target

        def _ring_target(self, axis, cone_cos, phi, dr):
            cone_cos = dr.clip(cone_cos, 0.0, 1.0)
            cone_sin = dr.sqrt(dr.maximum(1.0 - cone_cos * cone_cos, 0.0))
            local = mi.Vector3f(cone_sin * dr.cos(phi), cone_sin * dr.sin(phi), cone_cos)
            raw = mi.Frame3f(axis).to_world(local)
            norm = dr.rsqrt(dr.maximum(raw.x * raw.x + raw.y * raw.y + raw.z * raw.z, 1e-8))
            return mi.Vector3f(raw.x * norm, raw.y * norm, dr.abs(raw.z * norm))

        def _vmf_pdf(self, cos_angle, dr):
            kappa = self.lobe_kappa
            denom = 2.0 * dr.pi * (1.0 - dr.exp(-2.0 * kappa))
            return (kappa / denom) * dr.exp(kappa * (dr.clip(cos_angle, -1.0, 1.0) - 1.0))

        def to_string(self):
            return f"KokoroNeuralReflector[lobe_kappa={self.lobe_kappa}, ring_lobe_count={self.ring_lobe_count}]"

    mi.register_bsdf("kokoro_neural_reflector", lambda props: KokoroNeuralReflector(props))
    _REGISTERED = True


def _component(vector: Any, index: int) -> Any:
    try:
        return vector[index]
    except TypeError:
        return (vector.x, vector.y, vector.z)[index]
