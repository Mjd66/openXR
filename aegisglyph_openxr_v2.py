#!/usr/bin/env python3
"""
AegisGlyph OpenXR v2
====================

Desktop-first, accessibility-oriented XR guidance engine with:
- optional OpenXR capability probing via pyopenxr (if installed)
- deterministic simulator fallback when no XR runtime/bindings are available
- symbolic guidance synthesis for blind / low-vision navigation experiments
- negative-layer filtering (noise, contradiction, overload, stale-signal suppression)
- affect-to-energy conversion for adaptive cue intensity
- JSONL logging for later analysis

This script is intentionally honest about the state of Python + OpenXR:
OpenXR support is real; robust Python graphics/session integration is still the fiddly part.
So v2 gives you a production-shaped architecture with a simulator that runs today,
and a clean integration seam for a real OpenXR session.
"""
from __future__ import annotations

import argparse
import dataclasses
from dataclasses import dataclass, field
from enum import Enum
import json
import math
import os
from pathlib import Path
import random
import sys
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Math / core data
# ---------------------------------------------------------------------------


def clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


@dataclass
class Vec3:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def __add__(self, other: "Vec3") -> "Vec3":
        return Vec3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: "Vec3") -> "Vec3":
        return Vec3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, k: float) -> "Vec3":
        return Vec3(self.x * k, self.y * k, self.z * k)

    __rmul__ = __mul__

    def length(self) -> float:
        return math.sqrt(self.x * self.x + self.y * self.y + self.z * self.z)

    def normalized(self) -> "Vec3":
        n = self.length()
        if n <= 1e-9:
            return Vec3()
        return Vec3(self.x / n, self.y / n, self.z / n)

    def dot(self, other: "Vec3") -> float:
        return self.x * other.x + self.y * other.y + self.z * other.z

    def to_dict(self) -> Dict[str, float]:
        return {"x": round(self.x, 5), "y": round(self.y, 5), "z": round(self.z, 5)}


@dataclass
class Pose:
    position: Vec3 = field(default_factory=Vec3)
    forward: Vec3 = field(default_factory=lambda: Vec3(0.0, 0.0, -1.0))
    confidence: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "position": self.position.to_dict(),
            "forward": self.forward.to_dict(),
            "confidence": round(self.confidence, 4),
        }


@dataclass
class HandState:
    tracked: bool = False
    pinch: float = 0.0
    aim: Pose = field(default_factory=Pose)
    joint_energy: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tracked": self.tracked,
            "pinch": round(self.pinch, 4),
            "joint_energy": round(self.joint_energy, 4),
            "aim": self.aim.to_dict(),
        }


@dataclass
class GazeState:
    tracked: bool = False
    direction: Vec3 = field(default_factory=lambda: Vec3(0.0, 0.0, -1.0))
    stability: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tracked": self.tracked,
            "direction": self.direction.to_dict(),
            "stability": round(self.stability, 4),
        }


@dataclass
class AffectSignal:
    valence: float = 0.3   # -1..+1
    arousal: float = 0.4   # 0..1
    pain: float = 0.0      # 0..1
    overload: float = 0.0  # 0..1
    clarity: float = 0.7   # 0..1

    def normalized(self) -> "AffectSignal":
        return AffectSignal(
            valence=clamp(self.valence, -1.0, 1.0),
            arousal=clamp(self.arousal, 0.0, 1.0),
            pain=clamp(self.pain, 0.0, 1.0),
            overload=clamp(self.overload, 0.0, 1.0),
            clarity=clamp(self.clarity, 0.0, 1.0),
        )

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self.normalized())


class GuidanceMode(str, Enum):
    CALM = "calm"
    FOCUS = "focus"
    ALERT = "alert"
    RECOVER = "recover"


@dataclass
class RuntimeCaps:
    backend: str
    has_pyopenxr: bool = False
    openxr_extensions: List[str] = field(default_factory=list)
    hand_tracking: bool = False
    eye_gaze: bool = False
    visibility_mask: bool = False
    debug_utils: bool = False
    action_priorities: bool = False
    graphics_enable: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass
class WorldTarget:
    label: str
    position: Vec3
    importance: float = 1.0


@dataclass
class GuidanceFrame:
    t: float
    mode: GuidanceMode
    cue_word: str
    turn_bias: float
    pulse: float
    stability: float
    energy: float
    clutter_removed: float
    target: Optional[str]
    notes: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "t": round(self.t, 3),
            "mode": self.mode.value,
            "cue_word": self.cue_word,
            "turn_bias": round(self.turn_bias, 4),
            "pulse": round(self.pulse, 4),
            "stability": round(self.stability, 4),
            "energy": round(self.energy, 4),
            "clutter_removed": round(self.clutter_removed, 4),
            "target": self.target,
            "notes": list(self.notes),
        }


# ---------------------------------------------------------------------------
# OpenXR probing
# ---------------------------------------------------------------------------


class OpenXRProbe:
    """Best-effort capability discovery.

    This deliberately limits itself to extension probing, because robust session
    creation in Python depends on runtime/graphics bindings that vary a lot.
    """

    OPTIONAL_EXTENSIONS = {
        "XR_EXT_hand_tracking": "hand_tracking",
        "XR_EXT_eye_gaze_interaction": "eye_gaze",
        "XR_KHR_visibility_mask": "visibility_mask",
        "XR_EXT_debug_utils": "debug_utils",
        "XR_EXT_active_action_set_priority": "action_priorities",
    }

    GRAPHICS_EXTENSIONS = (
        "XR_KHR_opengl_enable",
        "XR_KHR_vulkan_enable",
        "XR_KHR_vulkan_enable2",
        "XR_KHR_D3D11_enable",
        "XR_KHR_D3D12_enable",
    )

    def _extension_name(self, obj: Any) -> str:
        if isinstance(obj, str):
            return obj
        for attr in ("extension_name", "extensionName", "name"):
            if hasattr(obj, attr):
                value = getattr(obj, attr)
                if isinstance(value, bytes):
                    return value.decode("utf-8", errors="ignore").strip("\x00")
                return str(value)
        return str(obj)

    def probe(self) -> RuntimeCaps:
        try:
            import xr  # type: ignore
        except Exception as exc:  # pragma: no cover - environment dependent
            return RuntimeCaps(
                backend="simulator",
                has_pyopenxr=False,
                notes=[f"pyopenxr unavailable: {exc!r}", "Running simulator backend."],
            )

        notes: List[str] = []
        names: List[str] = []
        try:
            raw = xr.enumerate_instance_extension_properties()
            names = sorted({self._extension_name(p) for p in raw})
        except Exception as exc:  # pragma: no cover - environment dependent
            return RuntimeCaps(
                backend="simulator",
                has_pyopenxr=True,
                notes=[
                    f"pyopenxr import succeeded but extension enumeration failed: {exc!r}",
                    "Falling back to simulator backend.",
                ],
            )

        caps = RuntimeCaps(
            backend="openxr-probed",
            has_pyopenxr=True,
            openxr_extensions=names,
            graphics_enable=[ext for ext in self.GRAPHICS_EXTENSIONS if ext in names],
            notes=notes,
        )
        for ext_name, field_name in self.OPTIONAL_EXTENSIONS.items():
            setattr(caps, field_name, ext_name in names)

        if not caps.graphics_enable:
            notes.append("No graphics-enable extension reported; real session creation may not be possible.")
        if caps.hand_tracking:
            notes.append("Hand tracking extension detected.")
        if caps.eye_gaze:
            notes.append("Eye gaze interaction extension detected.")
        if not notes:
            notes.append("OpenXR bindings detected. Use hello_xr / pyopenxr examples as the session/rendering reference path.")
        return caps


# ---------------------------------------------------------------------------
# Signal filtering / symbolic guidance
# ---------------------------------------------------------------------------


class NegativeLayerFilter:
    """Suppress contradictory, stale, or noisy guidance signals."""

    def __init__(self, alpha: float = 0.18, stale_after: float = 0.7) -> None:
        self.alpha = alpha
        self.stale_after = stale_after
        self._last_head: Optional[Pose] = None
        self._last_gaze: Optional[GazeState] = None
        self._last_t: Optional[float] = None
        self.removed_fraction: float = 0.0

    def filter_pose(self, pose: Pose) -> Pose:
        if self._last_head is None:
            self._last_head = pose
            return pose
        p = Vec3(
            lerp(self._last_head.position.x, pose.position.x, self.alpha),
            lerp(self._last_head.position.y, pose.position.y, self.alpha),
            lerp(self._last_head.position.z, pose.position.z, self.alpha),
        )
        f = Vec3(
            lerp(self._last_head.forward.x, pose.forward.x, self.alpha),
            lerp(self._last_head.forward.y, pose.forward.y, self.alpha),
            lerp(self._last_head.forward.z, pose.forward.z, self.alpha),
        ).normalized()
        confidence = clamp(0.6 * self._last_head.confidence + 0.4 * pose.confidence, 0.0, 1.0)
        filtered = Pose(position=p, forward=f, confidence=confidence)
        delta = (pose.position - filtered.position).length()
        self.removed_fraction = clamp(delta * 2.0, 0.0, 1.0)
        self._last_head = filtered
        return filtered

    def filter_gaze(self, gaze: GazeState, t: float) -> GazeState:
        if not gaze.tracked:
            return GazeState(tracked=False, direction=gaze.direction, stability=0.0)
        if self._last_gaze is None:
            self._last_gaze = gaze
            self._last_t = t
            return gaze
        dt = 0.0 if self._last_t is None else max(0.0, t - self._last_t)
        stale_penalty = 0.35 if dt > self.stale_after else 0.0
        d = Vec3(
            lerp(self._last_gaze.direction.x, gaze.direction.x, self.alpha),
            lerp(self._last_gaze.direction.y, gaze.direction.y, self.alpha),
            lerp(self._last_gaze.direction.z, gaze.direction.z, self.alpha),
        ).normalized()
        stability = clamp((0.7 * self._last_gaze.stability + 0.3 * gaze.stability) - stale_penalty, 0.0, 1.0)
        filtered = GazeState(tracked=True, direction=d, stability=stability)
        self._last_gaze = filtered
        self._last_t = t
        return filtered


class EnergyConverter:
    """Turns affect-like inputs into guidance energy and comfort policy."""

    def convert(self, affect: AffectSignal) -> Dict[str, float]:
        a = affect.normalized()
        safety_drag = 0.55 * a.pain + 0.35 * a.overload
        drive = 0.45 * (a.valence + 1.0) / 2.0 + 0.35 * a.arousal + 0.20 * a.clarity
        energy = clamp(drive - safety_drag * 0.8, 0.0, 1.0)
        comfort = clamp(1.0 - (0.75 * a.overload + 0.5 * a.pain), 0.0, 1.0)
        cue_density = clamp(0.25 + 0.65 * comfort + 0.15 * a.clarity, 0.1, 1.0)
        return {
            "energy": energy,
            "comfort": comfort,
            "cue_density": cue_density,
        }


class SymbolicGuidanceEngine:
    def __init__(self) -> None:
        self.filter = NegativeLayerFilter()
        self.energy = EnergyConverter()
        self.targets: List[WorldTarget] = [
            WorldTarget("echo_gate", Vec3(0.0, 1.4, -4.0), 1.0),
            WorldTarget("soft_ramp", Vec3(2.0, 1.2, -7.0), 0.7),
            WorldTarget("quiet_anchor", Vec3(-2.5, 1.3, -5.0), 0.65),
        ]

    def _best_target(self, head: Pose, gaze: GazeState) -> Tuple[Optional[WorldTarget], float]:
        best: Optional[WorldTarget] = None
        best_score = -1e9
        basis = gaze.direction.normalized() if gaze.tracked else head.forward.normalized()
        for target in self.targets:
            delta = target.position - head.position
            dist = max(0.001, delta.length())
            align = basis.dot(delta.normalized())
            score = target.importance * (1.25 * align + 0.55 / dist)
            if score > best_score:
                best = target
                best_score = score
        return best, best_score

    def _turn_bias(self, forward: Vec3, target_delta: Vec3) -> float:
        # crude horizontal signed angle proxy, enough for cue generation
        left_right = forward.x * target_delta.z - forward.z * target_delta.x
        return clamp(left_right, -1.0, 1.0)

    def step(
        self,
        t: float,
        head: Pose,
        left_hand: HandState,
        right_hand: HandState,
        gaze: GazeState,
        affect: AffectSignal,
    ) -> GuidanceFrame:
        head = self.filter.filter_pose(head)
        gaze = self.filter.filter_gaze(gaze, t)
        e = self.energy.convert(affect)
        target, score = self._best_target(head, gaze)
        notes: List[str] = []

        if target is None:
            return GuidanceFrame(
                t=t,
                mode=GuidanceMode.RECOVER,
                cue_word="hold",
                turn_bias=0.0,
                pulse=0.15,
                stability=0.0,
                energy=e["energy"],
                clutter_removed=self.filter.removed_fraction,
                target=None,
                notes=["No stable target."],
            )

        delta = target.position - head.position
        distance = delta.length()
        turn_bias = self._turn_bias(head.forward.normalized(), delta.normalized())
        hand_activity = max(left_hand.pinch, right_hand.pinch, left_hand.joint_energy, right_hand.joint_energy)
        stability = clamp(
            0.35 * head.confidence +
            0.35 * (gaze.stability if gaze.tracked else 0.5) +
            0.30 * e["comfort"],
            0.0,
            1.0,
        )

        if affect.pain > 0.45 or affect.overload > 0.7:
            mode = GuidanceMode.RECOVER
            cue_word = "soften"
            pulse = clamp(0.18 + 0.25 * (1.0 - e["comfort"]), 0.12, 0.55)
            notes.append("Recovery prioritised over dense cueing.")
        elif abs(turn_bias) > 0.42 or distance < 1.1:
            mode = GuidanceMode.ALERT
            cue_word = "pivot"
            pulse = clamp(0.5 + 0.3 * e["energy"] + 0.15 * hand_activity, 0.35, 1.0)
            notes.append("Strong turn or approach cue.")
        elif score > 0.55:
            mode = GuidanceMode.FOCUS
            cue_word = "trace"
            pulse = clamp(0.35 + 0.25 * e["energy"], 0.2, 0.85)
            notes.append("Target alignment stable.")
        else:
            mode = GuidanceMode.CALM
            cue_word = "glide"
            pulse = clamp(0.2 + 0.2 * e["energy"], 0.15, 0.6)
            notes.append("Low-friction guidance mode.")

        if self.filter.removed_fraction > 0.2:
            notes.append("Noise suppression active.")
        if gaze.tracked:
            notes.append("Eye-gaze steering enabled.")
        elif left_hand.tracked or right_hand.tracked:
            notes.append("Hand-led fallback steering enabled.")
        else:
            notes.append("Head-forward fallback steering enabled.")

        return GuidanceFrame(
            t=t,
            mode=mode,
            cue_word=cue_word,
            turn_bias=turn_bias,
            pulse=pulse,
            stability=stability,
            energy=e["energy"],
            clutter_removed=self.filter.removed_fraction,
            target=target.label,
            notes=notes,
        )


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------


@dataclass
class SensorFrame:
    t: float
    head: Pose
    left_hand: HandState
    right_hand: HandState
    gaze: GazeState
    affect: AffectSignal

    def to_dict(self) -> Dict[str, Any]:
        return {
            "t": round(self.t, 3),
            "head": self.head.to_dict(),
            "left_hand": self.left_hand.to_dict(),
            "right_hand": self.right_hand.to_dict(),
            "gaze": self.gaze.to_dict(),
            "affect": self.affect.to_dict(),
        }


class BaseBackend:
    def capabilities(self) -> RuntimeCaps:
        raise NotImplementedError

    def frames(self, seconds: float, hz: float) -> Iterable[SensorFrame]:
        raise NotImplementedError


class SimulatorBackend(BaseBackend):
    def __init__(self, seed: int = 7) -> None:
        self.rng = random.Random(seed)
        self._caps = RuntimeCaps(
            backend="simulator",
            has_pyopenxr=False,
            notes=[
                "Deterministic simulator backend active.",
                "Use this to tune guidance logic before wiring a real OpenXR session.",
            ],
        )

    def capabilities(self) -> RuntimeCaps:
        return self._caps

    def frames(self, seconds: float, hz: float) -> Iterable[SensorFrame]:
        n = max(1, int(seconds * hz))
        for i in range(n):
            t = i / hz
            sway = 0.08 * math.sin(t * 1.7)
            drift = 0.03 * math.sin(t * 0.43)
            forward = Vec3(0.18 * math.sin(t * 0.8), 0.0, -1.0).normalized()
            head = Pose(
                position=Vec3(drift, 1.62 + 0.01 * math.sin(t * 2.1), sway),
                forward=forward,
                confidence=clamp(0.88 - 0.08 * abs(math.sin(t * 0.6)), 0.4, 1.0),
            )
            lh = HandState(
                tracked=True,
                pinch=clamp(0.25 + 0.2 * math.sin(t * 1.3), 0.0, 1.0),
                aim=Pose(position=Vec3(-0.22, 1.35, -0.35), forward=Vec3(-0.1, 0.0, -1.0).normalized(), confidence=0.8),
                joint_energy=clamp(0.35 + 0.2 * math.sin(t * 2.2), 0.0, 1.0),
            )
            rh = HandState(
                tracked=True,
                pinch=clamp(0.2 + 0.25 * math.sin(t * 1.8 + 0.6), 0.0, 1.0),
                aim=Pose(position=Vec3(0.22, 1.33, -0.32), forward=Vec3(0.1, 0.02, -1.0).normalized(), confidence=0.8),
                joint_energy=clamp(0.3 + 0.25 * math.sin(t * 2.4 + 0.3), 0.0, 1.0),
            )
            gaze = GazeState(
                tracked=True,
                direction=Vec3(0.12 * math.sin(t * 0.7), 0.02 * math.sin(t * 1.1), -1.0).normalized(),
                stability=clamp(0.82 - 0.18 * abs(math.sin(t * 0.9)), 0.0, 1.0),
            )
            affect = AffectSignal(
                valence=0.4 + 0.2 * math.sin(t * 0.22),
                arousal=0.48 + 0.12 * math.sin(t * 0.5 + 0.2),
                pain=0.08 + 0.04 * max(0.0, math.sin(t * 0.31 + 2.2)),
                overload=0.12 + 0.08 * max(0.0, math.sin(t * 0.73 + 1.2)),
                clarity=0.74 + 0.12 * math.sin(t * 0.41),
            ).normalized()
            yield SensorFrame(t=t, head=head, left_hand=lh, right_hand=rh, gaze=gaze, affect=affect)


class ProbedOpenXRBackend(BaseBackend):
    """Uses OpenXR capability probing and currently feeds synthetic tracking.

    Why synthetic tracking here? Because robust, cross-runtime Python session creation
    still depends on graphics/runtime specifics. This backend therefore gives you a
    truthful bridge: real capability discovery + a stable simulation layer until you
    wire your own rendering/session code from hello_xr or pyopenxr examples.
    """

    def __init__(self, caps: RuntimeCaps, seed: int = 17) -> None:
        self._caps = caps
        self._sim = SimulatorBackend(seed=seed)

    def capabilities(self) -> RuntimeCaps:
        return self._caps

    def frames(self, seconds: float, hz: float) -> Iterable[SensorFrame]:
        for frame in self._sim.frames(seconds, hz):
            # shape the simulated data according to discovered capabilities
            if not self._caps.hand_tracking:
                frame.left_hand.tracked = False
                frame.right_hand.tracked = False
                frame.left_hand.pinch = 0.0
                frame.right_hand.pinch = 0.0
                frame.left_hand.joint_energy = 0.0
                frame.right_hand.joint_energy = 0.0
            if not self._caps.eye_gaze:
                frame.gaze.tracked = False
                frame.gaze.stability = 0.0
            yield frame


# ---------------------------------------------------------------------------
# App / logging
# ---------------------------------------------------------------------------


class JSONLLogger:
    def __init__(self, path: Optional[Path]) -> None:
        self.path = path
        self._fh = None
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = path.open("w", encoding="utf-8")

    def write(self, obj: Dict[str, Any]) -> None:
        if self._fh is None:
            return
        self._fh.write(json.dumps(obj, ensure_ascii=False) + "\n")
        self._fh.flush()

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None


class AegisGlyphApp:
    def __init__(self, backend: BaseBackend, log_path: Optional[Path]) -> None:
        self.backend = backend
        self.engine = SymbolicGuidanceEngine()
        self.logger = JSONLLogger(log_path)

    def run(self, seconds: float, hz: float, quiet: bool = False) -> None:
        caps = self.backend.capabilities()
        self.logger.write({"event": "capabilities", "caps": caps.to_dict()})
        if not quiet:
            print("== AegisGlyph OpenXR v2 ==")
            print(json.dumps(caps.to_dict(), indent=2))
            print()

        for frame in self.backend.frames(seconds=seconds, hz=hz):
            guide = self.engine.step(
                t=frame.t,
                head=frame.head,
                left_hand=frame.left_hand,
                right_hand=frame.right_hand,
                gaze=frame.gaze,
                affect=frame.affect,
            )
            event = {
                "event": "frame",
                "sensor": frame.to_dict(),
                "guidance": guide.to_dict(),
            }
            self.logger.write(event)
            if not quiet:
                print(
                    f"t={frame.t:5.2f}s  mode={guide.mode.value:7s}  cue={guide.cue_word:6s}  "
                    f"turn={guide.turn_bias:+.2f}  pulse={guide.pulse:.2f}  "
                    f"stability={guide.stability:.2f}  target={guide.target}"
                )

        self.logger.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def choose_backend(mode: str) -> BaseBackend:
    probe = OpenXRProbe()
    caps = probe.probe()
    if mode == "sim":
        return SimulatorBackend()
    if mode == "probe":
        return ProbedOpenXRBackend(caps) if caps.has_pyopenxr else SimulatorBackend()
    # auto
    if caps.has_pyopenxr:
        return ProbedOpenXRBackend(caps)
    return SimulatorBackend()


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="AegisGlyph OpenXR v2")
    p.add_argument("--mode", choices=["auto", "probe", "sim"], default="auto")
    p.add_argument("--seconds", type=float, default=8.0)
    p.add_argument("--hz", type=float, default=12.0)
    p.add_argument("--log", type=str, default="")
    p.add_argument("--quiet", action="store_true")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    backend = choose_backend(args.mode)
    log_path = Path(args.log).expanduser() if args.log else None
    app = AegisGlyphApp(backend=backend, log_path=log_path)
    app.run(seconds=max(0.5, args.seconds), hz=max(1.0, args.hz), quiet=args.quiet)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
