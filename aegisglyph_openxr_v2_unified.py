#!/usr/bin/env python3
"""
AegisGlyph OpenXR v2 Unified
============================

Desktop-first, accessibility-oriented XR guidance engine with:
- optional OpenXR capability probing via pyopenxr (if installed)
- deterministic simulator fallback when no XR runtime/bindings are available
- symbolic guidance synthesis for blind / low-vision navigation experiments
- negative-signal conversion into safer recovery guidance
- positive-to-positive reinforcement (calm/clarity preservation without runaway intensity)
- feelings-to-energy system for cue intensity and comfort shaping
- object awareness addon with trees, vehicles, doors, ramps, benches, paths, and landmarks
- blind-support cue synthesis (speech text, haptic rhythm, spatial audio hints)
- JSONL logging for later analysis

This remains honest about reality:
OpenXR is real; robust Python session/render integration is runtime/graphics dependent.
So this script focuses on a production-shaped architecture that runs now, with a clean seam
for a real OpenXR session later.
"""
from __future__ import annotations

import argparse
import dataclasses
from dataclasses import dataclass, field
from enum import Enum
import json
import math
from pathlib import Path
import random
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# Math / core helpers
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

    def horizontal(self) -> "Vec3":
        return Vec3(self.x, 0.0, self.z)

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
    confidence: float = 0.6  # 0..1
    curiosity: float = 0.4   # 0..1

    def normalized(self) -> "AffectSignal":
        return AffectSignal(
            valence=clamp(self.valence, -1.0, 1.0),
            arousal=clamp(self.arousal, 0.0, 1.0),
            pain=clamp(self.pain, 0.0, 1.0),
            overload=clamp(self.overload, 0.0, 1.0),
            clarity=clamp(self.clarity, 0.0, 1.0),
            confidence=clamp(self.confidence, 0.0, 1.0),
            curiosity=clamp(self.curiosity, 0.0, 1.0),
        )

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self.normalized())


class GuidanceMode(str, Enum):
    CALM = "calm"
    FOCUS = "focus"
    ALERT = "alert"
    RECOVER = "recover"


class ObjectKind(str, Enum):
    LANDMARK = "landmark"
    PATH = "path"
    TREE = "tree"
    VEHICLE = "vehicle"
    BUILDING = "building"
    DOOR = "door"
    BENCH = "bench"
    RAMP = "ramp"
    STAIRS = "stairs"
    WATER = "water"
    PERSON = "person"
    POLE = "pole"


class CuePriority(str, Enum):
    INFO = "info"
    GUIDE = "guide"
    WARNING = "warning"
    CRITICAL = "critical"


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
class WorldObject:
    label: str
    kind: ObjectKind
    position: Vec3
    radius: float = 0.45
    importance: float = 0.5
    safe_clearance: float = 0.8
    velocity: Vec3 = field(default_factory=Vec3)
    tags: List[str] = field(default_factory=list)
    speakable_name: Optional[str] = None

    def moved(self, dt: float) -> "WorldObject":
        return WorldObject(
            label=self.label,
            kind=self.kind,
            position=self.position + self.velocity * dt,
            radius=self.radius,
            importance=self.importance,
            safe_clearance=self.safe_clearance,
            velocity=self.velocity,
            tags=list(self.tags),
            speakable_name=self.speakable_name,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "kind": self.kind.value,
            "position": self.position.to_dict(),
            "radius": round(self.radius, 3),
            "importance": round(self.importance, 3),
            "safe_clearance": round(self.safe_clearance, 3),
            "velocity": self.velocity.to_dict(),
            "tags": list(self.tags),
        }


@dataclass
class SpatialCue:
    text: str
    azimuth: float
    distance: float
    priority: CuePriority
    haptic: str
    tone: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "azimuth": round(self.azimuth, 4),
            "distance": round(self.distance, 3),
            "priority": self.priority.value,
            "haptic": self.haptic,
            "tone": self.tone,
        }


@dataclass
class GuidanceFrame:
    t: float
    mode: GuidanceMode
    cue_word: str
    turn_bias: float
    pulse: float
    stability: float
    energy: float
    comfort: float
    clutter_removed: float
    target: Optional[str]
    nearest_object: Optional[str]
    notes: List[str]
    speech: List[str] = field(default_factory=list)
    haptics: List[str] = field(default_factory=list)
    spatial_audio: List[SpatialCue] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "t": round(self.t, 3),
            "mode": self.mode.value,
            "cue_word": self.cue_word,
            "turn_bias": round(self.turn_bias, 4),
            "pulse": round(self.pulse, 4),
            "stability": round(self.stability, 4),
            "energy": round(self.energy, 4),
            "comfort": round(self.comfort, 4),
            "clutter_removed": round(self.clutter_removed, 4),
            "target": self.target,
            "nearest_object": self.nearest_object,
            "notes": list(self.notes),
            "speech": list(self.speech),
            "haptics": list(self.haptics),
            "spatial_audio": [cue.to_dict() for cue in self.spatial_audio],
        }


@dataclass
class SensorFrame:
    t: float
    dt: float
    head: Pose
    left_hand: HandState
    right_hand: HandState
    gaze: GazeState
    affect: AffectSignal

    def to_dict(self) -> Dict[str, Any]:
        return {
            "t": round(self.t, 3),
            "dt": round(self.dt, 4),
            "head": self.head.to_dict(),
            "left_hand": self.left_hand.to_dict(),
            "right_hand": self.right_hand.to_dict(),
            "gaze": self.gaze.to_dict(),
            "affect": self.affect.to_dict(),
        }


# ---------------------------------------------------------------------------
# OpenXR probing
# ---------------------------------------------------------------------------


class OpenXRProbe:
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
        except Exception as exc:
            return RuntimeCaps(
                backend="simulator",
                has_pyopenxr=False,
                notes=[f"pyopenxr unavailable: {exc!r}", "Running simulator backend."],
            )

        try:
            raw = xr.enumerate_instance_extension_properties()
            names = sorted({self._extension_name(p) for p in raw})
        except Exception as exc:
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
            notes=[],
        )
        for ext_name, field_name in self.OPTIONAL_EXTENSIONS.items():
            setattr(caps, field_name, ext_name in names)

        if not caps.graphics_enable:
            caps.notes.append("No graphics-enable extension reported; real session creation may be limited.")
        if caps.hand_tracking:
            caps.notes.append("Hand tracking extension detected.")
        if caps.eye_gaze:
            caps.notes.append("Eye gaze interaction extension detected.")
        if not caps.notes:
            caps.notes.append("OpenXR bindings detected; use official hello_xr patterns for real session wiring.")
        return caps


# ---------------------------------------------------------------------------
# Filtering and affect conversion
# ---------------------------------------------------------------------------


class NegativeLayerFilter:
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
        self.removed_fraction = clamp(delta * 2.1, 0.0, 1.0)
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


class NegativeConverter:
    """Convert pain/overload/negative valence into safer recovery behaviour.

    This does not pretend harm is good. It transforms bad signal into a slower,
    clearer operating mode with more protection and less cue spam.
    """

    def convert(self, affect: AffectSignal) -> Dict[str, float]:
        a = affect.normalized()
        load = clamp(0.6 * a.overload + 0.55 * a.pain + 0.25 * max(0.0, -a.valence), 0.0, 1.0)
        recovery_energy = clamp(0.25 + 0.5 * load + 0.15 * a.clarity, 0.0, 1.0)
        shield = clamp(0.3 + 0.7 * load, 0.0, 1.0)
        simplification = clamp(0.2 + 0.8 * load, 0.0, 1.0)
        return {
            "negative_load": load,
            "recovery_energy": recovery_energy,
            "shield": shield,
            "simplification": simplification,
        }


class PositiveToPositiveModule:
    """Reinforce healthy positive state without turning the system into a rave toaster."""

    def reinforce(self, affect: AffectSignal) -> Dict[str, float]:
        a = affect.normalized()
        positive_core = clamp(
            0.35 * (a.valence + 1.0) / 2.0 + 0.25 * a.clarity + 0.2 * a.confidence + 0.2 * a.curiosity,
            0.0,
            1.0,
        )
        calm_gain = clamp(0.25 + 0.6 * positive_core - 0.25 * a.overload, 0.0, 1.0)
        focus_gain = clamp(0.2 + 0.6 * a.clarity + 0.2 * a.confidence, 0.0, 1.0)
        return {
            "positive_core": positive_core,
            "calm_gain": calm_gain,
            "focus_gain": focus_gain,
        }


class FeelingsToEnergySystem:
    def __init__(self) -> None:
        self.negative = NegativeConverter()
        self.positive = PositiveToPositiveModule()

    def convert(self, affect: AffectSignal) -> Dict[str, float]:
        a = affect.normalized()
        neg = self.negative.convert(a)
        pos = self.positive.reinforce(a)

        raw_drive = (
            0.30 * pos["positive_core"] +
            0.18 * pos["focus_gain"] +
            0.16 * pos["calm_gain"] +
            0.18 * a.arousal +
            0.18 * a.curiosity
        )
        drag = 0.45 * a.pain + 0.35 * a.overload + 0.2 * max(0.0, -a.valence)

        energy = clamp(raw_drive + 0.18 * neg["recovery_energy"] - 0.55 * drag, 0.0, 1.0)
        comfort = clamp(1.0 - (0.70 * a.overload + 0.55 * a.pain) + 0.15 * pos["calm_gain"], 0.0, 1.0)
        cue_density = clamp(0.20 + 0.40 * comfort + 0.20 * pos["focus_gain"] - 0.30 * neg["simplification"], 0.08, 1.0)
        resilience = clamp(0.25 + 0.35 * pos["positive_core"] + 0.40 * neg["shield"], 0.0, 1.0)

        result = {
            "energy": energy,
            "comfort": comfort,
            "cue_density": cue_density,
            "resilience": resilience,
        }
        result.update(neg)
        result.update(pos)
        return result


# ---------------------------------------------------------------------------
# World model / object addon
# ---------------------------------------------------------------------------


@dataclass
class NearbyObject:
    obj: WorldObject
    distance: float
    bearing: float
    closing_speed: float
    hazard: float
    is_ahead: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "obj": self.obj.to_dict(),
            "distance": round(self.distance, 3),
            "bearing": round(self.bearing, 4),
            "closing_speed": round(self.closing_speed, 4),
            "hazard": round(self.hazard, 4),
            "is_ahead": self.is_ahead,
        }


class WorldModel:
    def __init__(self, rng: random.Random) -> None:
        self.rng = rng
        self.objects: List[WorldObject] = self._default_objects()

    def _default_objects(self) -> List[WorldObject]:
        return [
            WorldObject("echo_gate", ObjectKind.LANDMARK, Vec3(0.0, 1.4, -5.4), radius=0.8, importance=1.0, safe_clearance=1.0, speakable_name="echo gate"),
            WorldObject("main_path", ObjectKind.PATH, Vec3(0.0, 0.0, -4.0), radius=2.2, importance=0.9, safe_clearance=1.6, speakable_name="main path"),
            WorldObject("oak_tree_left", ObjectKind.TREE, Vec3(-2.2, 0.0, -3.8), radius=0.55, importance=0.25, safe_clearance=0.9, speakable_name="tree left"),
            WorldObject("pine_tree_right", ObjectKind.TREE, Vec3(2.4, 0.0, -6.0), radius=0.6, importance=0.25, safe_clearance=1.0, speakable_name="tree right"),
            WorldObject("bench_north", ObjectKind.BENCH, Vec3(-1.5, 0.0, -5.5), radius=0.5, importance=0.35, safe_clearance=0.8, speakable_name="bench"),
            WorldObject("door_gallery", ObjectKind.DOOR, Vec3(0.7, 1.0, -9.0), radius=0.6, importance=0.85, safe_clearance=0.8, speakable_name="gallery door"),
            WorldObject("ramp_access", ObjectKind.RAMP, Vec3(1.8, 0.3, -7.2), radius=0.9, importance=0.65, safe_clearance=1.0, speakable_name="access ramp"),
            WorldObject("stairs_warning", ObjectKind.STAIRS, Vec3(-2.8, -0.6, -7.5), radius=0.9, importance=0.75, safe_clearance=1.4, speakable_name="stairs"),
            WorldObject("water_feature", ObjectKind.WATER, Vec3(3.4, -0.2, -4.4), radius=1.4, importance=0.6, safe_clearance=1.6, speakable_name="water feature"),
            WorldObject("lamp_pole", ObjectKind.POLE, Vec3(1.4, 0.0, -2.8), radius=0.25, importance=0.5, safe_clearance=0.75, speakable_name="pole"),
            WorldObject("vehicle_shuttle", ObjectKind.VEHICLE, Vec3(-4.0, 0.0, -6.8), radius=1.0, importance=0.95, safe_clearance=2.2, velocity=Vec3(0.55, 0.0, 0.0), speakable_name="shuttle vehicle"),
            WorldObject("visitor", ObjectKind.PERSON, Vec3(2.8, 0.0, -5.0), radius=0.35, importance=0.45, safe_clearance=0.9, velocity=Vec3(-0.12, 0.0, -0.02), speakable_name="person"),
            WorldObject("north_building", ObjectKind.BUILDING, Vec3(0.0, 1.0, -12.5), radius=4.0, importance=0.55, safe_clearance=2.0, speakable_name="building"),
        ]

    def update(self, dt: float) -> None:
        updated: List[WorldObject] = []
        for obj in self.objects:
            moved = obj.moved(dt)
            if obj.kind == ObjectKind.VEHICLE:
                if moved.position.x > 4.2:
                    moved.position.x = -4.2
                if moved.position.x < -4.2:
                    moved.position.x = 4.2
            if obj.kind == ObjectKind.PERSON:
                if moved.position.z < -8.0 or moved.position.z > -3.6:
                    moved.velocity = Vec3(moved.velocity.x, 0.0, -moved.velocity.z)
            updated.append(moved)
        self.objects = updated

    def nearby(self, head: Pose, max_distance: float = 12.0) -> List[NearbyObject]:
        forward = head.forward.horizontal().normalized()
        if forward.length() <= 1e-6:
            forward = Vec3(0.0, 0.0, -1.0)
        right = Vec3(-forward.z, 0.0, forward.x)
        out: List[NearbyObject] = []
        for obj in self.objects:
            delta = obj.position - head.position
            horiz = delta.horizontal()
            dist = max(0.001, horiz.length())
            if dist > max_distance:
                continue
            rel = horiz.normalized()
            bearing = math.atan2(rel.dot(right), rel.dot(forward))
            radial = max(0.0, dist - obj.radius)
            closing_speed = max(0.0, -(obj.velocity.horizontal().dot(rel)))

            kind_base = {
                ObjectKind.VEHICLE: 1.0,
                ObjectKind.STAIRS: 0.95,
                ObjectKind.WATER: 0.8,
                ObjectKind.POLE: 0.65,
                ObjectKind.TREE: 0.45,
                ObjectKind.BENCH: 0.35,
                ObjectKind.PERSON: 0.35,
                ObjectKind.DOOR: 0.2,
                ObjectKind.RAMP: 0.15,
                ObjectKind.PATH: 0.05,
                ObjectKind.LANDMARK: 0.04,
                ObjectKind.BUILDING: 0.1,
            }.get(obj.kind, 0.2)
            clearance_gap = max(0.0, obj.safe_clearance - radial)
            hazard = clamp(kind_base * (1.6 / (dist + 0.45)) + 0.18 * closing_speed + 0.3 * clearance_gap, 0.0, 1.5)
            out.append(
                NearbyObject(
                    obj=obj,
                    distance=dist,
                    bearing=bearing,
                    closing_speed=closing_speed,
                    hazard=hazard,
                    is_ahead=abs(bearing) < 0.85,
                )
            )
        out.sort(key=lambda n: (-(n.hazard), n.distance))
        return out


# ---------------------------------------------------------------------------
# Blind support / spatial cues
# ---------------------------------------------------------------------------


class BlindSupportSystem:
    def relative_words(self, bearing: float) -> str:
        deg = math.degrees(bearing)
        if -18 <= deg <= 18:
            return "ahead"
        if 18 < deg <= 65:
            return "front-right"
        if 65 < deg <= 120:
            return "right"
        if deg > 120:
            return "back-right"
        if -65 <= deg < -18:
            return "front-left"
        if -120 <= deg < -65:
            return "left"
        return "back-left"

    def distance_words(self, distance: float) -> str:
        if distance < 1.0:
            return "very close"
        if distance < 2.0:
            return "close"
        if distance < 4.0:
            return "near"
        if distance < 7.0:
            return "ahead"
        return "far"

    def haptic_pattern(self, priority: CuePriority, pulse: float) -> str:
        if priority == CuePriority.CRITICAL:
            return f"rapid:{pulse:.2f}:triple"
        if priority == CuePriority.WARNING:
            return f"medium:{pulse:.2f}:double"
        if priority == CuePriority.GUIDE:
            return f"steady:{pulse:.2f}:single"
        return f"soft:{pulse:.2f}:single"

    def tone_name(self, priority: CuePriority, bearing: float) -> str:
        side = "center"
        if bearing < -0.2:
            side = "left"
        elif bearing > 0.2:
            side = "right"
        if priority == CuePriority.CRITICAL:
            return f"low_alarm_{side}"
        if priority == CuePriority.WARNING:
            return f"short_ping_{side}"
        if priority == CuePriority.GUIDE:
            return f"guide_chime_{side}"
        return f"soft_air_{side}"

    def make_object_cue(self, nearby: NearbyObject, pulse: float) -> SpatialCue:
        priority = CuePriority.INFO
        if nearby.hazard > 1.05:
            priority = CuePriority.CRITICAL
        elif nearby.hazard > 0.6:
            priority = CuePriority.WARNING
        elif nearby.obj.kind in {ObjectKind.DOOR, ObjectKind.RAMP, ObjectKind.PATH, ObjectKind.LANDMARK}:
            priority = CuePriority.GUIDE

        name = nearby.obj.speakable_name or nearby.obj.label.replace("_", " ")
        text = f"{name} {self.relative_words(nearby.bearing)}, {self.distance_words(nearby.distance)}"
        return SpatialCue(
            text=text,
            azimuth=nearby.bearing,
            distance=nearby.distance,
            priority=priority,
            haptic=self.haptic_pattern(priority, pulse),
            tone=self.tone_name(priority, nearby.bearing),
        )

    def compose(self, guide: GuidanceFrame, nearby: Sequence[NearbyObject]) -> GuidanceFrame:
        speech: List[str] = []
        haptics: List[str] = []
        audio: List[SpatialCue] = []

        if guide.mode == GuidanceMode.RECOVER:
            speech.append("Recovery mode. Slow down and hold your line.")
            haptics.append("recovery:slow:double")
        elif guide.mode == GuidanceMode.ALERT:
            speech.append("Alert. Adjust direction now.")
            haptics.append("alert:medium:double")
        elif guide.mode == GuidanceMode.FOCUS:
            speech.append("Focus. Follow the guide cue.")
            haptics.append("focus:steady:single")
        else:
            speech.append("Calm guidance active.")
            haptics.append("calm:soft:single")

        for item in nearby[:3]:
            cue = self.make_object_cue(item, guide.pulse)
            audio.append(cue)
            if cue.priority in {CuePriority.CRITICAL, CuePriority.WARNING, CuePriority.GUIDE}:
                speech.append(cue.text)
                haptics.append(cue.haptic)

        dedup_speech: List[str] = []
        seen = set()
        for line in speech:
            if line not in seen:
                dedup_speech.append(line)
                seen.add(line)
        guide.speech = dedup_speech[:5]
        guide.haptics = haptics[:5]
        guide.spatial_audio = audio[:4]
        return guide


# ---------------------------------------------------------------------------
# Guidance engine
# ---------------------------------------------------------------------------


class SymbolicGuidanceEngine:
    def __init__(self, world: WorldModel) -> None:
        self.filter = NegativeLayerFilter()
        self.energy = FeelingsToEnergySystem()
        self.world = world
        self.blind_support = BlindSupportSystem()

    def _steering_basis(self, head: Pose, gaze: GazeState, left: HandState, right: HandState) -> Tuple[Vec3, str]:
        if gaze.tracked and gaze.stability > 0.25:
            return gaze.direction.normalized(), "gaze"
        if right.tracked and right.aim.confidence >= left.aim.confidence and right.joint_energy > 0.12:
            return right.aim.forward.normalized(), "right-hand"
        if left.tracked and left.joint_energy > 0.12:
            return left.aim.forward.normalized(), "left-hand"
        return head.forward.normalized(), "head"

    def _turn_bias(self, forward: Vec3, target_delta: Vec3) -> float:
        left_right = forward.x * target_delta.z - forward.z * target_delta.x
        return clamp(left_right, -1.0, 1.0)

    def _choose_target(self, head: Pose, basis: Vec3, nearby: Sequence[NearbyObject]) -> Tuple[Optional[WorldObject], float]:
        best_obj: Optional[WorldObject] = None
        best_score = -1e9
        for item in nearby:
            obj = item.obj
            if obj.kind not in {ObjectKind.LANDMARK, ObjectKind.PATH, ObjectKind.DOOR, ObjectKind.RAMP, ObjectKind.BUILDING}:
                continue
            align = basis.dot((obj.position - head.position).horizontal().normalized())
            path_bonus = 0.18 if obj.kind == ObjectKind.PATH else 0.0
            score = obj.importance * (1.15 * align + 0.55 / (item.distance + 0.2) + path_bonus) - 0.35 * item.hazard
            if score > best_score:
                best_score = score
                best_obj = obj
        return best_obj, best_score

    def step(
        self,
        frame: SensorFrame,
    ) -> Tuple[GuidanceFrame, List[NearbyObject]]:
        self.world.update(frame.dt)
        head = self.filter.filter_pose(frame.head)
        gaze = self.filter.filter_gaze(frame.gaze, frame.t)
        affect_energy = self.energy.convert(frame.affect)
        nearby = self.world.nearby(head)
        basis, steering_mode = self._steering_basis(head, gaze, frame.left_hand, frame.right_hand)
        target, target_score = self._choose_target(head, basis, nearby)

        notes: List[str] = []
        strongest_hazard = nearby[0] if nearby else None
        nearest_object = strongest_hazard.obj.label if strongest_hazard else None

        if target is None:
            guide = GuidanceFrame(
                t=frame.t,
                mode=GuidanceMode.RECOVER,
                cue_word="hold",
                turn_bias=0.0,
                pulse=0.18,
                stability=0.0,
                energy=affect_energy["energy"],
                comfort=affect_energy["comfort"],
                clutter_removed=self.filter.removed_fraction,
                target=None,
                nearest_object=nearest_object,
                notes=["No stable target available."],
            )
            return self.blind_support.compose(guide, nearby), nearby

        target_delta = (target.position - head.position).horizontal()
        turn_bias = self._turn_bias(head.forward.horizontal().normalized(), target_delta.normalized())
        distance = max(0.01, target_delta.length())
        danger = strongest_hazard.hazard if strongest_hazard else 0.0
        hand_activity = max(frame.left_hand.pinch, frame.right_hand.pinch, frame.left_hand.joint_energy, frame.right_hand.joint_energy)

        stability = clamp(
            0.25 * head.confidence +
            0.20 * (gaze.stability if gaze.tracked else 0.5) +
            0.20 * affect_energy["comfort"] +
            0.15 * affect_energy["resilience"] +
            0.20 * frame.affect.confidence,
            0.0,
            1.0,
        )

        if frame.affect.pain > 0.45 or frame.affect.overload > 0.72 or danger > 1.0:
            mode = GuidanceMode.RECOVER
            cue_word = "soften"
            pulse = clamp(0.18 + 0.22 * affect_energy["negative_load"] + 0.18 * danger, 0.15, 0.62)
            notes.append("Recovery prioritised: simplify cues and protect movement.")
        elif danger > 0.62 or abs(turn_bias) > 0.45 or distance < 1.2:
            mode = GuidanceMode.ALERT
            cue_word = "pivot"
            pulse = clamp(0.48 + 0.22 * affect_energy["energy"] + 0.12 * hand_activity + 0.08 * danger, 0.35, 1.0)
            notes.append("Hazard or sharp turn requires quick correction.")
        elif target_score > 0.42:
            mode = GuidanceMode.FOCUS
            cue_word = "trace"
            pulse = clamp(0.34 + 0.22 * affect_energy["energy"] + 0.12 * affect_energy["focus_gain"], 0.2, 0.85)
            notes.append("Stable target lock established.")
        else:
            mode = GuidanceMode.CALM
            cue_word = "glide"
            pulse = clamp(0.20 + 0.14 * affect_energy["calm_gain"] + 0.10 * affect_energy["energy"], 0.15, 0.55)
            notes.append("Low-friction navigation mode.")

        if self.filter.removed_fraction > 0.2:
            notes.append("Noise suppression active.")
        notes.append(f"Steering basis: {steering_mode}.")
        if strongest_hazard and strongest_hazard.hazard > 0.55:
            notes.append(f"Hazard priority: {strongest_hazard.obj.kind.value}.")

        guide = GuidanceFrame(
            t=frame.t,
            mode=mode,
            cue_word=cue_word,
            turn_bias=turn_bias,
            pulse=pulse,
            stability=stability,
            energy=affect_energy["energy"],
            comfort=affect_energy["comfort"],
            clutter_removed=self.filter.removed_fraction,
            target=target.label,
            nearest_object=nearest_object,
            notes=notes,
        )
        guide = self.blind_support.compose(guide, nearby)
        return guide, nearby


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------


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
                "Use this to tune guidance and accessibility logic before wiring a real OpenXR session.",
            ],
        )

    def capabilities(self) -> RuntimeCaps:
        return self._caps

    def frames(self, seconds: float, hz: float) -> Iterable[SensorFrame]:
        n = max(1, int(seconds * hz))
        prev_t = 0.0
        for i in range(n):
            t = i / hz
            dt = 0.0 if i == 0 else t - prev_t
            prev_t = t
            sway = 0.08 * math.sin(t * 1.7)
            drift = 0.04 * math.sin(t * 0.43)
            forward = Vec3(0.18 * math.sin(t * 0.8), 0.0, -1.0).normalized()
            head = Pose(
                position=Vec3(drift, 1.62 + 0.01 * math.sin(t * 2.1), sway),
                forward=forward,
                confidence=clamp(0.88 - 0.08 * abs(math.sin(t * 0.6)), 0.4, 1.0),
            )
            lh = HandState(
                tracked=True,
                pinch=clamp(0.22 + 0.22 * math.sin(t * 1.3), 0.0, 1.0),
                aim=Pose(position=Vec3(-0.22, 1.35, -0.35), forward=Vec3(-0.1, 0.0, -1.0).normalized(), confidence=0.8),
                joint_energy=clamp(0.35 + 0.22 * math.sin(t * 2.2), 0.0, 1.0),
            )
            rh = HandState(
                tracked=True,
                pinch=clamp(0.18 + 0.25 * math.sin(t * 1.8 + 0.6), 0.0, 1.0),
                aim=Pose(position=Vec3(0.22, 1.33, -0.32), forward=Vec3(0.1, 0.02, -1.0).normalized(), confidence=0.82),
                joint_energy=clamp(0.28 + 0.25 * math.sin(t * 2.4 + 0.3), 0.0, 1.0),
            )
            gaze = GazeState(
                tracked=True,
                direction=Vec3(0.12 * math.sin(t * 0.7), 0.02 * math.sin(t * 1.1), -1.0).normalized(),
                stability=clamp(0.82 - 0.18 * abs(math.sin(t * 0.9)), 0.0, 1.0),
            )
            affect = AffectSignal(
                valence=0.38 + 0.24 * math.sin(t * 0.22),
                arousal=0.46 + 0.14 * math.sin(t * 0.5 + 0.2),
                pain=0.06 + 0.05 * max(0.0, math.sin(t * 0.31 + 2.2)),
                overload=0.10 + 0.12 * max(0.0, math.sin(t * 0.73 + 1.2)),
                clarity=0.74 + 0.12 * math.sin(t * 0.41),
                confidence=0.65 + 0.12 * math.sin(t * 0.35 + 0.7),
                curiosity=0.45 + 0.16 * math.sin(t * 0.27),
            ).normalized()
            yield SensorFrame(t=t, dt=dt, head=head, left_hand=lh, right_hand=rh, gaze=gaze, affect=affect)


class ProbedOpenXRBackend(BaseBackend):
    def __init__(self, caps: RuntimeCaps, seed: int = 17) -> None:
        self._caps = caps
        self._sim = SimulatorBackend(seed=seed)

    def capabilities(self) -> RuntimeCaps:
        return self._caps

    def frames(self, seconds: float, hz: float) -> Iterable[SensorFrame]:
        for frame in self._sim.frames(seconds, hz):
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
# Logging / app
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
    def __init__(self, backend: BaseBackend, log_path: Optional[Path], seed: int = 7) -> None:
        self.backend = backend
        self.world = WorldModel(random.Random(seed))
        self.engine = SymbolicGuidanceEngine(self.world)
        self.logger = JSONLLogger(log_path)

    def run(self, seconds: float, hz: float, quiet: bool = False) -> None:
        caps = self.backend.capabilities()
        self.logger.write({"event": "capabilities", "caps": caps.to_dict()})
        if not quiet:
            print("== AegisGlyph OpenXR v2 Unified ==")
            print(json.dumps(caps.to_dict(), indent=2))
            print()

        for frame in self.backend.frames(seconds=seconds, hz=hz):
            guide, nearby = self.engine.step(frame)
            event = {
                "event": "frame",
                "sensor": frame.to_dict(),
                "guidance": guide.to_dict(),
                "nearby": [n.to_dict() for n in nearby[:6]],
            }
            self.logger.write(event)
            if not quiet:
                lead_speech = guide.speech[0] if guide.speech else ""
                print(
                    f"t={frame.t:5.2f}s  mode={guide.mode.value:7s}  cue={guide.cue_word:6s}  "
                    f"turn={guide.turn_bias:+.2f}  pulse={guide.pulse:.2f}  "
                    f"stability={guide.stability:.2f}  target={guide.target or '-':12s}  "
                    f"near={guide.nearest_object or '-':14s}  say={lead_speech}"
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
    if caps.has_pyopenxr:
        return ProbedOpenXRBackend(caps)
    return SimulatorBackend()


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="AegisGlyph OpenXR v2 Unified")
    p.add_argument("--mode", choices=["auto", "probe", "sim"], default="auto")
    p.add_argument("--seconds", type=float, default=8.0)
    p.add_argument("--hz", type=float, default=12.0)
    p.add_argument("--log", type=str, default="")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--quiet", action="store_true")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    backend = choose_backend(args.mode)
    log_path = Path(args.log).expanduser() if args.log else None
    app = AegisGlyphApp(backend=backend, log_path=log_path, seed=args.seed)
    app.run(seconds=max(0.5, args.seconds), hz=max(1.0, args.hz), quiet=args.quiet)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
