#!/usr/bin/env python3
"""
AegisGlyph OpenXR v3 Evolved
============================

An advanced, single-file, desktop-first OpenXR architecture prototype for
accessibility-oriented XR guidance. This script is an evolutionary rewrite of
AegisGlyph OpenXR v2 Unified: it keeps the original spirit (blind/low-vision
support, feelings-to-energy conversion, negative-layer filtering, object-aware
navigation) while reorganizing the whole system around a more faithful OpenXR
shape:

- runtime discovery and capability registry
- explicit instance -> system -> session lifecycle model
- graphics binding abstraction (OpenGL / Vulkan / D3D11 / D3D12 descriptors)
- action system abstraction (action sets, interaction profiles, priorities)
- extension registry (hand tracking, eye gaze, passthrough, scene understanding)
- scene graph and world object model
- plugin/module host with ordered frame stages
- performance instrumentation hooks
- security boundary and telemetry redaction model
- accessibility bus for speech, haptics, and spatial cue synthesis
- simulator backend now, real OpenXR bootstrap seam later

This file is deliberately honest:
- It runs today with the Python standard library.
- It can optionally probe pyopenxr if installed.
- It does not pretend to ship a production compositor or headset renderer.
- It is structured so a real OpenXR session can be inserted without rewriting
  the guidance stack.

The design is suitable for:
- architecture prototyping
- guidance logic experimentation
- simulator-driven accessibility testing
- future real-runtime integration
"""
from __future__ import annotations

import argparse
from collections import deque
import dataclasses
from dataclasses import dataclass, field
from enum import Enum, auto
import json
import math
import os
from pathlib import Path
import platform
import random
import statistics
import sys
import time
from typing import Any, Dict, Iterable, Iterator, List, MutableMapping, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# Small math helpers
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

    def __mul__(self, scalar: float) -> "Vec3":
        return Vec3(self.x * scalar, self.y * scalar, self.z * scalar)

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

    def cross(self, other: "Vec3") -> "Vec3":
        return Vec3(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )

    def horizontal(self) -> "Vec3":
        return Vec3(self.x, 0.0, self.z)

    def distance_to(self, other: "Vec3") -> float:
        return (self - other).length()

    def to_dict(self) -> Dict[str, float]:
        return {"x": round(self.x, 5), "y": round(self.y, 5), "z": round(self.z, 5)}


@dataclass
class Transform:
    position: Vec3 = field(default_factory=Vec3)
    forward: Vec3 = field(default_factory=lambda: Vec3(0.0, 0.0, -1.0))
    up: Vec3 = field(default_factory=lambda: Vec3(0.0, 1.0, 0.0))
    scale: Vec3 = field(default_factory=lambda: Vec3(1.0, 1.0, 1.0))

    def right(self) -> Vec3:
        right = self.forward.cross(self.up)
        if right.length() <= 1e-9:
            return Vec3(1.0, 0.0, 0.0)
        return right.normalized()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "position": self.position.to_dict(),
            "forward": self.forward.to_dict(),
            "up": self.up.to_dict(),
            "scale": self.scale.to_dict(),
        }


# ---------------------------------------------------------------------------
# OpenXR-shaped enums and configuration types
# ---------------------------------------------------------------------------


class GraphicsAPI(str, Enum):
    NONE = "none"
    OPENGL = "opengl"
    VULKAN = "vulkan"
    D3D11 = "d3d11"
    D3D12 = "d3d12"


class SessionState(str, Enum):
    IDLE = "idle"
    INSTANCE_READY = "instance_ready"
    SYSTEM_READY = "system_ready"
    SESSION_READY = "session_ready"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    FAILED = "failed"


class CuePriority(str, Enum):
    INFO = "info"
    GUIDE = "guide"
    WARNING = "warning"
    CRITICAL = "critical"


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
    WALL = "wall"
    SIGNAL = "signal"


class HazardKind(str, Enum):
    COLLISION = "collision"
    DROP = "drop"
    MOVING_OBJECT = "moving_object"
    STAIRS = "stairs"
    WATER = "water"
    CROWD = "crowd"
    UNKNOWN = "unknown"


class ActionType(str, Enum):
    BOOLEAN = "boolean"
    FLOAT = "float"
    VECTOR2 = "vector2"
    POSE = "pose"
    HAPTIC = "haptic"


class ModuleStage(str, Enum):
    PRE_FRAME = "pre_frame"
    BUILD_SCENE = "build_scene"
    ANALYZE = "analyze"
    GUIDE = "guide"
    ACCESSIBILITY = "accessibility"
    POST_FRAME = "post_frame"


class TelemetryClass(str, Enum):
    PUBLIC = "public"
    OPERATIONAL = "operational"
    SENSITIVE = "sensitive"
    HIGHLY_SENSITIVE = "highly_sensitive"


# ---------------------------------------------------------------------------
# Runtime discovery and graphics binding model
# ---------------------------------------------------------------------------


@dataclass
class RuntimeLocation:
    source: str
    manifest_path: str

    def to_dict(self) -> Dict[str, str]:
        return dataclasses.asdict(self)


@dataclass
class GraphicsBindingDescriptor:
    api: GraphicsAPI
    extension_name: str
    platform_hint: str
    description: str

    def to_dict(self) -> Dict[str, str]:
        return dataclasses.asdict(self)


@dataclass
class ActionBinding:
    path: str
    comment: str = ""

    def to_dict(self) -> Dict[str, str]:
        return dataclasses.asdict(self)


@dataclass
class ActionDefinition:
    name: str
    localized_name: str
    action_type: ActionType
    semantic_role: str
    bindings: List[ActionBinding] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "localized_name": self.localized_name,
            "action_type": self.action_type.value,
            "semantic_role": self.semantic_role,
            "bindings": [b.to_dict() for b in self.bindings],
        }


@dataclass
class ActionSetDefinition:
    name: str
    localized_name: str
    priority: int
    actions: List[ActionDefinition] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "localized_name": self.localized_name,
            "priority": self.priority,
            "actions": [a.to_dict() for a in self.actions],
        }


@dataclass
class InteractionProfile:
    profile_path: str
    purpose: str
    supported_action_names: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass
class ExtensionRegistry:
    available_extensions: List[str] = field(default_factory=list)
    graphics_bindings: List[GraphicsBindingDescriptor] = field(default_factory=list)
    hand_tracking: bool = False
    eye_gaze: bool = False
    visibility_mask: bool = False
    debug_utils: bool = False
    action_priorities: bool = False
    passthrough: bool = False
    scene_understanding: bool = False
    anchors: bool = False
    notes: List[str] = field(default_factory=list)

    def choose_preferred_graphics(self) -> GraphicsBindingDescriptor:
        preferred_order = [GraphicsAPI.VULKAN, GraphicsAPI.OPENGL, GraphicsAPI.D3D12, GraphicsAPI.D3D11]
        for wanted in preferred_order:
            for desc in self.graphics_bindings:
                if desc.api == wanted:
                    return desc
        return GraphicsBindingDescriptor(
            api=GraphicsAPI.NONE,
            extension_name="",
            platform_hint="portable",
            description="No graphics binding extension detected; stay in simulator/probe mode.",
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "available_extensions": list(self.available_extensions),
            "graphics_bindings": [g.to_dict() for g in self.graphics_bindings],
            "hand_tracking": self.hand_tracking,
            "eye_gaze": self.eye_gaze,
            "visibility_mask": self.visibility_mask,
            "debug_utils": self.debug_utils,
            "action_priorities": self.action_priorities,
            "passthrough": self.passthrough,
            "scene_understanding": self.scene_understanding,
            "anchors": self.anchors,
            "notes": list(self.notes),
        }


@dataclass
class OpenXRSystemModel:
    runtime_locations: List[RuntimeLocation] = field(default_factory=list)
    runtime_selected: Optional[RuntimeLocation] = None
    pyopenxr_available: bool = False
    extension_registry: ExtensionRegistry = field(default_factory=ExtensionRegistry)
    graphics_binding: GraphicsBindingDescriptor = field(
        default_factory=lambda: GraphicsBindingDescriptor(GraphicsAPI.NONE, "", "portable", "No binding")
    )
    lifecycle_state: SessionState = SessionState.IDLE
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "runtime_locations": [r.to_dict() for r in self.runtime_locations],
            "runtime_selected": None if self.runtime_selected is None else self.runtime_selected.to_dict(),
            "pyopenxr_available": self.pyopenxr_available,
            "extension_registry": self.extension_registry.to_dict(),
            "graphics_binding": self.graphics_binding.to_dict(),
            "lifecycle_state": self.lifecycle_state.value,
            "notes": list(self.notes),
        }


class OpenXRRuntimeLocator:
    """Cross-platform runtime manifest discovery.

    This is a design-level helper: it follows the standard loader ideas without
    pretending to replace the loader itself.
    """

    def discover(self) -> List[RuntimeLocation]:
        found: List[RuntimeLocation] = []
        env_runtime = os.environ.get("XR_RUNTIME_JSON", "").strip()
        if env_runtime:
            found.append(RuntimeLocation(source="env:XR_RUNTIME_JSON", manifest_path=env_runtime))

        if os.name == "nt":
            try:
                import winreg  # type: ignore

                key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Khronos\OpenXR\1")
                value, _ = winreg.QueryValueEx(key, "ActiveRuntime")
                if value:
                    found.append(RuntimeLocation(source="windows-registry", manifest_path=str(value)))
            except Exception:
                pass
        else:
            candidates = []
            xdg_home = os.environ.get("XDG_CONFIG_HOME")
            home = os.path.expanduser("~")
            if xdg_home:
                candidates.append(Path(xdg_home) / "openxr/1/active_runtime.json")
            candidates.append(Path(home) / ".config/openxr/1/active_runtime.json")
            candidates.append(Path("/etc/xdg/openxr/1/active_runtime.json"))
            for candidate in candidates:
                if candidate.exists():
                    found.append(RuntimeLocation(source="xdg-config", manifest_path=str(candidate)))

        dedup: Dict[str, RuntimeLocation] = {}
        for item in found:
            dedup[item.manifest_path] = item
        return list(dedup.values())


class OpenXRCapabilityProbe:
    GRAPHICS = {
        "XR_KHR_vulkan_enable": GraphicsBindingDescriptor(GraphicsAPI.VULKAN, "XR_KHR_vulkan_enable", "cross-platform", "Vulkan graphics binding"),
        "XR_KHR_vulkan_enable2": GraphicsBindingDescriptor(GraphicsAPI.VULKAN, "XR_KHR_vulkan_enable2", "cross-platform", "Vulkan 2 graphics binding"),
        "XR_KHR_opengl_enable": GraphicsBindingDescriptor(GraphicsAPI.OPENGL, "XR_KHR_opengl_enable", "cross-platform", "OpenGL graphics binding"),
        "XR_KHR_D3D11_enable": GraphicsBindingDescriptor(GraphicsAPI.D3D11, "XR_KHR_D3D11_enable", "windows", "Direct3D 11 graphics binding"),
        "XR_KHR_D3D12_enable": GraphicsBindingDescriptor(GraphicsAPI.D3D12, "XR_KHR_D3D12_enable", "windows", "Direct3D 12 graphics binding"),
    }

    def _extension_name(self, item: Any) -> str:
        if isinstance(item, str):
            return item
        for attr in ("extension_name", "extensionName", "name"):
            if hasattr(item, attr):
                value = getattr(item, attr)
                if isinstance(value, bytes):
                    return value.decode("utf-8", errors="ignore").strip("\x00")
                return str(value)
        return str(item)

    def probe(self) -> Tuple[bool, ExtensionRegistry, List[str]]:
        notes: List[str] = []
        try:
            import xr  # type: ignore
        except Exception as exc:
            notes.append(f"pyopenxr unavailable: {exc!r}")
            return False, ExtensionRegistry(notes=notes), notes

        try:
            raw = xr.enumerate_instance_extension_properties()
            names = sorted({self._extension_name(x) for x in raw})
        except Exception as exc:
            notes.append(f"pyopenxr import succeeded but extension probe failed: {exc!r}")
            return True, ExtensionRegistry(notes=notes), notes

        registry = ExtensionRegistry(available_extensions=names)
        for ext_name, descriptor in self.GRAPHICS.items():
            if ext_name in names:
                registry.graphics_bindings.append(descriptor)

        registry.hand_tracking = "XR_EXT_hand_tracking" in names
        registry.eye_gaze = "XR_EXT_eye_gaze_interaction" in names
        registry.visibility_mask = "XR_KHR_visibility_mask" in names
        registry.debug_utils = "XR_EXT_debug_utils" in names
        registry.action_priorities = "XR_EXT_active_action_set_priority" in names
        registry.passthrough = any(name in names for name in [
            "XR_FB_passthrough",
            "XR_HTC_passthrough",
            "XR_VARJO_composition_layer_depth_test",
        ])
        registry.scene_understanding = any(name in names for name in [
            "XR_MSFT_scene_understanding",
            "XR_QCOM_scene_understanding",
            "XR_ANDROID_scene_understanding",
        ])
        registry.anchors = any(name in names for name in [
            "XR_MSFT_spatial_anchor",
            "XR_FB_spatial_entity",
            "XR_HTC_anchor",
        ])

        if registry.hand_tracking:
            registry.notes.append("Hand tracking extension detected.")
        if registry.eye_gaze:
            registry.notes.append("Eye gaze interaction extension detected.")
        if registry.scene_understanding:
            registry.notes.append("Scene understanding style extension detected.")
        if registry.passthrough:
            registry.notes.append("Passthrough-style extension detected.")
        if not registry.graphics_bindings:
            registry.notes.append("No graphics-enable extension reported; remain in probe/simulator mode.")
        return True, registry, registry.notes


class OpenXRSessionFacade:
    """Lifecycle-centric wrapper for future real runtime integration.

    The current implementation keeps the state machine honest and can attempt a
    tiny amount of real bootstrap work only when explicitly requested. The rest
    of the engine stays fully runnable without an XR runtime.
    """

    def __init__(self) -> None:
        self.model = OpenXRSystemModel()
        self.locator = OpenXRRuntimeLocator()
        self.probe = OpenXRCapabilityProbe()

    def initialize(self) -> OpenXRSystemModel:
        self.model.runtime_locations = self.locator.discover()
        self.model.runtime_selected = self.model.runtime_locations[0] if self.model.runtime_locations else None
        pyopenxr_ok, registry, notes = self.probe.probe()
        self.model.pyopenxr_available = pyopenxr_ok
        self.model.extension_registry = registry
        self.model.graphics_binding = registry.choose_preferred_graphics()
        self.model.notes.extend(notes)
        self.model.lifecycle_state = SessionState.INSTANCE_READY if pyopenxr_ok else SessionState.IDLE
        if self.model.runtime_selected is not None:
            self.model.notes.append(f"Runtime manifest discovered from {self.model.runtime_selected.source}.")
        else:
            self.model.notes.append("No runtime manifest discovered; simulation remains primary.")
        if self.model.graphics_binding.api != GraphicsAPI.NONE:
            self.model.lifecycle_state = SessionState.SYSTEM_READY
        return self.model

    def advance_to_session_ready(self) -> None:
        if self.model.lifecycle_state in {SessionState.SYSTEM_READY, SessionState.INSTANCE_READY}:
            self.model.lifecycle_state = SessionState.SESSION_READY

    def begin(self) -> None:
        if self.model.lifecycle_state == SessionState.SESSION_READY:
            self.model.lifecycle_state = SessionState.RUNNING

    def stop(self) -> None:
        if self.model.lifecycle_state == SessionState.RUNNING:
            self.model.lifecycle_state = SessionState.STOPPING

    def finalize(self) -> None:
        if self.model.lifecycle_state in {SessionState.STOPPING, SessionState.RUNNING, SessionState.SESSION_READY, SessionState.SYSTEM_READY}:
            self.model.lifecycle_state = SessionState.STOPPED


# ---------------------------------------------------------------------------
# Performance instrumentation
# ---------------------------------------------------------------------------


@dataclass
class SpanRecord:
    name: str
    ms: float

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "ms": round(self.ms, 4)}


@dataclass
class FrameMetrics:
    frame_index: int
    dt: float
    spans: List[SpanRecord] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def add_span(self, name: str, ms: float) -> None:
        self.spans.append(SpanRecord(name=name, ms=ms))

    def total_ms(self) -> float:
        return sum(span.ms for span in self.spans)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "frame_index": self.frame_index,
            "dt": round(self.dt, 5),
            "total_ms": round(self.total_ms(), 4),
            "spans": [s.to_dict() for s in self.spans],
            "warnings": list(self.warnings),
        }


class Profiler:
    def __init__(self) -> None:
        self._started_at: Dict[str, float] = {}

    def start(self, name: str) -> None:
        self._started_at[name] = time.perf_counter()

    def stop(self, frame_metrics: FrameMetrics, name: str) -> None:
        began = self._started_at.pop(name, None)
        if began is None:
            return
        elapsed_ms = (time.perf_counter() - began) * 1000.0
        frame_metrics.add_span(name, elapsed_ms)


# ---------------------------------------------------------------------------
# Security / telemetry boundary
# ---------------------------------------------------------------------------


@dataclass
class SecurityPolicy:
    quantize_positions: bool = True
    position_quantum: float = 0.05
    redact_gaze_vectors: bool = True
    redact_hand_pose_vectors: bool = True
    redact_affect_raw: bool = False
    store_runtime_manifest_paths: bool = False


class SecurityBoundary:
    def __init__(self, policy: SecurityPolicy) -> None:
        self.policy = policy

    def _quantize(self, value: float) -> float:
        q = self.policy.position_quantum
        return round(round(value / q) * q, 4)

    def redact_sensor(self, sensor: Dict[str, Any]) -> Dict[str, Any]:
        out = json.loads(json.dumps(sensor))
        if self.policy.quantize_positions:
            for key in ["head", "left_hand", "right_hand"]:
                aim_or_pose = out.get(key, {})
                pose_block = aim_or_pose.get("aim", aim_or_pose)
                position = pose_block.get("position")
                if isinstance(position, dict):
                    for axis in ["x", "y", "z"]:
                        if axis in position:
                            position[axis] = self._quantize(float(position[axis]))
        if self.policy.redact_gaze_vectors and "gaze" in out:
            out["gaze"]["direction"] = {"redacted": True}
        if self.policy.redact_hand_pose_vectors:
            for key in ["left_hand", "right_hand"]:
                if key in out and isinstance(out[key], dict) and "aim" in out[key]:
                    out[key]["aim"]["forward"] = {"redacted": True}
        if self.policy.redact_affect_raw and "affect" in out:
            out["affect"] = {"redacted": True}
        return out

    def redact_runtime(self, runtime: Dict[str, Any]) -> Dict[str, Any]:
        out = json.loads(json.dumps(runtime))
        if not self.policy.store_runtime_manifest_paths:
            if out.get("runtime_selected"):
                out["runtime_selected"]["manifest_path"] = "<redacted>"
            for item in out.get("runtime_locations", []):
                item["manifest_path"] = "<redacted>"
        return out


# ---------------------------------------------------------------------------
# Sensor and affect model
# ---------------------------------------------------------------------------


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
    valence: float = 0.25
    arousal: float = 0.4
    pain: float = 0.0
    overload: float = 0.0
    clarity: float = 0.7
    confidence: float = 0.6
    curiosity: float = 0.4

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
            "t": round(self.t, 4),
            "dt": round(self.dt, 5),
            "head": self.head.to_dict(),
            "left_hand": self.left_hand.to_dict(),
            "right_hand": self.right_hand.to_dict(),
            "gaze": self.gaze.to_dict(),
            "affect": self.affect.to_dict(),
        }


# ---------------------------------------------------------------------------
# Scene graph and world model
# ---------------------------------------------------------------------------


@dataclass
class SceneObject:
    object_id: str
    kind: ObjectKind
    label: str
    transform: Transform
    radius: float = 0.4
    importance: float = 0.5
    safe_clearance: float = 0.8
    velocity: Vec3 = field(default_factory=Vec3)
    tags: List[str] = field(default_factory=list)
    speakable_name: Optional[str] = None

    def moved(self, dt: float) -> "SceneObject":
        return SceneObject(
            object_id=self.object_id,
            kind=self.kind,
            label=self.label,
            transform=Transform(
                position=self.transform.position + self.velocity * dt,
                forward=self.transform.forward,
                up=self.transform.up,
                scale=self.transform.scale,
            ),
            radius=self.radius,
            importance=self.importance,
            safe_clearance=self.safe_clearance,
            velocity=self.velocity,
            tags=list(self.tags),
            speakable_name=self.speakable_name,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "object_id": self.object_id,
            "kind": self.kind.value,
            "label": self.label,
            "transform": self.transform.to_dict(),
            "radius": round(self.radius, 4),
            "importance": round(self.importance, 4),
            "safe_clearance": round(self.safe_clearance, 4),
            "velocity": self.velocity.to_dict(),
            "tags": list(self.tags),
        }


@dataclass
class NearbyObservation:
    obj: SceneObject
    distance: float
    bearing: float
    closing_speed: float
    hazard: float
    hazard_kind: HazardKind
    is_ahead: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "obj": self.obj.to_dict(),
            "distance": round(self.distance, 4),
            "bearing": round(self.bearing, 5),
            "closing_speed": round(self.closing_speed, 4),
            "hazard": round(self.hazard, 4),
            "hazard_kind": self.hazard_kind.value,
            "is_ahead": self.is_ahead,
        }


@dataclass
class PredictedHazard:
    object_id: str
    kind: ObjectKind
    time_ahead: float
    risk: float
    projected_position: Vec3
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "object_id": self.object_id,
            "kind": self.kind.value,
            "time_ahead": round(self.time_ahead, 4),
            "risk": round(self.risk, 4),
            "projected_position": self.projected_position.to_dict(),
            "reason": self.reason,
        }


class SceneGraph:
    def __init__(self) -> None:
        self.objects: List[SceneObject] = []

    def add(self, obj: SceneObject) -> None:
        self.objects.append(obj)

    def extend(self, objs: Iterable[SceneObject]) -> None:
        self.objects.extend(objs)

    def nearby(self, head: Pose, max_distance: float = 12.0) -> List[NearbyObservation]:
        forward = head.forward.horizontal().normalized()
        if forward.length() <= 1e-6:
            forward = Vec3(0.0, 0.0, -1.0)
        right = Vec3(-forward.z, 0.0, forward.x)
        items: List[NearbyObservation] = []
        for obj in self.objects:
            delta = obj.transform.position - head.position
            horiz = delta.horizontal()
            distance = max(0.001, horiz.length())
            if distance > max_distance:
                continue
            rel = horiz.normalized()
            bearing = math.atan2(rel.dot(right), rel.dot(forward))
            radial = max(0.0, distance - obj.radius)
            closing_speed = max(0.0, -(obj.velocity.horizontal().dot(rel)))
            kind_base = {
                ObjectKind.VEHICLE: (1.0, HazardKind.MOVING_OBJECT),
                ObjectKind.STAIRS: (0.95, HazardKind.STAIRS),
                ObjectKind.WATER: (0.82, HazardKind.WATER),
                ObjectKind.POLE: (0.66, HazardKind.COLLISION),
                ObjectKind.TREE: (0.48, HazardKind.COLLISION),
                ObjectKind.BENCH: (0.38, HazardKind.COLLISION),
                ObjectKind.PERSON: (0.35, HazardKind.CROWD),
                ObjectKind.DOOR: (0.18, HazardKind.UNKNOWN),
                ObjectKind.RAMP: (0.14, HazardKind.UNKNOWN),
                ObjectKind.PATH: (0.05, HazardKind.UNKNOWN),
                ObjectKind.LANDMARK: (0.04, HazardKind.UNKNOWN),
                ObjectKind.BUILDING: (0.1, HazardKind.COLLISION),
                ObjectKind.WALL: (0.7, HazardKind.COLLISION),
                ObjectKind.SIGNAL: (0.08, HazardKind.UNKNOWN),
            }.get(obj.kind, (0.2, HazardKind.UNKNOWN))
            base_weight, hazard_kind = kind_base
            clearance_gap = max(0.0, obj.safe_clearance - radial)
            hazard = clamp(base_weight * (1.6 / (distance + 0.45)) + 0.18 * closing_speed + 0.3 * clearance_gap, 0.0, 1.5)
            items.append(
                NearbyObservation(
                    obj=obj,
                    distance=distance,
                    bearing=bearing,
                    closing_speed=closing_speed,
                    hazard=hazard,
                    hazard_kind=hazard_kind,
                    is_ahead=abs(bearing) < 0.85,
                )
            )
        items.sort(key=lambda n: (-n.hazard, n.distance))
        return items


# ---------------------------------------------------------------------------
# Accessibility output model
# ---------------------------------------------------------------------------


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
            "azimuth": round(self.azimuth, 5),
            "distance": round(self.distance, 4),
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
    notes: List[str] = field(default_factory=list)
    speech: List[str] = field(default_factory=list)
    haptics: List[str] = field(default_factory=list)
    spatial_audio: List[SpatialCue] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "t": round(self.t, 4),
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
class TeleportCandidate:
    valid: bool
    position: Vec3 = field(default_factory=Vec3)
    floor_position: Vec3 = field(default_factory=Vec3)
    distance: float = 0.0
    hazard: float = 0.0
    reason: str = ""
    source: str = "none"
    snapped_to: Optional[str] = None
    arc_points: List[Vec3] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "valid": self.valid,
            "position": self.position.to_dict(),
            "floor_position": self.floor_position.to_dict(),
            "distance": round(self.distance, 4),
            "hazard": round(self.hazard, 4),
            "reason": self.reason,
            "source": self.source,
            "snapped_to": self.snapped_to,
            "arc_points": [p.to_dict() for p in self.arc_points],
        }


@dataclass
class TeleportState:
    world_offset: Vec3 = field(default_factory=Vec3)
    pending: Optional[TeleportCandidate] = None
    last_commit_t: float = -999.0
    pending_since_t: float = 0.0
    commit_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "world_offset": self.world_offset.to_dict(),
            "pending": None if self.pending is None else self.pending.to_dict(),
            "last_commit_t": round(self.last_commit_t, 4),
            "commit_count": self.commit_count,
        }


# ---------------------------------------------------------------------------
# Actions and interaction profiles
# ---------------------------------------------------------------------------


class ActionRegistry:
    def __init__(self) -> None:
        self.action_sets: List[ActionSetDefinition] = []
        self.interaction_profiles: List[InteractionProfile] = []
        self._build_defaults()

    def _build_defaults(self) -> None:
        locomotion = ActionSetDefinition(
            name="locomotion",
            localized_name="Locomotion",
            priority=50,
            actions=[
                ActionDefinition(
                    name="navigate",
                    localized_name="Navigate",
                    action_type=ActionType.VECTOR2,
                    semantic_role="desired_move_axis",
                    bindings=[
                        ActionBinding("/user/hand/left/input/thumbstick"),
                        ActionBinding("/user/hand/right/input/thumbstick"),
                    ],
                ),
                ActionDefinition(
                    name="head_pose",
                    localized_name="Head Pose",
                    action_type=ActionType.POSE,
                    semantic_role="head_tracking",
                    bindings=[ActionBinding("/user/head/input/pose")],
                ),
                ActionDefinition(
                    name="teleport_aim",
                    localized_name="Teleport Aim",
                    action_type=ActionType.POSE,
                    semantic_role="teleport_arc_origin_and_direction",
                    bindings=[
                        ActionBinding("/user/hand/right/input/aim/pose", "preferred controller or hand aim"),
                        ActionBinding("/user/hand/left/input/aim/pose", "left-hand fallback"),
                        ActionBinding("/user/head/input/pose", "head fallback"),
                    ],
                ),
                ActionDefinition(
                    name="teleport_commit",
                    localized_name="Teleport Commit",
                    action_type=ActionType.BOOLEAN,
                    semantic_role="activate_teleport",
                    bindings=[
                        ActionBinding("/user/hand/right/input/select/click"),
                        ActionBinding("/user/hand/right/input/thumbstick/click"),
                    ],
                ),
                ActionDefinition(
                    name="teleport_cancel",
                    localized_name="Teleport Cancel",
                    action_type=ActionType.BOOLEAN,
                    semantic_role="cancel_teleport_preview",
                    bindings=[
                        ActionBinding("/user/hand/left/input/select/click"),
                        ActionBinding("/user/hand/left/input/menu/click"),
                    ],
                ),
            ],
        )
        accessibility = ActionSetDefinition(
            name="accessibility",
            localized_name="Accessibility",
            priority=100,
            actions=[
                ActionDefinition(
                    name="focus_pose",
                    localized_name="Focus Pose",
                    action_type=ActionType.POSE,
                    semantic_role="aim_or_gaze_reference",
                    bindings=[
                        ActionBinding("/user/eyes_ext/input/gaze_ext/pose", "eye-gaze if available"),
                        ActionBinding("/user/hand/right/input/aim/pose", "fallback aim pose"),
                    ],
                ),
                ActionDefinition(
                    name="confirm",
                    localized_name="Confirm",
                    action_type=ActionType.BOOLEAN,
                    semantic_role="accept_guidance",
                    bindings=[
                        ActionBinding("/user/hand/right/input/select/click"),
                        ActionBinding("/user/hand/right/input/trigger/value"),
                    ],
                ),
                ActionDefinition(
                    name="haptic_out",
                    localized_name="Haptic Out",
                    action_type=ActionType.HAPTIC,
                    semantic_role="guidance_haptics",
                    bindings=[
                        ActionBinding("/user/hand/left/output/haptic"),
                        ActionBinding("/user/hand/right/output/haptic"),
                    ],
                ),
            ],
        )
        system = ActionSetDefinition(
            name="system",
            localized_name="System",
            priority=10,
            actions=[
                ActionDefinition(
                    name="menu",
                    localized_name="Menu",
                    action_type=ActionType.BOOLEAN,
                    semantic_role="system_menu",
                    bindings=[
                        ActionBinding("/user/hand/left/input/menu/click"),
                        ActionBinding("/user/hand/right/input/menu/click"),
                    ],
                )
            ],
        )
        self.action_sets = [locomotion, accessibility, system]
        self.interaction_profiles = [
            InteractionProfile("/interaction_profiles/khr/simple_controller", "baseline portability", ["navigate", "teleport_aim", "teleport_commit", "teleport_cancel", "confirm", "haptic_out"]),
            InteractionProfile("/interaction_profiles/oculus/touch_controller", "pcvr/mobile controller family", ["navigate", "teleport_aim", "teleport_commit", "teleport_cancel", "confirm", "menu", "haptic_out"]),
            InteractionProfile("/interaction_profiles/ext/hand_interaction_ext", "optical hands", ["confirm", "focus_pose", "teleport_aim", "teleport_commit"]),
            InteractionProfile("/interaction_profiles/microsoft/hand_interaction", "hand interaction profile", ["confirm", "focus_pose", "teleport_aim", "teleport_commit"]),
        ]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action_sets": [a.to_dict() for a in self.action_sets],
            "interaction_profiles": [i.to_dict() for i in self.interaction_profiles],
        }


# ---------------------------------------------------------------------------
# Filtering and affect-energy policy
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
        position = Vec3(
            lerp(self._last_head.position.x, pose.position.x, self.alpha),
            lerp(self._last_head.position.y, pose.position.y, self.alpha),
            lerp(self._last_head.position.z, pose.position.z, self.alpha),
        )
        forward = Vec3(
            lerp(self._last_head.forward.x, pose.forward.x, self.alpha),
            lerp(self._last_head.forward.y, pose.forward.y, self.alpha),
            lerp(self._last_head.forward.z, pose.forward.z, self.alpha),
        ).normalized()
        confidence = clamp(0.6 * self._last_head.confidence + 0.4 * pose.confidence, 0.0, 1.0)
        filtered = Pose(position=position, forward=forward, confidence=confidence)
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
        direction = Vec3(
            lerp(self._last_gaze.direction.x, gaze.direction.x, self.alpha),
            lerp(self._last_gaze.direction.y, gaze.direction.y, self.alpha),
            lerp(self._last_gaze.direction.z, gaze.direction.z, self.alpha),
        ).normalized()
        stability = clamp((0.7 * self._last_gaze.stability + 0.3 * gaze.stability) - stale_penalty, 0.0, 1.0)
        filtered = GazeState(tracked=True, direction=direction, stability=stability)
        self._last_gaze = filtered
        self._last_t = t
        return filtered


class NegativeConverter:
    def convert(self, affect: AffectSignal) -> Dict[str, float]:
        a = affect.normalized()
        load = clamp(0.6 * a.overload + 0.55 * a.pain + 0.25 * max(0.0, -a.valence), 0.0, 1.0)
        return {
            "negative_load": load,
            "recovery_energy": clamp(0.25 + 0.5 * load + 0.15 * a.clarity, 0.0, 1.0),
            "shield": clamp(0.3 + 0.7 * load, 0.0, 1.0),
            "simplification": clamp(0.2 + 0.8 * load, 0.0, 1.0),
        }


class PositiveToPositiveModule:
    def reinforce(self, affect: AffectSignal) -> Dict[str, float]:
        a = affect.normalized()
        positive_core = clamp(
            0.35 * (a.valence + 1.0) / 2.0 + 0.25 * a.clarity + 0.2 * a.confidence + 0.2 * a.curiosity,
            0.0,
            1.0,
        )
        return {
            "positive_core": positive_core,
            "calm_gain": clamp(0.25 + 0.6 * positive_core - 0.25 * a.overload, 0.0, 1.0),
            "focus_gain": clamp(0.2 + 0.6 * a.clarity + 0.2 * a.confidence, 0.0, 1.0),
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
# Module host system
# ---------------------------------------------------------------------------


@dataclass
class FrameContext:
    frame_index: int
    runtime: OpenXRSystemModel
    actions: ActionRegistry
    sensor: SensorFrame
    scene: SceneGraph
    frame_metrics: FrameMetrics
    shared: MutableMapping[str, Any] = field(default_factory=dict)
    guidance: Optional[GuidanceFrame] = None


class Module:
    name: str = "module"
    order: int = 100

    def on_boot(self, engine: "AegisGlyphEngine") -> None:
        pass

    def process(self, stage: ModuleStage, ctx: FrameContext) -> None:
        pass


class ModuleHost:
    def __init__(self) -> None:
        self.modules: List[Module] = []

    def register(self, module: Module) -> None:
        self.modules.append(module)
        self.modules.sort(key=lambda m: (m.order, m.name))

    def boot(self, engine: "AegisGlyphEngine") -> None:
        for module in self.modules:
            module.on_boot(engine)

    def run_stage(self, stage: ModuleStage, ctx: FrameContext) -> None:
        for module in self.modules:
            module.process(stage, ctx)


# ---------------------------------------------------------------------------
# Concrete modules
# ---------------------------------------------------------------------------


class WorldPopulationModule(Module):
    name = "world_population"
    order = 10

    def __init__(self) -> None:
        self.objects: List[SceneObject] = []

    def on_boot(self, engine: "AegisGlyphEngine") -> None:
        self.objects = [
            SceneObject("echo_gate", ObjectKind.LANDMARK, "Echo Gate", Transform(position=Vec3(0.0, 1.4, -5.4)), radius=0.8, importance=1.0, safe_clearance=1.0, speakable_name="echo gate"),
            SceneObject("main_path", ObjectKind.PATH, "Main Path", Transform(position=Vec3(0.0, 0.0, -4.0)), radius=2.2, importance=0.9, safe_clearance=1.6, speakable_name="main path"),
            SceneObject("oak_tree_left", ObjectKind.TREE, "Oak Tree", Transform(position=Vec3(-2.2, 0.0, -3.8)), radius=0.55, importance=0.25, safe_clearance=0.9, speakable_name="tree left"),
            SceneObject("pine_tree_right", ObjectKind.TREE, "Pine Tree", Transform(position=Vec3(2.4, 0.0, -6.0)), radius=0.6, importance=0.25, safe_clearance=1.0, speakable_name="tree right"),
            SceneObject("bench_north", ObjectKind.BENCH, "Bench", Transform(position=Vec3(-1.5, 0.0, -5.5)), radius=0.5, importance=0.35, safe_clearance=0.8, speakable_name="bench"),
            SceneObject("door_gallery", ObjectKind.DOOR, "Gallery Door", Transform(position=Vec3(0.7, 1.0, -9.0)), radius=0.6, importance=0.85, safe_clearance=0.8, speakable_name="gallery door"),
            SceneObject("ramp_access", ObjectKind.RAMP, "Access Ramp", Transform(position=Vec3(1.8, 0.3, -7.2)), radius=0.9, importance=0.65, safe_clearance=1.0, speakable_name="access ramp"),
            SceneObject("stairs_warning", ObjectKind.STAIRS, "Stairs", Transform(position=Vec3(-2.8, -0.6, -7.5)), radius=0.9, importance=0.75, safe_clearance=1.4, speakable_name="stairs"),
            SceneObject("water_feature", ObjectKind.WATER, "Water Feature", Transform(position=Vec3(3.4, -0.2, -4.4)), radius=1.4, importance=0.6, safe_clearance=1.6, speakable_name="water feature"),
            SceneObject("lamp_pole", ObjectKind.POLE, "Lamp Pole", Transform(position=Vec3(1.4, 0.0, -2.8)), radius=0.25, importance=0.5, safe_clearance=0.75, speakable_name="pole"),
            SceneObject("vehicle_shuttle", ObjectKind.VEHICLE, "Shuttle Vehicle", Transform(position=Vec3(-4.0, 0.0, -6.8)), radius=1.0, importance=0.95, safe_clearance=2.2, velocity=Vec3(0.55, 0.0, 0.0), speakable_name="shuttle vehicle"),
            SceneObject("visitor", ObjectKind.PERSON, "Visitor", Transform(position=Vec3(2.8, 0.0, -5.0)), radius=0.35, importance=0.45, safe_clearance=0.9, velocity=Vec3(-0.12, 0.0, -0.02), speakable_name="person"),
            SceneObject("north_building", ObjectKind.BUILDING, "North Building", Transform(position=Vec3(0.0, 1.0, -12.5)), radius=4.0, importance=0.55, safe_clearance=2.0, speakable_name="building"),
        ]

    def process(self, stage: ModuleStage, ctx: FrameContext) -> None:
        if stage != ModuleStage.BUILD_SCENE:
            return
        dt = ctx.sensor.dt
        updated: List[SceneObject] = []
        for obj in self.objects:
            moved = obj.moved(dt)
            if obj.kind == ObjectKind.VEHICLE:
                if moved.transform.position.x > 4.2:
                    moved.transform.position.x = -4.2
                elif moved.transform.position.x < -4.2:
                    moved.transform.position.x = 4.2
            if obj.kind == ObjectKind.PERSON:
                if moved.transform.position.z < -8.0 or moved.transform.position.z > -3.6:
                    moved.velocity = Vec3(moved.velocity.x, 0.0, -moved.velocity.z)
            updated.append(moved)
        self.objects = updated
        ctx.scene.extend(self.objects)


class FilterAndAffectModule(Module):
    name = "filter_and_affect"
    order = 20

    def __init__(self) -> None:
        self.filter = NegativeLayerFilter()
        self.energy = FeelingsToEnergySystem()

    def process(self, stage: ModuleStage, ctx: FrameContext) -> None:
        if stage != ModuleStage.ANALYZE:
            return
        filtered_head = self.filter.filter_pose(ctx.sensor.head)
        filtered_gaze = self.filter.filter_gaze(ctx.sensor.gaze, ctx.sensor.t)
        affect_map = self.energy.convert(ctx.sensor.affect)
        ctx.shared["filtered_head"] = filtered_head
        ctx.shared["filtered_gaze"] = filtered_gaze
        ctx.shared["affect_energy"] = affect_map
        ctx.shared["clutter_removed"] = self.filter.removed_fraction


class PredictiveHazardModule(Module):
    name = "predictive_hazard"
    order = 35

    def __init__(self) -> None:
        self.horizon_s = 2.4
        self.samples = 6
        self.walk_speed = 1.05

    def _project_head_path(self, head: Pose, basis: Vec3, t: float) -> Vec3:
        return head.position + basis * (self.walk_speed * t)

    def _risk_score(self, obj: SceneObject, projected_obj: Vec3, projected_head: Vec3) -> float:
        rel = projected_obj.horizontal() - projected_head.horizontal()
        distance = max(0.001, rel.length())
        radial = max(0.0, distance - obj.radius)
        rel_dir = rel.normalized()
        closing = max(0.0, -(obj.velocity.horizontal().dot(rel_dir))) if rel.length() > 1e-6 else 0.0
        base = {
            ObjectKind.VEHICLE: 1.0,
            ObjectKind.PERSON: 0.62,
            ObjectKind.STAIRS: 0.88,
            ObjectKind.WATER: 0.82,
            ObjectKind.WALL: 0.75,
        }.get(obj.kind, 0.32)
        return clamp(base * (1.35 / (radial + 0.45)) + 0.32 * closing, 0.0, 1.8)

    def process(self, stage: ModuleStage, ctx: FrameContext) -> None:
        if stage != ModuleStage.ANALYZE:
            return
        head: Pose = ctx.shared.get("filtered_head", ctx.sensor.head)
        basis = head.forward.horizontal().normalized()
        if basis.length() <= 1e-6:
            basis = Vec3(0.0, 0.0, -1.0)

        predictions: List[PredictedHazard] = []
        for obj in ctx.scene.objects:
            if obj.velocity.horizontal().length() < 0.05:
                continue
            best: Optional[PredictedHazard] = None
            for i in range(1, self.samples + 1):
                t = self.horizon_s * (i / self.samples)
                projected_obj = obj.transform.position + obj.velocity * t
                projected_obj = Vec3(projected_obj.x, head.position.y, projected_obj.z)
                projected_head = self._project_head_path(head, basis, t)
                risk = self._risk_score(obj, projected_obj, projected_head)
                reason = f"future-{obj.kind.value}-crossing"
                candidate = PredictedHazard(
                    object_id=obj.object_id,
                    kind=obj.kind,
                    time_ahead=t,
                    risk=risk,
                    projected_position=projected_obj,
                    reason=reason,
                )
                if best is None or candidate.risk > best.risk:
                    best = candidate
            if best is not None and best.risk > 0.42:
                predictions.append(best)

        predictions.sort(key=lambda p: (-p.risk, p.time_ahead))
        predictions = predictions[:6]
        ctx.shared["predicted_hazards"] = predictions

        top_risk = predictions[0].risk if predictions else 0.0
        path_safety = clamp(1.0 - top_risk / 1.6, 0.0, 1.0)
        ctx.shared["path_safety_index"] = path_safety
        if predictions:
            ctx.shared["predictive_warning"] = (
                f"{predictions[0].kind.value} risk in {predictions[0].time_ahead:.1f}s"
            )


class GuidancePlannerModule(Module):
    name = "guidance_planner"
    order = 40

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

    def _choose_target(self, head: Pose, basis: Vec3, nearby: Sequence[NearbyObservation]) -> Tuple[Optional[SceneObject], float]:
        best_obj: Optional[SceneObject] = None
        best_score = -1e9
        for item in nearby:
            obj = item.obj
            if obj.kind not in {ObjectKind.LANDMARK, ObjectKind.PATH, ObjectKind.DOOR, ObjectKind.RAMP, ObjectKind.BUILDING, ObjectKind.SIGNAL}:
                continue
            align = basis.dot((obj.transform.position - head.position).horizontal().normalized())
            path_bonus = 0.18 if obj.kind == ObjectKind.PATH else 0.0
            score = obj.importance * (1.15 * align + 0.55 / (item.distance + 0.2) + path_bonus) - 0.35 * item.hazard
            if score > best_score:
                best_score = score
                best_obj = obj
        return best_obj, best_score

    def process(self, stage: ModuleStage, ctx: FrameContext) -> None:
        if stage != ModuleStage.GUIDE:
            return
        head: Pose = ctx.shared.get("filtered_head", ctx.sensor.head)
        gaze: GazeState = ctx.shared.get("filtered_gaze", ctx.sensor.gaze)
        affect_energy: Dict[str, float] = ctx.shared.get("affect_energy", {})
        clutter_removed = float(ctx.shared.get("clutter_removed", 0.0))
        nearby = ctx.scene.nearby(head)
        ctx.shared["nearby"] = nearby

        basis, steering_mode = self._steering_basis(head, gaze, ctx.sensor.left_hand, ctx.sensor.right_hand)
        target, target_score = self._choose_target(head, basis, nearby)
        strongest_hazard = nearby[0] if nearby else None
        nearest_object = strongest_hazard.obj.object_id if strongest_hazard else None
        predicted: List[PredictedHazard] = ctx.shared.get("predicted_hazards", [])
        predictive_risk = predicted[0].risk if predicted else 0.0
        notes: List[str] = []

        if target is None:
            ctx.guidance = GuidanceFrame(
                t=ctx.sensor.t,
                mode=GuidanceMode.RECOVER,
                cue_word="hold",
                turn_bias=0.0,
                pulse=0.18,
                stability=0.0,
                energy=float(affect_energy.get("energy", 0.2)),
                comfort=float(affect_energy.get("comfort", 0.3)),
                clutter_removed=clutter_removed,
                target=None,
                nearest_object=nearest_object,
                notes=["No stable target available."],
            )
            return

        target_delta = (target.transform.position - head.position).horizontal()
        turn_bias = self._turn_bias(head.forward.horizontal().normalized(), target_delta.normalized())
        distance = max(0.01, target_delta.length())
        danger = strongest_hazard.hazard if strongest_hazard else 0.0
        hand_activity = max(ctx.sensor.left_hand.pinch, ctx.sensor.right_hand.pinch, ctx.sensor.left_hand.joint_energy, ctx.sensor.right_hand.joint_energy)
        stability = clamp(
            0.25 * head.confidence +
            0.20 * (gaze.stability if gaze.tracked else 0.5) +
            0.20 * float(affect_energy.get("comfort", 0.3)) +
            0.15 * float(affect_energy.get("resilience", 0.3)) +
            0.20 * ctx.sensor.affect.confidence,
            0.0,
            1.0,
        )

        if ctx.sensor.affect.pain > 0.45 or ctx.sensor.affect.overload > 0.72 or danger > 1.0 or predictive_risk > 1.05:
            mode = GuidanceMode.RECOVER
            cue_word = "soften"
            pulse = clamp(0.18 + 0.22 * float(affect_energy.get("negative_load", 0.0)) + 0.18 * danger, 0.15, 0.62)
            notes.append("Recovery prioritised: simplify cues and protect movement.")
        elif danger > 0.62 or abs(turn_bias) > 0.45 or distance < 1.2 or predictive_risk > 0.72:
            mode = GuidanceMode.ALERT
            cue_word = "pivot"
            pulse = clamp(0.48 + 0.22 * float(affect_energy.get("energy", 0.3)) + 0.12 * hand_activity + 0.08 * danger, 0.35, 1.0)
            notes.append("Hazard or sharp turn requires quick correction.")
        elif target_score > 0.42:
            mode = GuidanceMode.FOCUS
            cue_word = "trace"
            pulse = clamp(0.34 + 0.22 * float(affect_energy.get("energy", 0.3)) + 0.12 * float(affect_energy.get("focus_gain", 0.3)), 0.2, 0.85)
            notes.append("Stable target lock established.")
        else:
            mode = GuidanceMode.CALM
            cue_word = "glide"
            pulse = clamp(0.20 + 0.14 * float(affect_energy.get("calm_gain", 0.3)) + 0.10 * float(affect_energy.get("energy", 0.3)), 0.15, 0.55)
            notes.append("Low-friction navigation mode.")

        if clutter_removed > 0.2:
            notes.append("Noise suppression active.")
        notes.append(f"Steering basis: {steering_mode}.")
        if strongest_hazard and strongest_hazard.hazard > 0.55:
            notes.append(f"Hazard priority: {strongest_hazard.obj.kind.value}.")
        if predicted and predicted[0].risk > 0.58:
            notes.append(
                f"Predictive risk: {predicted[0].kind.value} in {predicted[0].time_ahead:.1f}s."
            )

        ctx.guidance = GuidanceFrame(
            t=ctx.sensor.t,
            mode=mode,
            cue_word=cue_word,
            turn_bias=turn_bias,
            pulse=pulse,
            stability=stability,
            energy=float(affect_energy.get("energy", 0.2)),
            comfort=float(affect_energy.get("comfort", 0.3)),
            clutter_removed=clutter_removed,
            target=target.object_id,
            nearest_object=nearest_object,
            notes=notes,
        )


class SpatialMemoryModule(Module):
    name = "spatial_memory"
    order = 45

    def __init__(self) -> None:
        self.visited = deque(maxlen=120)
        self.object_hits: Dict[str, int] = {}

    def process(self, stage: ModuleStage, ctx: FrameContext) -> None:
        head: Pose = ctx.shared.get("filtered_head", ctx.sensor.head)
        if stage == ModuleStage.PRE_FRAME:
            self.visited.append(head.position)
            return
        if stage != ModuleStage.GUIDE or ctx.guidance is None:
            return

        nearby: List[NearbyObservation] = ctx.shared.get("nearby", [])
        if not nearby:
            return

        anchor = nearby[0]
        hits = self.object_hits.get(anchor.obj.object_id, 0) + 1
        self.object_hits[anchor.obj.object_id] = hits

        travel = 0.0
        if len(self.visited) > 1:
            prev = None
            for p in self.visited:
                if prev is not None:
                    travel += p.distance_to(prev)
                prev = p

        confidence = clamp(
            0.24 + 0.10 * min(hits, 6) + 0.45 * float(ctx.shared.get("path_safety_index", 0.5)),
            0.0,
            1.0,
        )
        rel = "ahead" if abs(anchor.bearing) < 0.25 else ("left" if anchor.bearing < 0 else "right")
        memory_payload = {
            "anchor_id": anchor.obj.object_id,
            "anchor_name": anchor.obj.speakable_name or anchor.obj.label.lower(),
            "anchor_relative": rel,
            "anchor_hits": hits,
            "confidence": confidence,
            "travel_distance": round(travel, 3),
        }
        ctx.shared["spatial_memory"] = memory_payload

        if hits >= 3 and anchor.obj.kind in {ObjectKind.PATH, ObjectKind.LANDMARK, ObjectKind.DOOR, ObjectKind.RAMP}:
            ctx.guidance.notes.append(
                f"Familiar anchor recognised: {memory_payload['anchor_name']} ({rel})."
            )


class TeleportationModule(Module):
    name = "teleportation"
    order = 50

    def __init__(self) -> None:
        self.state = TeleportState()
        self.cooldown_s = 0.9
        self.max_distance = 7.5
        self.safe_hazard_limit = 0.78
        self.preview_hold_s = 0.08

    def on_boot(self, engine: "AegisGlyphEngine") -> None:
        self.state = TeleportState()
        engine.teleport = self.state

    def _effective_head(self, ctx: FrameContext) -> Pose:
        head: Pose = ctx.shared.get("filtered_head", ctx.sensor.head)
        return Pose(
            position=head.position + self.state.world_offset,
            forward=head.forward,
            confidence=head.confidence,
        )

    def _aim_basis(self, ctx: FrameContext, head: Pose) -> Tuple[Vec3, str]:
        gaze: GazeState = ctx.shared.get("filtered_gaze", ctx.sensor.gaze)
        if ctx.sensor.right_hand.tracked and ctx.sensor.right_hand.aim.confidence > 0.55:
            return ctx.sensor.right_hand.aim.forward.normalized(), "right-hand"
        if ctx.sensor.left_hand.tracked and ctx.sensor.left_hand.aim.confidence > 0.55:
            return ctx.sensor.left_hand.aim.forward.normalized(), "left-hand"
        if gaze.tracked and gaze.stability > 0.3:
            return gaze.direction.normalized(), "gaze"
        return head.forward.normalized(), "head"

    def _ray_to_floor(self, origin: Vec3, direction: Vec3) -> Optional[Vec3]:
        d = direction.normalized()
        if d.y < -0.05:
            t = (0.0 - origin.y) / d.y
            if 0.1 <= t <= self.max_distance * 1.35:
                p = origin + d * t
                flat = Vec3(p.x, 0.0, p.z)
                if flat.horizontal().length() <= 10000:  # keep the tiny goblin quiet
                    return flat
        flat_dir = direction.horizontal().normalized()
        if flat_dir.length() <= 1e-6:
            flat_dir = Vec3(0.0, 0.0, -1.0)
        return Vec3(origin.x, 0.0, origin.z) + flat_dir * min(3.2, self.max_distance)

    def _build_arc(self, origin: Vec3, direction: Vec3, landing: Vec3) -> List[Vec3]:
        pts: List[Vec3] = []
        d = direction.normalized()
        flat = Vec3(landing.x, origin.y, landing.z)
        distance = max(0.2, origin.horizontal().distance_to(flat.horizontal()))
        gravity = 3.0
        lift = max(0.5, min(2.0, origin.y * 0.55 + 0.25 * distance))
        for i in range(12):
            t = i / 11.0
            base = origin + (flat - origin) * t
            arc_y = origin.y + lift * math.sin(math.pi * t) - gravity * (t * t) * 0.08 * distance
            pts.append(Vec3(base.x, max(0.0, arc_y), base.z))
        pts[-1] = landing
        return pts

    def _snap_candidate(self, landing: Vec3, ctx: FrameContext) -> Tuple[Vec3, Optional[str], str]:
        best_obj = None
        best_score = -1e9
        for obj in ctx.scene.objects:
            if obj.kind not in {ObjectKind.PATH, ObjectKind.DOOR, ObjectKind.RAMP, ObjectKind.LANDMARK, ObjectKind.SIGNAL}:
                continue
            dist = obj.transform.position.horizontal().distance_to(landing.horizontal())
            if dist > 1.6:
                continue
            score = obj.importance * 1.4 - dist
            if score > best_score:
                best_score = score
                best_obj = obj
        if best_obj is None:
            return landing, None, "floor-cast"
        snapped = Vec3(best_obj.transform.position.x, 0.0, best_obj.transform.position.z)
        return snapped, best_obj.object_id, f"snap:{best_obj.kind.value}"

    def _candidate_hazard(self, candidate: Vec3, ctx: FrameContext, head_forward: Vec3) -> Tuple[float, str]:
        probe = Pose(position=Vec3(candidate.x, ctx.sensor.head.position.y + self.state.world_offset.y, candidate.z), forward=head_forward, confidence=1.0)
        nearby = ctx.scene.nearby(probe, max_distance=2.4)
        if not nearby:
            return 0.0, "clear"
        primary = max(nearby, key=lambda n: n.hazard)
        hard_block = {ObjectKind.VEHICLE, ObjectKind.WATER, ObjectKind.STAIRS, ObjectKind.WALL}
        if primary.obj.kind in hard_block and primary.distance < max(1.0, primary.obj.safe_clearance):
            return max(primary.hazard, 1.0), f"blocked-by-{primary.obj.kind.value}"
        return primary.hazard, primary.hazard_kind.value

    def _should_commit(self, ctx: FrameContext, candidate: TeleportCandidate) -> bool:
        if not candidate.valid:
            return False
        if ctx.sensor.t - self.state.last_commit_t < self.cooldown_s:
            return False
        pinch_commit = ctx.sensor.right_hand.pinch > 0.41 or ctx.sensor.left_hand.pinch > 0.43
        focused_commit = ctx.guidance is not None and ctx.guidance.mode == GuidanceMode.RECOVER and ctx.sensor.right_hand.pinch > 0.7
        return pinch_commit or focused_commit

    def process(self, stage: ModuleStage, ctx: FrameContext) -> None:
        if stage == ModuleStage.ANALYZE:
            head = self._effective_head(ctx)
            ctx.shared["filtered_head"] = head
            direction, source = self._aim_basis(ctx, head)
            origin = Vec3(head.position.x, max(1.15, head.position.y), head.position.z)
            landing = self._ray_to_floor(origin, direction)
            if landing is None:
                candidate = TeleportCandidate(False, reason="no-surface", source=source)
            else:
                snapped, snapped_to, reason = self._snap_candidate(landing, ctx)
                hazard, hazard_reason = self._candidate_hazard(snapped, ctx, head.forward)
                distance = snapped.horizontal().distance_to(head.position.horizontal())
                valid = distance <= self.max_distance and hazard <= self.safe_hazard_limit
                candidate = TeleportCandidate(
                    valid=valid,
                    position=Vec3(snapped.x, head.position.y, snapped.z),
                    floor_position=snapped,
                    distance=distance,
                    hazard=hazard,
                    reason=hazard_reason if not valid else reason,
                    source=source,
                    snapped_to=snapped_to,
                    arc_points=self._build_arc(origin, direction, snapped),
                )

                predicted: List[PredictedHazard] = ctx.shared.get("predicted_hazards", [])
                if candidate.valid and predicted:
                    nearest_pred = min(
                        predicted,
                        key=lambda p: p.projected_position.horizontal().distance_to(candidate.floor_position.horizontal()),
                    )
                    pred_distance = nearest_pred.projected_position.horizontal().distance_to(
                        candidate.floor_position.horizontal()
                    )
                    if nearest_pred.risk > 0.72 and pred_distance < 1.25:
                        candidate.valid = False
                        candidate.reason = nearest_pred.reason
                        candidate.hazard = max(candidate.hazard, nearest_pred.risk)

            previous = self.state.pending
            if candidate.valid:
                if previous and previous.valid and previous.floor_position.distance_to(candidate.floor_position) < 0.2:
                    pass
                else:
                    self.state.pending_since_t = ctx.sensor.t
                self.state.pending = candidate
            else:
                self.state.pending = candidate
                self.state.pending_since_t = ctx.sensor.t

            if self._should_commit(ctx, self.state.pending) and (ctx.sensor.t - self.state.pending_since_t) >= self.preview_hold_s:
                dx = self.state.pending.floor_position.x - head.position.x
                dz = self.state.pending.floor_position.z - head.position.z
                self.state.world_offset = self.state.world_offset + Vec3(dx, 0.0, dz)
                self.state.last_commit_t = ctx.sensor.t
                self.state.commit_count += 1
                committed_head = Pose(
                    position=head.position + Vec3(dx, 0.0, dz),
                    forward=head.forward,
                    confidence=head.confidence,
                )
                ctx.shared["filtered_head"] = committed_head
                ctx.shared["teleport_committed"] = True
                ctx.shared["teleport_commit_target"] = self.state.pending.to_dict()
            else:
                ctx.shared["teleport_committed"] = False

            ctx.shared["teleport_state"] = self.state.to_dict()

        elif stage == ModuleStage.GUIDE:
            if ctx.guidance is None:
                return
            pending = self.state.pending
            if pending and pending.valid:
                ctx.guidance.notes.append(
                    f"Teleport preview ready via {pending.source}; target {pending.distance:.1f}m away."
                )
                if ctx.guidance.mode != GuidanceMode.RECOVER:
                    ctx.guidance.cue_word = "blink"
            elif pending and not pending.valid and pending.reason:
                ctx.guidance.notes.append(f"Teleport blocked: {pending.reason}.")

            if ctx.shared.get("teleport_committed"):
                ctx.guidance.notes.append("Teleport committed this frame.")
                ctx.guidance.cue_word = "phase"
                ctx.guidance.pulse = clamp(ctx.guidance.pulse + 0.18, 0.0, 1.0)



class ComfortAdaptationModule(Module):
    name = "comfort_adaptation"
    order = 55

    def __init__(self) -> None:
        self.turn_history = deque(maxlen=20)

    def process(self, stage: ModuleStage, ctx: FrameContext) -> None:
        if stage != ModuleStage.GUIDE or ctx.guidance is None:
            return
        guide = ctx.guidance
        self.turn_history.append(guide.turn_bias)

        jitter = 0.0
        if len(self.turn_history) >= 2:
            samples = list(self.turn_history)
            diffs = [abs(samples[i] - samples[i - 1]) for i in range(1, len(samples))]
            jitter = clamp(sum(diffs) / len(diffs), 0.0, 1.5)

        predicted: List[PredictedHazard] = ctx.shared.get("predicted_hazards", [])
        predictive_risk = predicted[0].risk if predicted else 0.0
        overload = ctx.sensor.affect.overload
        pain = ctx.sensor.affect.pain
        comfort_load = clamp(0.38 * overload + 0.26 * pain + 0.22 * jitter + 0.18 * predictive_risk, 0.0, 1.2)

        if comfort_load > 0.62:
            guide.pulse = clamp(guide.pulse * 0.78, 0.12, 0.85)
            if guide.mode != GuidanceMode.RECOVER and overload > 0.45:
                guide.mode = GuidanceMode.RECOVER
                guide.cue_word = "anchor"
            guide.notes.append("Comfort adaptation reduced cue intensity.")
        elif comfort_load < 0.26 and guide.energy > 0.55:
            guide.pulse = clamp(guide.pulse + 0.08, 0.0, 1.0)
            guide.notes.append("Comfort adaptation allowed higher tempo.")

        directive = {
            "comfort_load": round(comfort_load, 4),
            "vignette_strength": round(clamp(0.15 + 0.75 * comfort_load, 0.1, 0.92), 4),
            "movement_gain": round(clamp(1.0 - 0.45 * comfort_load, 0.45, 1.0), 4),
            "path_safety_index": round(float(ctx.shared.get("path_safety_index", 0.5)), 4),
        }
        ctx.shared["comfort_directive"] = directive


class AccessibilitySynthesisModule(Module):
    name = "accessibility_synthesis"
    order = 60

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

    def make_object_cue(self, nearby: NearbyObservation, pulse: float) -> SpatialCue:
        priority = CuePriority.INFO
        if nearby.hazard > 1.05:
            priority = CuePriority.CRITICAL
        elif nearby.hazard > 0.6:
            priority = CuePriority.WARNING
        elif nearby.obj.kind in {ObjectKind.DOOR, ObjectKind.RAMP, ObjectKind.PATH, ObjectKind.LANDMARK, ObjectKind.SIGNAL}:
            priority = CuePriority.GUIDE
        name = nearby.obj.speakable_name or nearby.obj.label.lower()
        text = f"{name} {self.relative_words(nearby.bearing)}, {self.distance_words(nearby.distance)}"
        return SpatialCue(
            text=text,
            azimuth=nearby.bearing,
            distance=nearby.distance,
            priority=priority,
            haptic=self.haptic_pattern(priority, pulse),
            tone=self.tone_name(priority, nearby.bearing),
        )

    def process(self, stage: ModuleStage, ctx: FrameContext) -> None:
        if stage != ModuleStage.ACCESSIBILITY or ctx.guidance is None:
            return
        nearby: List[NearbyObservation] = ctx.shared.get("nearby", [])
        speech: List[str] = []
        haptics: List[str] = []
        audio: List[SpatialCue] = []
        guide = ctx.guidance
        comfort_directive = ctx.shared.get("comfort_directive", {})
        memory = ctx.shared.get("spatial_memory", {})
        predicted: List[PredictedHazard] = ctx.shared.get("predicted_hazards", [])

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

        if isinstance(comfort_directive, dict):
            load = float(comfort_directive.get("comfort_load", 0.0))
            if load > 0.65:
                speech.append("Comfort shield active. Short turns and smaller steps recommended.")
                haptics.append("comfort_shield:slow:double")
            elif load < 0.25 and guide.mode != GuidanceMode.RECOVER:
                speech.append("Comfort stable. Pace can increase.")

        if predicted:
            top_pred = predicted[0]
            if top_pred.risk > 0.62:
                speech.append(
                    f"Predictive warning: {top_pred.kind.value} crossing in {top_pred.time_ahead:.1f} seconds."
                )
                haptics.append("predictive_warning:rapid:double")

        if isinstance(memory, dict) and memory.get("anchor_name"):
            speech.append(
                f"Memory anchor: {memory['anchor_name']} {memory.get('anchor_relative', 'ahead')}."
            )

        teleport_state = ctx.shared.get("teleport_state", {})
        pending = teleport_state.get("pending") if isinstance(teleport_state, dict) else None
        if isinstance(pending, dict) and pending.get("valid"):
            distance = float(pending.get("distance", 0.0))
            speech.append(f"Teleport ready, {distance:.1f} meters ahead.")
            haptics.append(f"teleport_ready:{guide.pulse:.2f}:double")
            audio.append(SpatialCue(
                text=f"teleport target ahead, {distance:.1f} meters",
                azimuth=0.0,
                distance=distance,
                priority=CuePriority.GUIDE,
                haptic=f"teleport_ready:{guide.pulse:.2f}:double",
                tone="teleport_chime_center",
            ))
        elif isinstance(pending, dict) and pending.get("reason") and not pending.get("valid"):
            speech.append(f"Teleport blocked, {pending.get('reason')}.")
            haptics.append(f"teleport_blocked:{guide.pulse:.2f}:triple")

        if ctx.shared.get("teleport_committed"):
            speech.insert(0, "Teleport complete.")
            haptics.insert(0, "teleport_commit:strong:single")

        for item in nearby[:4]:
            cue = self.make_object_cue(item, guide.pulse)
            audio.append(cue)
            if cue.priority in {CuePriority.CRITICAL, CuePriority.WARNING, CuePriority.GUIDE}:
                speech.append(cue.text)
                haptics.append(cue.haptic)

        dedup: List[str] = []
        seen = set()
        for line in speech:
            if line not in seen:
                dedup.append(line)
                seen.add(line)
        guide.speech = dedup[:6]
        guide.haptics = haptics[:6]
        guide.spatial_audio = audio[:5]


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------


class Backend:
    def capabilities(self) -> Dict[str, Any]:
        raise NotImplementedError

    def frames(self, seconds: float, hz: float) -> Iterable[SensorFrame]:
        raise NotImplementedError


class SimulatorBackend(Backend):
    def __init__(self, seed: int = 7) -> None:
        self.rng = random.Random(seed)

    def capabilities(self) -> Dict[str, Any]:
        return {
            "backend": "simulator",
            "deterministic": True,
            "notes": [
                "Deterministic simulator backend active.",
                "Use this to tune the guidance and accessibility stack before wiring a real OpenXR frame loop.",
            ],
        }

    def frames(self, seconds: float, hz: float) -> Iterable[SensorFrame]:
        count = max(1, int(seconds * hz))
        prev_t = 0.0
        for i in range(count):
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
            left = HandState(
                tracked=True,
                pinch=clamp(0.22 + 0.22 * math.sin(t * 1.3), 0.0, 1.0),
                aim=Pose(position=Vec3(-0.22, 1.35, -0.35), forward=Vec3(-0.1, 0.0, -1.0).normalized(), confidence=0.8),
                joint_energy=clamp(0.35 + 0.22 * math.sin(t * 2.2), 0.0, 1.0),
            )
            right = HandState(
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
            yield SensorFrame(t=t, dt=dt, head=head, left_hand=left, right_hand=right, gaze=gaze, affect=affect)


class AdaptiveOpenXRBackend(Backend):
    """A real-runtime-shaped backend that currently falls back to simulator frames.

    It uses runtime discovery and extension probing, but keeps actual rendering and
    native session creation outside the default execution path so the script stays
    runnable on ordinary machines.
    """

    def __init__(self, runtime_model: OpenXRSystemModel, seed: int = 17) -> None:
        self.runtime_model = runtime_model
        self.sim = SimulatorBackend(seed=seed)

    def capabilities(self) -> Dict[str, Any]:
        return {
            "backend": "adaptive-openxr",
            "runtime_model": self.runtime_model.to_dict(),
            "notes": [
                "Using OpenXR-shaped adaptive backend.",
                "Sensor frames currently come from simulator data unless a real runtime path is explicitly added.",
            ],
        }

    def frames(self, seconds: float, hz: float) -> Iterable[SensorFrame]:
        for frame in self.sim.frames(seconds, hz):
            if not self.runtime_model.extension_registry.hand_tracking:
                frame.left_hand.tracked = False
                frame.right_hand.tracked = False
                frame.left_hand.pinch = 0.0
                frame.right_hand.pinch = 0.0
                frame.left_hand.joint_energy = 0.0
                frame.right_hand.joint_energy = 0.0
            if not self.runtime_model.extension_registry.eye_gaze:
                frame.gaze.tracked = False
                frame.gaze.stability = 0.0
            yield frame


# ---------------------------------------------------------------------------
# Logging
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


# ---------------------------------------------------------------------------
# Main engine
# ---------------------------------------------------------------------------


class AegisGlyphEngine:
    def __init__(self, backend_mode: str, log_path: Optional[Path], seed: int = 7) -> None:
        self.seed = seed
        self.backend_mode = backend_mode
        self.log = JSONLLogger(log_path)
        self.security = SecurityBoundary(SecurityPolicy())
        self.runtime = OpenXRSessionFacade()
        self.actions = ActionRegistry()
        self.modules = ModuleHost()
        self.profiler = Profiler()
        self.backend: Backend = SimulatorBackend(seed=seed)
        self.teleport = TeleportState()
        self.history_total_ms: List[float] = []

        self.modules.register(WorldPopulationModule())
        self.modules.register(FilterAndAffectModule())
        self.modules.register(PredictiveHazardModule())
        self.modules.register(GuidancePlannerModule())
        self.modules.register(SpatialMemoryModule())
        self.modules.register(TeleportationModule())
        self.modules.register(ComfortAdaptationModule())
        self.modules.register(AccessibilitySynthesisModule())

    def choose_backend(self) -> None:
        model = self.runtime.initialize()
        model.notes.append(f"Host OS: {platform.system()} {platform.release()}")
        self.runtime.advance_to_session_ready()
        if self.backend_mode == "sim":
            self.backend = SimulatorBackend(seed=self.seed)
        elif self.backend_mode == "probe":
            self.backend = AdaptiveOpenXRBackend(model, seed=self.seed + 10)
        else:
            if model.pyopenxr_available:
                self.backend = AdaptiveOpenXRBackend(model, seed=self.seed + 10)
            else:
                self.backend = SimulatorBackend(seed=self.seed)
        self.modules.boot(self)

    def boot(self) -> None:
        self.choose_backend()
        self.runtime.begin()
        backend_caps = self.backend.capabilities()
        runtime_payload = self.security.redact_runtime(self.runtime.model.to_dict())
        self.log.write({
            "event": "boot",
            "backend_caps": backend_caps,
            "runtime": runtime_payload,
            "actions": self.actions.to_dict(),
        })

    def shutdown(self) -> None:
        self.runtime.stop()
        self.runtime.finalize()
        summary = {}
        if self.history_total_ms:
            summary = {
                "avg_ms": round(statistics.mean(self.history_total_ms), 4),
                "max_ms": round(max(self.history_total_ms), 4),
                "min_ms": round(min(self.history_total_ms), 4),
            }
        self.log.write({
            "event": "shutdown",
            "runtime_state": self.runtime.model.lifecycle_state.value,
            "performance_summary": summary,
        })
        self.log.close()

    def run(self, seconds: float, hz: float, quiet: bool = False) -> None:
        self.boot()
        if not quiet:
            print("== AegisGlyph OpenXR v3 Evolved ==")
            print(json.dumps(self.security.redact_runtime(self.runtime.model.to_dict()), indent=2))
            print()

        for frame_index, sensor in enumerate(self.backend.frames(seconds=seconds, hz=hz)):
            metrics = FrameMetrics(frame_index=frame_index, dt=sensor.dt)
            scene = SceneGraph()
            ctx = FrameContext(
                frame_index=frame_index,
                runtime=self.runtime.model,
                actions=self.actions,
                sensor=sensor,
                scene=scene,
                frame_metrics=metrics,
            )

            for stage, span_name in [
                (ModuleStage.PRE_FRAME, "pre_frame"),
                (ModuleStage.BUILD_SCENE, "build_scene"),
                (ModuleStage.ANALYZE, "analyze"),
                (ModuleStage.GUIDE, "guide"),
                (ModuleStage.ACCESSIBILITY, "accessibility"),
                (ModuleStage.POST_FRAME, "post_frame"),
            ]:
                self.profiler.start(span_name)
                self.modules.run_stage(stage, ctx)
                self.profiler.stop(metrics, span_name)

            if ctx.guidance is None:
                ctx.guidance = GuidanceFrame(
                    t=sensor.t,
                    mode=GuidanceMode.RECOVER,
                    cue_word="hold",
                    turn_bias=0.0,
                    pulse=0.12,
                    stability=0.0,
                    energy=0.0,
                    comfort=0.0,
                    clutter_removed=0.0,
                    target=None,
                    nearest_object=None,
                    notes=["Guidance pipeline produced no output; forced fallback frame."],
                )

            total_ms = metrics.total_ms()
            self.history_total_ms.append(total_ms)
            if total_ms > (1000.0 / max(1.0, hz)):
                metrics.warnings.append("Frame time exceeded nominal budget.")

            event = {
                "event": "frame",
                "sensor": self.security.redact_sensor(sensor.to_dict()),
                "guidance": ctx.guidance.to_dict(),
                "teleport": ctx.shared.get("teleport_state", self.teleport.to_dict()),
                "nearby": [item.to_dict() for item in ctx.shared.get("nearby", [])[:6]],
                "predicted_hazards": [p.to_dict() for p in ctx.shared.get("predicted_hazards", [])[:6]],
                "comfort_directive": ctx.shared.get("comfort_directive", {}),
                "spatial_memory": ctx.shared.get("spatial_memory", {}),
                "metrics": metrics.to_dict(),
            }
            self.log.write(event)

            if not quiet:
                lead = ctx.guidance.speech[0] if ctx.guidance.speech else ""
                print(
                    f"t={sensor.t:5.2f}s  state={self.runtime.model.lifecycle_state.value:13s}  "
                    f"mode={ctx.guidance.mode.value:7s}  cue={ctx.guidance.cue_word:7s}  "
                    f"turn={ctx.guidance.turn_bias:+.2f}  pulse={ctx.guidance.pulse:.2f}  "
                    f"target={ctx.guidance.target or '-':14s}  near={ctx.guidance.nearest_object or '-':14s}  "
                    f"tp={self.teleport.commit_count:02d}  cpu={metrics.total_ms():6.2f}ms  say={lead}"
                )

        self.shutdown()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AegisGlyph OpenXR v3 Evolved")
    parser.add_argument("--mode", choices=["auto", "probe", "sim"], default="auto")
    parser.add_argument("--seconds", type=float, default=8.0)
    parser.add_argument("--hz", type=float, default=12.0)
    parser.add_argument("--log", type=str, default="")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    log_path = Path(args.log).expanduser() if args.log else None
    engine = AegisGlyphEngine(backend_mode=args.mode, log_path=log_path, seed=args.seed)
    engine.run(seconds=max(0.5, args.seconds), hz=max(1.0, args.hz), quiet=args.quiet)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
