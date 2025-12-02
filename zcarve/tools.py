"""Tool library management for end mills and v-bits."""

import json
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Optional


class ToolType(Enum):
    ENDMILL = "endmill"
    VBIT = "vbit"


@dataclass
class EndMill:
    """Flat end mill for roughing/clearing."""
    id: str
    name: str
    diameter: float  # mm
    feed_rate: float  # mm/min
    plunge_rate: float  # mm/min
    spindle_rpm: int
    flute_count: int = 2
    stepover_percent: float = 40.0  # percentage of diameter
    max_depth_per_pass: float = 2.0  # mm
    tool_type: ToolType = field(default=ToolType.ENDMILL)
    
    @property
    def stepover(self) -> float:
        """Calculate stepover distance in mm."""
        return self.diameter * (self.stepover_percent / 100.0)
    
    def to_dict(self) -> dict:
        d = asdict(self)
        d["tool_type"] = self.tool_type.value
        return d


@dataclass
class VBit:
    """V-bit for v-carving."""
    id: str
    name: str
    diameter: float  # mm
    feed_rate: float  # mm/min
    plunge_rate: float  # mm/min
    spindle_rpm: int
    angle: float = 60.0  # included angle in degrees
    tip_diameter: float = 0.0  # flat tip diameter, 0 for sharp
    tool_type: ToolType = field(default=ToolType.VBIT)
    
    def depth_for_width(self, width: float) -> float:
        """Calculate cutting depth needed for a given carved width."""
        import math
        half_angle = math.radians(self.angle / 2)
        effective_width = width - self.tip_diameter
        if effective_width <= 0:
            return 0.0
        return (effective_width / 2) / math.tan(half_angle)
    
    def to_dict(self) -> dict:
        d = asdict(self)
        d["tool_type"] = self.tool_type.value
        return d


# Type alias for any tool
Tool = EndMill | VBit


def tool_from_dict(data: dict) -> Tool:
    """Create a tool from a dictionary."""
    data = data.copy()
    tool_type = ToolType(data.pop("tool_type"))
    
    if tool_type == ToolType.ENDMILL:
        return EndMill(**data, tool_type=tool_type)
    elif tool_type == ToolType.VBIT:
        return VBit(**data, tool_type=tool_type)
    raise ValueError(f"Unknown tool type: {tool_type}")


class ToolLibrary:
    """Persistent storage for tools."""
    
    DEFAULT_TOOLS = [
        EndMill(
            id="em-6mm-2f",
            name="6mm 2-Flute End Mill",
            diameter=6.0,
            feed_rate=1000.0,
            plunge_rate=300.0,
            spindle_rpm=12000,
            flute_count=2,
            stepover_percent=40.0,
            max_depth_per_pass=2.0,
        ),
        EndMill(
            id="em-3mm-2f",
            name="3mm 2-Flute End Mill",
            diameter=3.0,
            feed_rate=800.0,
            plunge_rate=200.0,
            spindle_rpm=15000,
            flute_count=2,
            stepover_percent=40.0,
            max_depth_per_pass=1.5,
        ),
        VBit(
            id="vb-60deg",
            name="60° V-Bit",
            diameter=12.0,
            feed_rate=600.0,
            plunge_rate=150.0,
            spindle_rpm=12000,
            angle=60.0,
            tip_diameter=0.0,
        ),
        VBit(
            id="vb-90deg",
            name="90° V-Bit",
            diameter=12.0,
            feed_rate=500.0,
            plunge_rate=120.0,
            spindle_rpm=10000,
            angle=90.0,
            tip_diameter=0.0,
        ),
    ]
    
    def __init__(self, library_path: Optional[Path] = None):
        if library_path is None:
            config_dir = Path.home() / ".config" / "zcarve"
            config_dir.mkdir(parents=True, exist_ok=True)
            library_path = config_dir / "tools.json"
        
        self.library_path = library_path
        self.tools: dict[str, Tool] = {}
        self.load()
    
    def load(self) -> None:
        """Load tools from disk, or initialize with defaults."""
        if self.library_path.exists():
            try:
                with open(self.library_path, "r") as f:
                    data = json.load(f)
                    for tool_data in data.get("tools", []):
                        tool = tool_from_dict(tool_data)
                        self.tools[tool.id] = tool
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Warning: Could not load tool library: {e}")
                self._init_defaults()
        else:
            self._init_defaults()
    
    def _init_defaults(self) -> None:
        """Initialize with default tools."""
        for tool in self.DEFAULT_TOOLS:
            self.tools[tool.id] = tool
        self.save()
    
    def save(self) -> None:
        """Persist tools to disk."""
        data = {"tools": [tool.to_dict() for tool in self.tools.values()]}
        with open(self.library_path, "w") as f:
            json.dump(data, f, indent=2)
    
    def add(self, tool: Tool) -> None:
        """Add or update a tool."""
        self.tools[tool.id] = tool
        self.save()
    
    def remove(self, tool_id: str) -> None:
        """Remove a tool by ID."""
        if tool_id in self.tools:
            del self.tools[tool_id]
            self.save()
    
    def get(self, tool_id: str) -> Optional[Tool]:
        """Get a tool by ID."""
        return self.tools.get(tool_id)
    
    def get_endmills(self) -> list[EndMill]:
        """Get all end mills."""
        return [t for t in self.tools.values() if isinstance(t, EndMill)]
    
    def get_vbits(self) -> list[VBit]:
        """Get all v-bits."""
        return [t for t in self.tools.values() if isinstance(t, VBit)]
