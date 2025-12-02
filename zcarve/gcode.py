"""G-code generation for GRBL controllers."""

from dataclasses import dataclass
from enum import Enum
from io import StringIO
from pathlib import Path
from typing import Optional

import numpy as np

from .tools import Tool, EndMill, VBit


class Units(Enum):
    MM = "mm"
    INCH = "inch"


@dataclass
class GCodeSettings:
    """Settings for G-code generation."""
    units: Units = Units.MM
    safe_z: float = 5.0  # Safe retract height
    feed_rate: Optional[float] = None  # Override tool feed rate
    plunge_rate: Optional[float] = None  # Override tool plunge rate
    spindle_rpm: Optional[int] = None  # Override tool RPM


class GCodeBuilder:
    """Builds G-code for GRBL controllers."""
    
    def __init__(self, settings: Optional[GCodeSettings] = None):
        self.settings = settings or GCodeSettings()
        self.buffer = StringIO()
        self._spindle_running: bool = False
        # Position tracking to avoid redundant moves
        self._pos_x: Optional[float] = None
        self._pos_y: Optional[float] = None
        self._pos_z: Optional[float] = None
    
    def _coords_equal(self, a: Optional[float], b: float) -> bool:
        """Check if coordinates are equal within tolerance."""
        if a is None:
            return False
        return abs(a - b) < 0.0001
    
    def _write(self, line: str) -> None:
        """Write a line of G-code."""
        self.buffer.write(line + "\n")
    
    def add_line(self, line: str) -> "GCodeBuilder":
        """Add a raw G-code line."""
        self._write(line)
        return self
    
    def _format_coord(self, value: float) -> str:
        """Format a coordinate value."""
        return f"{value:.4f}".rstrip("0").rstrip(".")
    
    def header(self, comment: str = "") -> "GCodeBuilder":
        """Write G-code header."""
        self._write(f"; ZCarve G-code")
        if comment:
            self._write(f"; {comment}")
        self._write("")
        
        # Set units
        if self.settings.units == Units.MM:
            self._write("G21 ; Units: mm")
        else:
            self._write("G20 ; Units: inches")
        
        self._write("G90 ; Absolute positioning")
        self._write("G17 ; XY plane selection")
        self._write("")
        
        return self
    
    def footer(self) -> "GCodeBuilder":
        """Write G-code footer."""
        self._write("")
        self.spindle_off()
        self.rapid_z(self.settings.safe_z)
        self._write("G0 X0 Y0 ; Return to origin")
        self._write("M30 ; Program end")
        return self
    
    def comment(self, text: str) -> "GCodeBuilder":
        """Add a comment."""
        self._write(f"; {text}")
        return self
    
    def spindle_on(self, rpm: int) -> "GCodeBuilder":
        """Turn spindle on at specified RPM."""
        self._write(f"M3 S{rpm} ; Spindle on CW")
        self._write("G4 P2 ; Dwell 2 seconds for spindle")
        self._spindle_running = True
        return self
    
    def spindle_off(self) -> "GCodeBuilder":
        """Turn spindle off."""
        if self._spindle_running:
            self._write("M5 ; Spindle off")
            self._spindle_running = False
        return self
    
    def rapid_xy(self, x: float, y: float) -> "GCodeBuilder":
        """Rapid move to XY position."""
        if self._coords_equal(self._pos_x, x) and self._coords_equal(self._pos_y, y):
            return self  # Skip redundant move
        self._write(f"G0 X{self._format_coord(x)} Y{self._format_coord(y)}")
        self._pos_x = x
        self._pos_y = y
        return self
    
    def rapid_z(self, z: float) -> "GCodeBuilder":
        """Rapid move to Z position."""
        if self._coords_equal(self._pos_z, z):
            return self  # Skip redundant move
        self._write(f"G0 Z{self._format_coord(z)}")
        self._pos_z = z
        return self
    
    def plunge(self, z: float, feed: float) -> "GCodeBuilder":
        """Plunge to Z at specified feed rate."""
        if self._coords_equal(self._pos_z, z):
            return self  # Skip redundant move
        self._write(f"G1 Z{self._format_coord(z)} F{self._format_coord(feed)}")
        self._pos_z = z
        return self
    
    def linear_xy(self, x: float, y: float, feed: float) -> "GCodeBuilder":
        """Linear move to XY at specified feed rate."""
        self._write(f"G1 X{self._format_coord(x)} Y{self._format_coord(y)} F{self._format_coord(feed)}")
        self._pos_x = x
        self._pos_y = y
        return self
    
    def linear_xyz(self, x: float, y: float, z: float, feed: float) -> "GCodeBuilder":
        """Linear move to XYZ at specified feed rate."""
        self._write(
            f"G1 X{self._format_coord(x)} Y{self._format_coord(y)} "
            f"Z{self._format_coord(z)} F{self._format_coord(feed)}"
        )
        self._pos_x = x
        self._pos_y = y
        self._pos_z = z
        return self
    
    def cut_path(
        self,
        points: np.ndarray,
        z_depth: float,
        tool: Tool,
        close_path: bool = False,
    ) -> "GCodeBuilder":
        """Cut along a path of points at specified depth.
        
        Args:
            points: Nx2 array of (x, y) coordinates
            z_depth: Cutting depth (negative for below surface)
            tool: Tool to use for feeds/speeds
            close_path: If True, return to first point at end
        """
        if len(points) < 2:
            return self
        
        feed = self.settings.feed_rate or tool.feed_rate
        plunge = self.settings.plunge_rate or tool.plunge_rate
        
        # Move to start position
        self.rapid_z(self.settings.safe_z)
        self.rapid_xy(points[0, 0], points[0, 1])
        
        # Plunge to depth
        self.plunge(z_depth, plunge)
        
        # Cut along path
        for point in points[1:]:
            self.linear_xy(point[0], point[1], feed)
        
        # Close path if requested
        if close_path:
            self.linear_xy(points[0, 0], points[0, 1], feed)
        
        # Retract
        self.rapid_z(self.settings.safe_z)
        
        return self
    
    def get_gcode(self) -> str:
        """Get the generated G-code as a string."""
        return self.buffer.getvalue()
    
    def save(self, path: Path) -> None:
        """Save G-code to a file."""
        with open(path, "w") as f:
            f.write(self.get_gcode())
    
    def reset(self) -> None:
        """Reset the builder for new G-code."""
        self.buffer = StringIO()
        self._spindle_running = False
        self._pos_x = None
        self._pos_y = None
        self._pos_z = None
