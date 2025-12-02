"""Geometry handling and SVG parsing."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from svgpathtools import svg2paths2, Path as SvgPath, Line, CubicBezier, QuadraticBezier, Arc
from shapely.geometry import Polygon, MultiPolygon, LineString
from shapely.ops import unary_union


@dataclass
class BoundingBox:
    """Axis-aligned bounding box."""
    min_x: float
    min_y: float
    max_x: float
    max_y: float
    
    @property
    def width(self) -> float:
        return self.max_x - self.min_x
    
    @property
    def height(self) -> float:
        return self.max_y - self.min_y
    
    @property
    def center(self) -> tuple[float, float]:
        return (
            (self.min_x + self.max_x) / 2,
            (self.min_y + self.max_y) / 2,
        )


@dataclass 
class CarvePath:
    """A path to be carved, with metadata."""
    id: str
    points: np.ndarray  # Nx2 array of (x, y) coordinates
    is_closed: bool
    layer: Optional[str] = None
    color: Optional[str] = None
    
    @property
    def bounds(self) -> BoundingBox:
        return BoundingBox(
            min_x=float(self.points[:, 0].min()),
            min_y=float(self.points[:, 1].min()),
            max_x=float(self.points[:, 0].max()),
            max_y=float(self.points[:, 1].max()),
        )
    
    def to_shapely(self) -> LineString | Polygon:
        """Convert to Shapely geometry."""
        coords = [(p[0], p[1]) for p in self.points]
        if self.is_closed and len(coords) >= 3:
            return Polygon(coords)
        return LineString(coords)


class SVGLoader:
    """Load and parse SVG files into carve paths."""
    
    # Number of points to sample along curves
    CURVE_SAMPLES = 20
    
    def __init__(self):
        self.paths: list[CarvePath] = []
        self.bounds: Optional[BoundingBox] = None
    
    def load(self, svg_path: Path) -> list[CarvePath]:
        """Load an SVG file and extract paths."""
        self.paths = []
        
        paths, attributes, svg_attributes = svg2paths2(str(svg_path))
        
        for i, (svg_path_obj, attrs) in enumerate(zip(paths, attributes)):
            points = self._path_to_points(svg_path_obj)
            if len(points) < 2:
                continue
            
            # Check if path is closed
            is_closed = svg_path_obj.isclosed()
            
            # Extract metadata
            path_id = attrs.get("id", f"path_{i}")
            layer = self._extract_layer(attrs)
            color = self._extract_color(attrs)
            
            carve_path = CarvePath(
                id=path_id,
                points=points,
                is_closed=is_closed,
                layer=layer,
                color=color,
            )
            self.paths.append(carve_path)
        
        self._compute_bounds()
        return self.paths
    
    def _path_to_points(self, svg_path: SvgPath) -> np.ndarray:
        """Convert an SVG path to a numpy array of points."""
        points = []
        
        for segment in svg_path:
            if isinstance(segment, Line):
                # Just need start point; end point comes from next segment
                points.append((segment.start.real, segment.start.imag))
            elif isinstance(segment, (CubicBezier, QuadraticBezier, Arc)):
                # Sample points along the curve
                for t in np.linspace(0, 1, self.CURVE_SAMPLES, endpoint=False):
                    pt = segment.point(t)
                    points.append((pt.real, pt.imag))
        
        # Add final point
        if svg_path:
            end = svg_path[-1].end
            points.append((end.real, end.imag))
        
        if not points:
            return np.array([]).reshape(0, 2)
        
        return np.array(points)
    
    def _extract_layer(self, attrs: dict) -> Optional[str]:
        """Extract layer name from SVG attributes."""
        # Inkscape uses inkscape:label for layer names
        for key in attrs:
            if "label" in key.lower():
                return attrs[key]
        return None
    
    def _extract_color(self, attrs: dict) -> Optional[str]:
        """Extract stroke/fill color from SVG attributes."""
        style = attrs.get("style", "")
        
        # Parse style attribute
        for part in style.split(";"):
            if ":" in part:
                key, value = part.split(":", 1)
                key = key.strip().lower()
                value = value.strip()
                if key in ("stroke", "fill") and value != "none":
                    return value
        
        # Check direct attributes
        for attr in ("stroke", "fill"):
            if attr in attrs and attrs[attr] != "none":
                return attrs[attr]
        
        return None
    
    def _compute_bounds(self) -> None:
        """Compute overall bounding box of all paths."""
        if not self.paths:
            self.bounds = None
            return
        
        all_points = np.vstack([p.points for p in self.paths])
        self.bounds = BoundingBox(
            min_x=float(all_points[:, 0].min()),
            min_y=float(all_points[:, 1].min()),
            max_x=float(all_points[:, 0].max()),
            max_y=float(all_points[:, 1].max()),
        )


def create_offset_polygon(paths: list[CarvePath], offset: float) -> MultiPolygon | Polygon:
    """Create an offset (inset/outset) polygon from closed paths.
    
    Args:
        paths: List of closed carve paths
        offset: Positive for outset, negative for inset
    
    Returns:
        Offset polygon geometry
    """
    polygons = []
    for path in paths:
        if path.is_closed:
            poly = path.to_shapely()
            if isinstance(poly, Polygon) and poly.is_valid:
                polygons.append(poly)
    
    if not polygons:
        return MultiPolygon()
    
    combined = unary_union(polygons)
    return combined.buffer(offset, join_style=2)  # mitre join
