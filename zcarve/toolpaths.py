"""Toolpath generation for roughing and v-carving."""

from dataclasses import dataclass
from enum import Enum
from typing import Iterator

import numpy as np
from shapely.geometry import Polygon, MultiPolygon, LineString, MultiLineString, Point
from shapely.ops import unary_union

from .geometry import CarvePath
from .tools import EndMill, VBit


class ClearingStrategy(Enum):
    """Strategy for pocket clearing."""
    ADAPTIVE = "adaptive"  # Adaptive HSM - constant engagement with smooth curves
    CONTOUR = "contour"  # Contour-parallel clearing
    RASTER = "raster"  # Parallel lines (zigzag pattern)


@dataclass
class RoughingParams:
    """Parameters for roughing pass generation."""
    tool: EndMill
    target_depth: float  # Total depth to cut (positive value, will be negated)
    stock_to_leave: float = 0.0  # Material to leave for finishing pass
    climb_milling: bool = True  # True for climb, False for conventional
    strategy: ClearingStrategy = ClearingStrategy.CONTOUR  # Clearing strategy


@dataclass 
class Toolpath:
    """A single toolpath segment."""
    points: np.ndarray  # Nx2 array of (x, y) or Nx3 for (x, y, z) ramps
    z_depth: float  # Cutting depth (negative), ignored if points has Z
    is_rapid: bool = False  # True if this is a rapid move
    is_arc: bool = False  # True if this is an arc move (G2/G3)
    arc_center: tuple[float, float] | None = None  # (I, J) offset for arcs
    clockwise: bool = True  # True for G2, False for G3
    

def generate_roughing_toolpaths(
    paths: list[CarvePath],
    params: RoughingParams,
) -> list[Toolpath]:
    """Generate roughing toolpaths for pocket clearing.
    
    Args:
        paths: Closed paths defining the pocket boundaries
        params: Roughing parameters
        
    Returns:
        List of toolpaths for roughing
    """
    tool = params.tool
    
    # Combine all closed paths into pocket geometry
    polygons = []
    for path in paths:
        if path.is_closed and len(path.points) >= 3:
            try:
                poly = Polygon([(p[0], p[1]) for p in path.points])
                if poly.is_valid:
                    polygons.append(poly)
            except Exception:
                continue
    
    if not polygons:
        return []
    
    pocket = unary_union(polygons)
    
    # Offset inward by tool radius + stock to leave
    offset = -(tool.diameter / 2 + params.stock_to_leave)
    pocket_offset = pocket.buffer(offset, join_style=2)
    
    if pocket_offset.is_empty:
        return []
    
    # Calculate depth passes
    depth_passes = _calculate_depth_passes(
        params.target_depth, 
        tool.max_depth_per_pass
    )
    
    # Dispatch to appropriate strategy
    if params.strategy == ClearingStrategy.ADAPTIVE:
        return _generate_adaptive_hsm(
            pocket_offset,
            tool.stepover,
            tool.diameter,
            depth_passes,
            params.climb_milling,
        )
    elif params.strategy == ClearingStrategy.CONTOUR:
        return _generate_contour_parallel(
            pocket_offset,
            tool.stepover,
            depth_passes,
            params.climb_milling,
        )
    else:  # RASTER
        toolpaths = []
        for depth in depth_passes:
            pass_toolpaths = _generate_raster_pattern(
                pocket_offset,
                tool.stepover,
                depth,
                params.climb_milling,
            )
            toolpaths.extend(pass_toolpaths)
        return toolpaths


def _calculate_depth_passes(target_depth: float, max_per_pass: float) -> list[float]:
    """Calculate the Z depths for each pass.
    
    Returns list of negative Z values, shallowest to deepest.
    """
    if target_depth <= 0:
        return []
    
    passes = []
    remaining = target_depth
    current_depth = 0.0
    
    while remaining > 0.001:  # Small epsilon for float comparison
        cut = min(remaining, max_per_pass)
        current_depth += cut
        passes.append(-current_depth)
        remaining -= cut
    
    return passes


def _generate_adaptive_hsm(
    polygon: Polygon | MultiPolygon,
    stepover: float,
    tool_diameter: float,
    depth_passes: list[float],
    climb_milling: bool,
) -> list[Toolpath]:
    """Generate Adaptive High Speed Machining toolpaths.
    
    Key features:
    - Constant tool engagement (optimal load)
    - Smooth curves around corners (no engagement spikes)
    - Safe Z retracts between spiral and arms, and between arms
    """
    toolpaths = []
    
    for z_depth in depth_passes:
        # Get segments - already separated by region (spiral + each arm)
        segments = _generate_adaptive_level(
            polygon, stepover, tool_diameter, climb_milling
        )
        
        # Each segment is a distinct region - create separate toolpath for each
        # This ensures safe Z retracts between spiral and arms, and between arms
        for segment in segments:
            if len(segment) >= 2:
                toolpaths.append(Toolpath(
                    points=np.array(segment),
                    z_depth=z_depth,
                    is_rapid=False,
                ))
    
    return toolpaths


def _generate_adaptive_level(
    polygon: Polygon | MultiPolygon,
    stepover: float,
    tool_diameter: float,
    climb_milling: bool,
) -> list[list[tuple[float, float]]]:
    """Generate adaptive HSM path using triangulation approach.
    
    Decomposes the pocket into center region + arm regions.
    Returns separate segments for safe Z retracts between them.
    """
    import math
    
    tool_radius = tool_diameter / 2
    min_radius = tool_diameter * 0.3
    
    # Safe cutting area
    safe_area = polygon.buffer(-tool_radius, join_style=1, resolution=16)
    if safe_area.is_empty:
        return []
    
    segments = []  # List of separate toolpath segments
    pts_per_circle = 24
    
    # Find the centroid and largest inscribed circle
    centroid = safe_area.centroid
    cx, cy = centroid.x, centroid.y
    
    # Find max radius for center (distance to nearest boundary)
    center_pt = Point(cx, cy)
    if isinstance(safe_area, Polygon):
        max_center_radius = safe_area.exterior.distance(center_pt)
    else:
        max_center_radius = min(p.exterior.distance(center_pt) for p in safe_area.geoms)
    
    # PHASE 1: Clear center with circular spiral (separate segment)
    if max_center_radius > min_radius:
        spiral_pts = _generate_center_spiral(
            cx, cy, max_center_radius, stepover, climb_milling
        )
        if spiral_pts:
            segments.append(spiral_pts)
    
    # PHASE 2: Decompose remaining area into arm regions
    center_cleared = Point(cx, cy).buffer(max_center_radius)
    remaining = safe_area.difference(center_cleared)
    
    if not remaining.is_empty:
        # Get arm regions
        arm_regions = _get_arm_regions(remaining)
        
        # Sort arms by angle for efficient routing
        arm_regions.sort(key=lambda r: math.atan2(
            r.centroid.y - cy, r.centroid.x - cx
        ))
        
        # Clear each arm region
        for arm in arm_regions:
            if arm.is_empty or arm.area < min_radius * min_radius:
                continue
            
            arm_pts = _clear_arm_with_circles(
                arm, polygon, cx, cy, tool_radius, min_radius, stepover, 
                pts_per_circle, climb_milling, max_center_radius
            )
            if arm_pts:
                segments.append(arm_pts)
    
    return segments


def _get_arm_regions(remaining) -> list:
    """Extract individual arm regions from the remaining geometry.
    
    If remaining is a single connected region (like a star ring around center),
    split it into separate wedges for each arm.
    """
    from shapely.geometry import GeometryCollection
    import math
    
    regions = []
    if isinstance(remaining, Polygon):
        regions = [remaining]
    elif isinstance(remaining, MultiPolygon):
        regions = list(remaining.geoms)
    elif isinstance(remaining, GeometryCollection):
        for geom in remaining.geoms:
            if isinstance(geom, Polygon) and not geom.is_empty:
                regions.append(geom)
            elif isinstance(geom, MultiPolygon):
                regions.extend([p for p in geom.geoms if not p.is_empty])
    
    # If we have a single region, try to split it into radial wedges
    if len(regions) == 1 and isinstance(regions[0], Polygon):
        region = regions[0]
        split = _split_into_radial_wedges(region)
        if len(split) > 1:
            return split
    
    return [r for r in regions if not r.is_empty]


def _split_into_radial_wedges(region: Polygon, num_wedges: int = 8) -> list:
    """Split a region into radial wedges from its centroid.
    
    This separates a ring-like region (e.g., star arms around center)
    into individual wedge-shaped pieces.
    """
    import math
    
    centroid = region.centroid
    cx, cy = centroid.x, centroid.y
    
    # Find the maximum extent
    coords = list(region.exterior.coords)
    max_dist = max(math.sqrt((x-cx)**2 + (y-cy)**2) for x, y in coords)
    far = max_dist * 2
    
    wedges = []
    angle_step = 2 * math.pi / num_wedges
    
    for i in range(num_wedges):
        angle1 = i * angle_step
        angle2 = (i + 1) * angle_step
        
        # Create wedge polygon
        wedge_pts = [(cx, cy)]
        for a in [angle1, (angle1 + angle2) / 2, angle2]:
            wedge_pts.append((cx + far * math.cos(a), cy + far * math.sin(a)))
        wedge_pts.append((cx, cy))
        
        wedge = Polygon(wedge_pts)
        piece = region.intersection(wedge)
        
        if not piece.is_empty and piece.area > 0.1:
            if isinstance(piece, Polygon):
                wedges.append(piece)
            elif isinstance(piece, MultiPolygon):
                wedges.extend([p for p in piece.geoms if not p.is_empty])
    
    return wedges if wedges else [region]


def _split_toolpath_at_jumps(
    points: list[tuple[float, float]], 
    max_jump: float,
    safe_area: Polygon | MultiPolygon | None = None,
) -> list[list[tuple[float, float]]]:
    """Split a toolpath into segments wherever a jump would exit the safe area.
    
    This prevents the tool from cutting through material when it should
    be retracting to safe Z and moving rapidly.
    
    Args:
        points: List of (x, y) coordinates
        max_jump: Maximum allowed distance between consecutive points
                  before checking if the jump exits safe area.
        safe_area: The cleared/safe cutting area. If None, only uses distance.
    
    Returns:
        List of point segments, each segment is a continuous toolpath.
    """
    import math
    from shapely.geometry import LineString
    
    if len(points) < 2:
        return [points] if points else []
    
    segments = []
    current_segment = [points[0]]
    
    for i in range(1, len(points)):
        prev = points[i-1]
        curr = points[i]
        
        dist = math.sqrt((curr[0] - prev[0])**2 + (curr[1] - prev[1])**2)
        
        should_split = False
        
        if dist > max_jump:
            # Check if the line between prev and curr exits the safe area
            if safe_area is not None:
                line = LineString([prev, curr])
                # If the line is not fully within safe area, we need to retract
                if not safe_area.contains(line):
                    should_split = True
            else:
                # No safe area provided, split on large jumps
                should_split = True
        
        if should_split:
            # Split here - end current segment and start new one
            if len(current_segment) >= 2:
                segments.append(current_segment)
            current_segment = [curr]
        else:
            current_segment.append(curr)
    
    # Don't forget the last segment
    if len(current_segment) >= 2:
        segments.append(current_segment)
    
    return segments


def _clear_arm_with_circles(
    arm: Polygon,
    original_pocket: Polygon,
    center_x: float,
    center_y: float,
    tool_radius: float,
    min_radius: float,
    stepover: float,
    pts_per_circle: int,
    climb_milling: bool,
    spiral_radius: float,
) -> list[tuple[float, float]]:
    """Clear an arm/corner region using offset contours.
    
    Generates offset contours of the arm region - these naturally form
    arcs that span edge-to-edge, getting smaller toward the tip.
    """
    import math
    
    points = []
    
    # Heavy smoothing to round all corners
    smooth_amount = stepover * 3.0
    smoothed = arm.buffer(-smooth_amount, join_style=1, resolution=32)
    if not smoothed.is_empty:
        smoothed = smoothed.buffer(smooth_amount, join_style=1, resolution=32)
    else:
        smoothed = arm
    
    if smoothed.is_empty:
        smoothed = arm
    
    # Generate offset contours of the smoothed arm region
    contours = []
    current = smoothed
    offset_step = stepover * 0.8
    
    while not current.is_empty:
        # Also smooth each contour level to remove any remaining sharp corners
        smooth_contour = current.buffer(-stepover * 0.3, join_style=1, resolution=16)
        if not smooth_contour.is_empty:
            smooth_contour = smooth_contour.buffer(stepover * 0.3, join_style=1, resolution=16)
            contours.append(smooth_contour)
        else:
            contours.append(current)
        
        current = current.buffer(-offset_step, join_style=1, resolution=16)
        if len(contours) > 50:
            break
    
    if not contours:
        return points
    
    # Reverse to go from outer (near spiral) to inner (corner tip)
    contours = list(reversed(contours))
    
    # Extract and connect contour paths
    for contour in contours:
        if isinstance(contour, Polygon):
            coords = list(contour.exterior.coords)
        elif isinstance(contour, MultiPolygon):
            coords = []
            for p in contour.geoms:
                coords.extend(list(p.exterior.coords))
        else:
            continue
        
        if len(coords) < 3:
            continue
        
        # Reverse for conventional milling
        if not climb_milling:
            coords = coords[::-1]
        
        # Reorder to start from point nearest to last point
        if points:
            coords = _reorder_ring_nearest(coords, points[-1])
        
        points.extend(coords)
    
    return points


def _generate_arc(
    cx: float, cy: float, radius: float,
    dir_x: float, dir_y: float,
    num_pts: int,
    climb_milling: bool,
) -> list[tuple[float, float]]:
    """Generate arc points curving perpendicular to the given direction."""
    import math
    
    points = []
    
    # Arc sweeps perpendicular to the direction (clearing the corner)
    # Start angle is perpendicular to direction
    base_angle = math.atan2(dir_y, dir_x)
    
    # Sweep 180 degrees (half circle)
    sweep = math.pi
    start_angle = base_angle - sweep/2
    
    if not climb_milling:
        start_angle = base_angle + sweep/2
        sweep = -sweep
    
    for i in range(num_pts + 1):
        t = i / num_pts
        angle = start_angle + sweep * t
        x = cx + radius * math.cos(angle)
        y = cy + radius * math.sin(angle)
        points.append((x, y))
    
    return points


def _distance_to_polygon_boundary(x: float, y: float, polygon: Polygon) -> float:
    """Calculate distance from point to polygon boundary."""
    pt = Point(x, y)
    return polygon.exterior.distance(pt)


def _reorder_ring_nearest(coords: list, ref_point: tuple) -> list:
    """Reorder a closed ring to start from the point nearest to ref_point."""
    import math
    
    if len(coords) < 2:
        return coords
    
    # Check if closed ring
    is_closed = (coords[0][0] == coords[-1][0] and coords[0][1] == coords[-1][1])
    if is_closed:
        coords = coords[:-1]
    
    if not coords:
        return []
    
    # Find nearest point
    rx, ry = ref_point
    min_dist = float('inf')
    min_idx = 0
    
    for i, (x, y) in enumerate(coords):
        d = (x - rx)**2 + (y - ry)**2
        if d < min_dist:
            min_dist = d
            min_idx = i
    
    # Reorder
    result = coords[min_idx:] + coords[:min_idx]
    
    # Close the ring
    if result:
        result.append(result[0])
    
    return result


def _get_medial_axis_for_arms(
    remaining,
    original_polygon: Polygon,
    tool_radius: float,
    stepover: float,
    center_x: float,
    center_y: float,
) -> list[tuple[float, float]]:
    """Get medial axis path through the remaining arm regions."""
    import math
    from shapely.geometry import GeometryCollection
    
    path = []
    
    # Get individual regions
    regions = []
    if isinstance(remaining, Polygon):
        regions = [remaining]
    elif isinstance(remaining, MultiPolygon):
        regions = list(remaining.geoms)
    elif isinstance(remaining, GeometryCollection):
        for geom in remaining.geoms:
            if isinstance(geom, Polygon) and not geom.is_empty:
                regions.append(geom)
    
    if not regions:
        return path
    
    # Sort regions by angle for continuous path
    def region_angle(r):
        c = r.centroid
        return math.atan2(c.y - center_y, c.x - center_x)
    
    regions = sorted(regions, key=region_angle)
    
    # For each arm, trace from base to tip
    for region in regions:
        if region.is_empty or region.area < tool_radius * tool_radius:
            continue
        
        arm_pts = _trace_arm_medial_axis(region, tool_radius, stepover, center_x, center_y)
        path.extend(arm_pts)
    
    return path


def _trace_arm_medial_axis(
    region: Polygon,
    tool_radius: float,
    stepover: float,
    center_x: float,
    center_y: float,
) -> list[tuple[float, float]]:
    """Trace medial axis of a single arm from base toward tip."""
    import math
    
    path = []
    
    # Find the direction from center toward this arm's tip
    arm_centroid = region.centroid
    dir_x = arm_centroid.x - center_x
    dir_y = arm_centroid.y - center_y
    dir_len = math.sqrt(dir_x*dir_x + dir_y*dir_y)
    
    if dir_len < 0.1:
        return [(arm_centroid.x, arm_centroid.y)]
    
    # Normalize direction
    dir_x /= dir_len
    dir_y /= dir_len
    
    # Find the extent of this arm along its direction
    # Start from near the center and step outward
    step = stepover * 0.6
    
    # Find base point (closest to center) and tip (farthest)
    coords = list(region.exterior.coords)
    
    # Project all points onto the direction vector to find extent
    min_proj = float('inf')
    max_proj = float('-inf')
    
    for x, y in coords:
        proj = (x - center_x) * dir_x + (y - center_y) * dir_y
        min_proj = min(min_proj, proj)
        max_proj = max(max_proj, proj)
    
    # Walk along the medial axis from base to tip
    current_dist = min_proj
    
    while current_dist <= max_proj:
        # Point along the medial axis
        px = center_x + dir_x * current_dist
        py = center_y + dir_y * current_dist
        
        # Check if this point is inside the arm region
        pt = Point(px, py)
        if region.contains(pt):
            path.append((px, py))
        
        current_dist += step
    
    # If no points found, at least add centroid
    if not path:
        path.append((arm_centroid.x, arm_centroid.y))
    
    return path


def _get_medial_axis_path(
    polygon: Polygon | MultiPolygon,
    tool_radius: float,
    stepover: float,
) -> list[tuple[float, float]]:
    """Get medial axis (skeleton) path through the polygon.
    
    Uses progressive erosion to find the centerline, then traces it.
    """
    import math
    
    if isinstance(polygon, MultiPolygon):
        # Handle first polygon for now
        polygon = polygon.geoms[0]
    
    # Erode progressively to find skeleton points
    skeleton_points = []
    current = polygon.buffer(-tool_radius, join_style=1, resolution=8)
    
    erosion_step = stepover * 0.5
    level_points = []
    
    while not current.is_empty:
        # Get centroid of this level
        if isinstance(current, Polygon):
            c = current.centroid
            level_points.append((c.x, c.y, current))
        elif isinstance(current, MultiPolygon):
            for p in current.geoms:
                if not p.is_empty:
                    c = p.centroid
                    level_points.append((c.x, c.y, p))
        
        current = current.buffer(-erosion_step, join_style=1, resolution=4)
    
    if not level_points:
        return []
    
    # Start from the innermost point (deepest erosion = center)
    # Then trace outward along branches
    
    # Get the innermost centroid as starting point
    innermost = level_points[-1] if level_points else None
    if not innermost:
        return []
    
    start_x, start_y = innermost[0], innermost[1]
    
    # Generate path by sampling from center outward in a spiral pattern
    path = []
    
    # Start with center
    path.append((start_x, start_y))
    
    # Expand outward in spiral, staying within polygon
    safe_area = polygon.buffer(-tool_radius * 0.5, join_style=1, resolution=8)
    if safe_area.is_empty:
        return path
    
    # Spiral outward from center, following the shape
    max_radius = _get_max_extent(polygon, start_x, start_y)
    
    angle = 0.0
    radius = stepover
    points_per_rev = 24
    step_angle = 2 * math.pi / points_per_rev
    radius_step = stepover * 0.6 / points_per_rev
    
    while radius < max_radius * 1.5:
        x = start_x + radius * math.cos(angle)
        y = start_y + radius * math.sin(angle)
        
        pt = Point(x, y)
        if safe_area.contains(pt):
            # Only add if not too close to last point
            if not path or math.sqrt((x - path[-1][0])**2 + (y - path[-1][1])**2) > stepover * 0.4:
                path.append((x, y))
        
        angle += step_angle
        radius += radius_step
        
        if len(path) > 5000:
            break
    
    return path


def _get_max_extent(polygon, cx: float, cy: float) -> float:
    """Get maximum distance from center to any point on polygon."""
    import math
    
    if isinstance(polygon, Polygon):
        coords = list(polygon.exterior.coords)
    else:
        coords = []
        for p in polygon.geoms:
            coords.extend(list(p.exterior.coords))
    
    max_r = 0.0
    for x, y in coords:
        r = math.sqrt((x - cx)**2 + (y - cy)**2)
        max_r = max(max_r, r)
    
    return max_r


def _generate_center_spiral(
    cx: float, cy: float,
    max_radius: float,
    stepover: float,
    climb_milling: bool,
) -> list[tuple[float, float]]:
    """Generate Archimedean spiral from center outward."""
    import math
    
    points = []
    points_per_rev = 32
    arc_step = stepover * 0.8  # Overlap between spiral loops
    
    num_revolutions = int(max_radius / arc_step) + 1
    total_points = num_revolutions * points_per_rev
    
    for i in range(total_points):
        r = arc_step * (i / points_per_rev)
        
        if r > max_radius:
            break
        
        if climb_milling:
            a = 2 * math.pi * (i / points_per_rev)
        else:
            a = -2 * math.pi * (i / points_per_rev)
        
        px = cx + r * math.cos(a)
        py = cy + r * math.sin(a)
        points.append((px, py))
    
    return points


def _clear_remaining_regions(
    remaining,
    original_pocket: Polygon | MultiPolygon,
    tool_radius: float,
    min_radius: float,
    stepover: float,
    climb_milling: bool,
    existing_points: list,
    spiral_end: tuple,
    spiral_cx: float,
    spiral_cy: float,
    spiral_radius: float,
) -> list[tuple[float, float]]:
    """Clear remaining regions (star arms) with trochoidal arcs and smooth transitions."""
    import math
    from shapely.geometry import GeometryCollection
    
    all_points = []
    
    # Get individual regions to clear
    regions = []
    if isinstance(remaining, Polygon):
        regions = [remaining]
    elif isinstance(remaining, MultiPolygon):
        regions = list(remaining.geoms)
    elif isinstance(remaining, GeometryCollection):
        for geom in remaining.geoms:
            if isinstance(geom, Polygon) and not geom.is_empty:
                regions.append(geom)
            elif isinstance(geom, MultiPolygon):
                regions.extend([p for p in geom.geoms if not p.is_empty])
    
    if not regions:
        return []
    
    # Sort regions by angle from center for continuous spiral-like clearing
    def region_angle(r):
        c = r.centroid
        return math.atan2(c.y - spiral_cy, c.x - spiral_cx)
    
    # Find starting angle from spiral end point
    start_angle = math.atan2(spiral_end[1] - spiral_cy, spiral_end[0] - spiral_cx)
    
    # Sort by angle relative to spiral end (continue in same direction)
    def angle_diff(r):
        a = region_angle(r)
        diff = a - start_angle
        if climb_milling:
            if diff < 0:
                diff += 2 * math.pi
        else:
            if diff > 0:
                diff -= 2 * math.pi
            diff = -diff
        return diff
    
    regions = sorted(regions, key=angle_diff)
    
    current_pos = spiral_end
    
    for region in regions:
        if region.is_empty or region.area < min_radius * min_radius:
            continue
        
        # Find entry point on this region closest to current position
        # Entry should be on the edge closest to the spiral
        entry_pt = _find_arm_entry_point(region, spiral_cx, spiral_cy, spiral_radius)
        
        # Create smooth arc transition from current position to entry point
        if current_pos and entry_pt:
            arc_pts = _create_spiral_to_arm_arc(
                current_pos, entry_pt, spiral_cx, spiral_cy, climb_milling
            )
            all_points.extend(arc_pts)
        
        # Generate trochoidal path into this region
        region_pts = _clear_arm_region(
            region, original_pocket, tool_radius, min_radius, stepover, climb_milling
        )
        
        all_points.extend(region_pts)
        
        # Update current position
        if region_pts:
            current_pos = region_pts[-1]
    
    return all_points


def _find_arm_entry_point(region: Polygon, cx: float, cy: float, spiral_radius: float) -> tuple:
    """Find the best entry point into an arm region from the spiral edge."""
    import math
    
    # Get the centroid direction
    centroid = region.centroid
    angle = math.atan2(centroid.y - cy, centroid.x - cx)
    
    # Entry point is on the spiral edge in the direction of this arm
    entry_x = cx + spiral_radius * math.cos(angle)
    entry_y = cy + spiral_radius * math.sin(angle)
    
    return (entry_x, entry_y)


def _create_spiral_to_arm_arc(
    start: tuple, 
    end: tuple, 
    cx: float, 
    cy: float,
    climb_milling: bool,
) -> list[tuple[float, float]]:
    """Create smooth arc transition from spiral end to arm entry."""
    import math
    
    # Arc along the spiral edge
    start_angle = math.atan2(start[1] - cy, start[0] - cx)
    end_angle = math.atan2(end[1] - cy, end[0] - cx)
    
    # Calculate radius (average of start and end distances)
    r1 = math.sqrt((start[0] - cx)**2 + (start[1] - cy)**2)
    r2 = math.sqrt((end[0] - cx)**2 + (end[1] - cy)**2)
    radius = (r1 + r2) / 2
    
    # Determine arc direction
    angle_diff = end_angle - start_angle
    
    if climb_milling:
        # Continue in positive direction
        if angle_diff < 0:
            angle_diff += 2 * math.pi
    else:
        # Continue in negative direction
        if angle_diff > 0:
            angle_diff -= 2 * math.pi
    
    # Generate arc points
    points = []
    num_pts = max(4, int(abs(angle_diff) * radius / 2))
    
    for i in range(1, num_pts + 1):
        t = i / num_pts
        angle = start_angle + angle_diff * t
        x = cx + radius * math.cos(angle)
        y = cy + radius * math.sin(angle)
        points.append((x, y))
    
    return points


def _clear_arm_region(
    region: Polygon,
    original_pocket: Polygon | MultiPolygon,
    tool_radius: float,
    min_radius: float,
    stepover: float,
    climb_milling: bool,
) -> list[tuple[float, float]]:
    """Clear a single arm/region with adaptive trochoidal arcs."""
    import math
    
    points = []
    pts_per_circle = 16
    
    # Use offset contours within this region as paths
    # This ensures we follow the arm shape
    current = region
    paths = []
    
    while not current.is_empty:
        if isinstance(current, Polygon):
            paths.append(list(current.exterior.coords))
        elif isinstance(current, MultiPolygon):
            for p in current.geoms:
                if not p.is_empty:
                    paths.append(list(p.exterior.coords))
        
        current = current.buffer(-stepover * 0.8, join_style=1, resolution=4)
        
        if len(paths) > 20:
            break
    
    # Reverse for center-out
    paths = list(reversed(paths))
    
    for path in paths:
        if len(path) < 3:
            continue
        
        # Walk along path generating adaptive circles
        path_len = 0.0
        segment_starts = [0.0]
        
        for i in range(1, len(path)):
            dx = path[i][0] - path[i-1][0]
            dy = path[i][1] - path[i-1][1]
            path_len += math.sqrt(dx*dx + dy*dy)
            segment_starts.append(path_len)
        
        if path_len < stepover:
            continue
        
        dist = 0.0
        step = stepover * 1.0
        
        while dist <= path_len:
            cx, cy = _point_at_distance(path, segment_starts, dist)
            
            # Adaptive radius
            radius = _distance_to_boundary(cx, cy, original_pocket) - tool_radius * 0.2
            radius = max(radius, min_radius)
            radius = min(radius, stepover * 4)
            
            if radius >= min_radius:
                circle_pts = _generate_circle(cx, cy, radius, pts_per_circle, climb_milling)
                points.extend(circle_pts)
            
            # Adaptive step
            dist += max(step * 0.5, radius * 0.7)
    
    return points


def _distance_to_boundary(x: float, y: float, pocket) -> float:
    """Get distance from point to pocket boundary."""
    pt = Point(x, y)
    if isinstance(pocket, Polygon):
        return pocket.exterior.distance(pt)
    else:
        return min(p.exterior.distance(pt) for p in pocket.geoms)


def _trochoidal_adaptive_path(
    path: list[tuple[float, float]],
    pocket: Polygon | MultiPolygon,
    tool_radius: float,
    min_radius: float,
    step: float,
    stepover: float,
    climb_milling: bool,
) -> list[tuple[float, float]]:
    """Generate trochoidal circles with adaptive radius along a path.
    
    Circle radius at each point = distance to pocket boundary.
    Fills the full width of the pocket at each position.
    """
    import math
    
    if len(path) < 2:
        return []
    
    points = []
    pts_per_circle = 16
    
    # Calculate path length and segment starts
    path_len = 0.0
    segment_starts = [0.0]
    
    for i in range(1, len(path)):
        dx = path[i][0] - path[i-1][0]
        dy = path[i][1] - path[i-1][1]
        path_len += math.sqrt(dx*dx + dy*dy)
        segment_starts.append(path_len)
    
    if path_len < step:
        return []
    
    # Walk along path, generating adaptive circles
    dist = 0.0
    
    while dist <= path_len:
        # Get position on path
        cx, cy = _point_at_distance(path, segment_starts, dist)
        
        # Calculate max radius at this point (distance to boundary)
        pt = Point(cx, cy)
        if isinstance(pocket, Polygon):
            boundary_dist = pocket.exterior.distance(pt)
        else:
            # For MultiPolygon, find closest boundary
            boundary_dist = float('inf')
            for p in pocket.geoms:
                d = p.exterior.distance(pt)
                boundary_dist = min(boundary_dist, d)
        
        # Use boundary distance as radius - fill the available width
        radius = boundary_dist - tool_radius * 0.1  # Smaller margin = larger circles
        radius = max(radius, min_radius)
        # Cap at reasonable max to prevent giant circles
        radius = min(radius, stepover * 4)
        
        # Generate circle if radius is useful
        if radius >= min_radius:
            circle_pts = _generate_circle(cx, cy, radius, pts_per_circle, climb_milling)
            points.extend(circle_pts)
        
        # Adaptive step - overlap circles by ~50%
        adaptive_step = max(step * 0.5, radius * 0.8)
        dist += adaptive_step
    
    return points


def _trochoidal_along_path(
    path: list[tuple[float, float]],
    radius: float,
    step: float,
    climb_milling: bool,
) -> list[tuple[float, float]]:
    """Generate trochoidal circles along a path.
    
    Each circle overlaps the previous, advancing along the path.
    """
    import math
    
    if len(path) < 2:
        return []
    
    points = []
    pts_per_circle = 12  # Points per circular motion
    
    # Walk along path, placing circle centers at step intervals
    path_len = 0.0
    segment_starts = [0.0]
    
    for i in range(1, len(path)):
        dx = path[i][0] - path[i-1][0]
        dy = path[i][1] - path[i-1][1]
        path_len += math.sqrt(dx*dx + dy*dy)
        segment_starts.append(path_len)
    
    if path_len < step:
        # Path too short, just one circle at center
        mx = sum(p[0] for p in path) / len(path)
        my = sum(p[1] for p in path) / len(path)
        return _generate_circle(mx, my, radius, pts_per_circle, climb_milling)
    
    # Generate circles at step intervals along path
    dist = 0.0
    while dist <= path_len:
        # Find position on path at this distance
        cx, cy = _point_at_distance(path, segment_starts, dist)
        
        # Generate circle at this position
        circle_pts = _generate_circle(cx, cy, radius, pts_per_circle, climb_milling)
        points.extend(circle_pts)
        
        dist += step
    
    return points


def _point_at_distance(
    path: list[tuple[float, float]],
    segment_starts: list[float],
    dist: float,
) -> tuple[float, float]:
    """Get point on path at given distance from start."""
    import math
    
    for i in range(1, len(path)):
        if segment_starts[i] >= dist:
            # Interpolate within this segment
            seg_start = segment_starts[i-1]
            seg_len = segment_starts[i] - seg_start
            
            if seg_len < 0.001:
                return path[i-1]
            
            t = (dist - seg_start) / seg_len
            x = path[i-1][0] + t * (path[i][0] - path[i-1][0])
            y = path[i-1][1] + t * (path[i][1] - path[i-1][1])
            return (x, y)
    
    return path[-1]


def _generate_circle(
    cx: float, cy: float, 
    radius: float, 
    num_pts: int,
    climb_milling: bool,
) -> list[tuple[float, float]]:
    """Generate points for a circle."""
    import math
    
    points = []
    direction = 1 if climb_milling else -1
    
    for i in range(num_pts + 1):
        angle = direction * 2 * math.pi * i / num_pts
        x = cx + radius * math.cos(angle)
        y = cy + radius * math.sin(angle)
        points.append((x, y))
    
    return points


def _extract_polygon_coords(geom, climb_milling: bool) -> list[tuple[float, float]]:
    """Extract coordinates from polygon geometry."""
    coords = []
    
    if isinstance(geom, Polygon):
        ring = list(geom.exterior.coords)
        if not climb_milling:
            ring = ring[::-1]
        coords.extend(ring)
    elif isinstance(geom, MultiPolygon):
        for poly in geom.geoms:
            ring = list(poly.exterior.coords)
            if not climb_milling:
                ring = ring[::-1]
            coords.extend(ring)
    
    return coords


def _reorder_coords_nearest(coords: list, ref: tuple) -> list:
    """Reorder closed ring to start from point nearest to reference."""
    if len(coords) < 2:
        return coords
    
    # Check if closed
    is_closed = coords[0] == coords[-1]
    if is_closed:
        coords = coords[:-1]
    
    if not coords:
        return []
    
    # Find nearest
    min_d = float('inf')
    min_i = 0
    rx, ry = ref
    
    for i, (x, y) in enumerate(coords):
        d = (x-rx)**2 + (y-ry)**2
        if d < min_d:
            min_d = d
            min_i = i
    
    result = coords[min_i:] + coords[:min_i]
    if is_closed and result:
        result.append(result[0])
    
    return result


def _smooth_arc_transition(start, end, next_pt, climb_milling) -> list:
    """Create smooth arc from start to end flowing toward next_pt."""
    import math
    
    sx, sy = start
    ex, ey = end
    nx, ny = next_pt
    
    dx = ex - sx
    dy = ey - sy
    dist = math.sqrt(dx*dx + dy*dy)
    
    if dist < 0.5:
        return []
    
    # Direction toward next point
    cdx = nx - ex
    cdy = ny - ey
    clen = math.sqrt(cdx*cdx + cdy*cdy)
    
    if clen < 0.001:
        return []
    
    # Perpendicular for arc bulge
    px = -dy / dist
    py = dx / dist
    
    # Bulge in direction of cut
    dot = px * cdx / clen + py * cdy / clen
    bulge = dist * 0.25 * (1 if dot > 0 else -1)
    if not climb_milling:
        bulge = -bulge
    
    mx = (sx + ex) / 2 + px * bulge
    my = (sy + ey) / 2 + py * bulge
    
    # Bezier arc
    pts = []
    n = max(3, int(dist / 2))
    for i in range(1, n):
        t = i / n
        t1 = 1 - t
        x = t1*t1*sx + 2*t1*t*mx + t*t*ex
        y = t1*t1*sy + 2*t1*t*my + t*t*ey
        pts.append((x, y))
    
    return pts


def _fill_region_with_spirals(
    region,
    arc_step: float,
    tool_radius: float,
    climb_milling: bool,
    all_points: list,
    depth: int = 0,
):
    """Recursively fill a region with circular spirals.
    
    Generates a spiral from the center, then finds remaining
    uncovered areas and fills them with smaller spirals.
    """
    import math
    
    if region.is_empty or depth > 10:
        return
    
    # Handle MultiPolygon by processing each part
    if isinstance(region, MultiPolygon):
        for poly in region.geoms:
            if not poly.is_empty and poly.area > tool_radius * tool_radius:
                _fill_region_with_spirals(
                    poly, arc_step, tool_radius, climb_milling, all_points, depth
                )
        return
    
    # Skip tiny regions
    if region.area < tool_radius * tool_radius * 2:
        return
    
    # Find center point (use centroid, but ensure it's inside)
    centroid = region.centroid
    if not region.contains(centroid):
        # Use representative point if centroid is outside
        centroid = region.representative_point()
    
    cx, cy = centroid.x, centroid.y
    
    # Find max radius that fits in this region from the center
    max_r = _max_radius_in_region(region, cx, cy, arc_step)
    
    if max_r < arc_step:
        return
    
    # Generate spiral for this region
    spiral_points = _generate_spiral(cx, cy, max_r, arc_step, climb_milling)
    
    if spiral_points:
        all_points.extend(spiral_points)
        
        # Calculate area covered by spiral
        covered = Point(cx, cy).buffer(max_r + tool_radius)
        
        # Find remaining uncovered area
        remaining = region.difference(covered)
        
        if not remaining.is_empty and remaining.area > tool_radius * tool_radius:
            # Recursively fill remaining areas with smaller spirals
            _fill_region_with_spirals(
                remaining, arc_step, tool_radius, climb_milling, all_points, depth + 1
            )


def _max_radius_in_region(region, cx: float, cy: float, step: float) -> float:
    """Find the maximum spiral radius that fits in region from center point."""
    import math
    
    # Check distances at various angles
    max_safe_r = 0.0
    center = Point(cx, cy)
    
    for angle_deg in range(0, 360, 15):
        angle = math.radians(angle_deg)
        
        # Binary search for max radius in this direction
        low, high = 0.0, 500.0
        
        for _ in range(10):  # Binary search iterations
            mid = (low + high) / 2
            test_pt = Point(cx + mid * math.cos(angle), cy + mid * math.sin(angle))
            
            if region.contains(test_pt):
                low = mid
            else:
                high = mid
        
        # Use the minimum across all directions for circular spiral
        if max_safe_r == 0.0:
            max_safe_r = low
        else:
            max_safe_r = min(max_safe_r, low)
    
    return max_safe_r


def _generate_spiral(
    cx: float, cy: float, 
    max_radius: float, 
    arc_step: float,
    climb_milling: bool,
) -> list[tuple[float, float]]:
    """Generate an Archimedean spiral from center outward."""
    import math
    
    if max_radius < arc_step:
        return []
    
    points = []
    points_per_rev = 32
    
    num_revolutions = int(max_radius / arc_step) + 1
    total_points = num_revolutions * points_per_rev
    
    for i in range(total_points):
        r = arc_step * (i / points_per_rev)
        
        if r > max_radius:
            break
        
        if climb_milling:
            a = 2 * math.pi * (i / points_per_rev)
        else:
            a = -2 * math.pi * (i / points_per_rev)
        
        px = cx + r * math.cos(a)
        py = cy + r * math.sin(a)
        points.append((px, py))
    
    return points


def _extract_line_points(geom) -> list[tuple[float, float]]:
    """Extract points from a line geometry (handles MultiLineString)."""
    points = []
    
    if geom.is_empty:
        return points
    
    if isinstance(geom, LineString):
        points.extend(list(geom.coords))
    elif isinstance(geom, MultiLineString):
        # Connect segments with the closest points
        segments = list(geom.geoms)
        if not segments:
            return points
        
        # Start with first segment
        points.extend(list(segments[0].coords))
        
        # Add remaining segments, connecting them
        for seg in segments[1:]:
            seg_coords = list(seg.coords)
            if seg_coords:
                points.extend(seg_coords)
    
    return points


def _get_max_radius(polygon, cx: float, cy: float) -> float:
    """Get maximum distance from center to polygon boundary."""
    import math
    
    if isinstance(polygon, Polygon):
        coords = list(polygon.exterior.coords)
    elif isinstance(polygon, MultiPolygon):
        coords = []
        for p in polygon.geoms:
            coords.extend(list(p.exterior.coords))
    else:
        return 100.0
    
    max_r = 0.0
    for x, y in coords:
        r = math.sqrt((x - cx)**2 + (y - cy)**2)
        max_r = max(max_r, r)
    
    return max_r


def _morph_contours_to_spiral(
    offset_levels: list,
    climb_milling: bool,
) -> list[tuple[float, float]]:
    """Create a smooth morphing spiral from offset contours.
    
    Uses arc lead-ins between contours to maintain constant
    tool engagement during transitions.
    """
    import math
    
    if not offset_levels:
        return []
    
    all_points = []
    
    for level_idx, level_geom in enumerate(offset_levels):
        if level_geom.is_empty:
            continue
        
        # Get exterior ring(s)
        if isinstance(level_geom, Polygon):
            rings = [level_geom.exterior]
        elif isinstance(level_geom, MultiPolygon):
            rings = [p.exterior for p in level_geom.geoms if not p.is_empty]
        else:
            continue
        
        for ring in rings:
            coords = list(ring.coords)
            if len(coords) < 3:
                continue
            
            # Ensure correct direction for climb/conventional
            if not climb_milling:
                coords = coords[::-1]
            
            # If we have previous points, find best connection point
            if all_points:
                last_pt = all_points[-1]
                coords = _reorder_to_nearest(coords, last_pt)
            
            # Add arc lead-in from last point to next contour
            if all_points and len(coords) > 1:
                lead_in = _create_arc_lead_in(
                    all_points[-1], 
                    coords[0], 
                    coords[1],  # Next point gives us direction
                    climb_milling
                )
                all_points.extend(lead_in)
            
            # Add all points from this contour
            all_points.extend(coords)
    
    return all_points


def _create_arc_lead_in(
    start: tuple[float, float],
    end: tuple[float, float],
    next_pt: tuple[float, float],
    climb_milling: bool,
) -> list[tuple[float, float]]:
    """Create an arc lead-in from start to end that flows into the cut direction.
    
    The arc curves in the direction of the upcoming cut to maintain
    engagement as the tool transitions to the next contour.
    """
    import math
    
    sx, sy = start
    ex, ey = end
    nx, ny = next_pt
    
    # Vector from end to next point (cut direction)
    cut_dx = nx - ex
    cut_dy = ny - ey
    cut_len = math.sqrt(cut_dx*cut_dx + cut_dy*cut_dy)
    
    if cut_len < 0.001:
        return []
    
    # Normalize cut direction
    cut_dx /= cut_len
    cut_dy /= cut_len
    
    # Vector from start to end
    dx = ex - sx
    dy = ey - sy
    dist = math.sqrt(dx*dx + dy*dy)
    
    if dist < 0.1:
        return []
    
    # Perpendicular to start-end line (for arc bulge)
    perp_x = -dy / dist
    perp_y = dx / dist
    
    # Arc bulges in direction that flows into the cut
    # Use dot product to determine which side
    dot = perp_x * cut_dx + perp_y * cut_dy
    bulge_dir = 1 if dot > 0 else -1
    if not climb_milling:
        bulge_dir *= -1
    
    # Arc height proportional to distance (but capped)
    arc_height = min(dist * 0.3, 2.0)
    
    # Arc midpoint
    mid_x = (sx + ex) / 2 + perp_x * arc_height * bulge_dir
    mid_y = (sy + ey) / 2 + perp_y * arc_height * bulge_dir
    
    # Generate arc points using quadratic bezier
    points = []
    num_pts = max(4, int(dist / 1.0))  # At least 4 points, more for longer arcs
    
    for i in range(1, num_pts):
        t = i / num_pts
        t1 = 1 - t
        # Quadratic bezier
        px = t1*t1*sx + 2*t1*t*mid_x + t*t*ex
        py = t1*t1*sy + 2*t1*t*mid_y + t*t*ey
        points.append((px, py))
    
    return points


def _reorder_to_nearest(
    coords: list[tuple[float, float]],
    reference: tuple[float, float],
) -> list[tuple[float, float]]:
    """Reorder closed ring to start from point nearest to reference."""
    if len(coords) < 2:
        return coords
    
    # Check if closed
    is_closed = coords[0] == coords[-1]
    if is_closed:
        coords = coords[:-1]
    
    if not coords:
        return []
    
    # Find nearest point
    ref_x, ref_y = reference
    min_dist = float('inf')
    min_idx = 0
    
    for i, (x, y) in enumerate(coords):
        d = (x - ref_x)**2 + (y - ref_y)**2
        if d < min_dist:
            min_dist = d
            min_idx = i
    
    # Reorder
    result = coords[min_idx:] + coords[:min_idx]
    
    # Re-close if was closed
    if is_closed and result:
        result.append(result[0])
    
    return result


def _generate_contour_parallel(
    polygon: Polygon | MultiPolygon,
    stepover: float,
    depth_passes: list[float],
    climb_milling: bool,
) -> list[Toolpath]:
    """Generate contour-parallel adaptive clearing toolpaths.
    
    Creates successive inward offsets following the pocket shape,
    maintaining consistent tool engagement.
    
    Args:
        polygon: The pocket geometry (already offset by tool radius)
        stepover: Distance between adjacent contours
        depth_passes: List of Z depths for each level
        climb_milling: True for climb, False for conventional
    
    Returns:
        List of toolpaths
    """
    toolpaths = []
    
    if polygon.is_empty:
        return []
    
    # Generate contours for each depth level
    for z_depth in depth_passes:
        level_toolpaths = _generate_contour_level(
            polygon, stepover, z_depth, climb_milling
        )
        toolpaths.extend(level_toolpaths)
    
    return toolpaths


def _generate_contour_level(
    polygon: Polygon | MultiPolygon,
    stepover: float,
    z_depth: float,
    climb_milling: bool,
) -> list[Toolpath]:
    """Generate contour toolpaths for a single depth level.
    
    Uses morphological smoothing to round sharp corners,
    then generates offset contours that flow smoothly.
    """
    # Heavy morphological smoothing - round ALL corners
    smooth_amount = stepover * 2.0
    
    smoothed = polygon.buffer(-smooth_amount, join_style=1, resolution=16)
    if not smoothed.is_empty:
        smoothed = smoothed.buffer(smooth_amount, join_style=1, resolution=16)
    else:
        smoothed = polygon
    
    if smoothed.is_empty:
        return []
    
    # Generate offset contours from the smoothed shape
    effective_stepover = stepover * 0.8
    contours = []
    current = smoothed
    
    while not current.is_empty:
        contours.append(current)
        current = current.buffer(-effective_stepover, join_style=1, resolution=16)
        if len(contours) > 200:
            break
    
    if not contours:
        return []
    
    # Reverse to go center-out
    contours = list(reversed(contours))
    
    # Stitch contours into one continuous spiral
    all_points = []
    
    for contour in contours:
        if isinstance(contour, Polygon):
            coords = list(contour.exterior.coords)
        elif isinstance(contour, MultiPolygon):
            coords = []
            for p in contour.geoms:
                coords.extend(list(p.exterior.coords))
        else:
            continue
        
        if not coords:
            continue
        
        if not climb_milling:
            coords = coords[::-1]
        
        if all_points:
            coords = _reorder_ring_nearest(coords, all_points[-1])
        
        all_points.extend(coords)
    
    toolpaths = []
    if len(all_points) >= 2:
        toolpaths.append(Toolpath(
            points=np.array(all_points),
            z_depth=z_depth,
            is_rapid=False,
        ))
    
    return toolpaths


def _build_spiral_center_out(
    offset_levels: list,
    stepover: float,
    climb_milling: bool,
) -> list[tuple[float, float]]:
    """Build a continuous spiral from center outward.
    
    Each contour completes a full loop, then smoothly steps outward
    to the next contour, maintaining constant tool engagement.
    """
    all_points = []
    
    for i, level_geom in enumerate(offset_levels):
        if level_geom.is_empty:
            continue
        
        # Get rings at this level
        rings = _extract_rings_for_spiral(level_geom, climb_milling)
        
        for ring in rings:
            if len(ring) < 3:
                continue
            
            # Ensure ring is closed (ends at start point)
            is_closed = (len(ring) > 1 and 
                        ring[0][0] == ring[-1][0] and 
                        ring[0][1] == ring[-1][1])
            
            if not is_closed:
                ring = list(ring) + [ring[0]]
            
            # If we have previous points, align this ring's start 
            # with where we left off and add smooth transition
            if all_points:
                last_point = all_points[-1]
                ring = _reorder_ring_to_align(ring, last_point, stepover)
                
                # The transition arc from last point to ring start
                # This arc steps outward while maintaining engagement
                transition = _create_stepout_arc(
                    last_point, ring[0], stepover, climb_milling
                )
                all_points.extend(transition)
            
            # Add all points of this ring (complete the loop)
            all_points.extend(ring)
    
    return all_points


def _reorder_ring_to_align(
    ring: list[tuple[float, float]],
    last_point: tuple[float, float],
    stepover: float,
) -> list[tuple[float, float]]:
    """Reorder ring to start from point that creates smooth outward transition.
    
    Finds the point on the ring that is approximately 'stepover' distance
    away from last_point, enabling a smooth stepping transition.
    """
    if len(ring) < 2:
        return ring
    
    # Remove closing point for processing
    is_closed = (ring[0][0] == ring[-1][0] and ring[0][1] == ring[-1][1])
    if is_closed:
        ring = ring[:-1]
    
    if len(ring) < 2:
        return ring + [ring[0]] if is_closed else ring
    
    # Find point closest to ideal stepover distance from last_point
    target_dist = stepover
    best_idx = 0
    best_score = float('inf')
    
    lx, ly = last_point
    
    for i, (x, y) in enumerate(ring):
        dist = ((x - lx) ** 2 + (y - ly) ** 2) ** 0.5
        # Score: how close to ideal stepover distance
        score = abs(dist - target_dist)
        if score < best_score:
            best_score = score
            best_idx = i
    
    # Reorder ring to start from best point
    reordered = ring[best_idx:] + ring[:best_idx]
    
    # Re-close the ring
    if is_closed:
        reordered = reordered + [reordered[0]]
    
    return reordered


def _create_stepout_arc(
    start: tuple[float, float],
    end: tuple[float, float],
    stepover: float,
    climb_milling: bool,
) -> list[tuple[float, float]]:
    """Create a smooth arc that steps outward from one contour to the next.
    
    This arc maintains tool engagement during the transition between loops.
    """
    import math
    
    sx, sy = start
    ex, ey = end
    
    dx = ex - sx
    dy = ey - sy
    dist = math.sqrt(dx * dx + dy * dy)
    
    # If very close, no transition needed
    if dist < 0.1:
        return []
    
    # For proper stepping, we want a smooth arc
    # The arc bulges in the direction of cut (climb vs conventional)
    
    mid_x = (sx + ex) / 2
    mid_y = (sy + ey) / 2
    
    # Perpendicular direction for arc bulge
    if dist > 0.001:
        perp_x = -dy / dist
        perp_y = dx / dist
    else:
        perp_x, perp_y = 0, 0
    
    # Bulge outward (away from center) for smooth engagement
    # Bulge amount proportional to step distance
    bulge = min(dist * 0.3, stepover * 0.5)
    if not climb_milling:
        bulge = -bulge
    
    arc_x = mid_x + perp_x * bulge
    arc_y = mid_y + perp_y * bulge
    
    # Generate arc points using quadratic bezier
    points = []
    num_pts = max(3, int(dist / (stepover * 0.25)))
    
    for i in range(1, num_pts):
        t = i / num_pts
        t1 = 1 - t
        px = t1 * t1 * sx + 2 * t1 * t * arc_x + t * t * ex
        py = t1 * t1 * sy + 2 * t1 * t * arc_y + t * t * ey
        points.append((px, py))
    
    return points


def _build_spiral_for_region(
    region: Polygon,
    offset_levels: list,
    stepover: float,
    climb_milling: bool,
) -> list[tuple[float, float]]:
    """Build a continuous spiral path for a single polygon region.
    
    Connects successive contours with smooth transitions to maintain
    constant tool engagement.
    """
    if region.is_empty:
        return []
    
    all_points = []
    last_point = None
    
    for level_geom in offset_levels:
        # Get the portion of this offset level within our region
        # (handles multi-polygon cases where regions separate)
        level_in_region = level_geom.intersection(region.buffer(stepover * 0.1))
        
        if level_in_region.is_empty:
            continue
        
        # Extract exterior ring for this level
        rings = _extract_rings_for_spiral(level_in_region, climb_milling)
        
        for ring in rings:
            if len(ring) < 3:
                continue
            
            # Reorder ring to start from nearest point to last position
            if last_point is not None:
                ring = _reorder_ring_from_nearest(ring, last_point)
                
                # Add smooth arc transition from last point to ring start
                transition = _create_arc_transition(
                    last_point, ring[0], stepover, climb_milling
                )
                all_points.extend(transition)
            
            # Add ring points (excluding the closing duplicate if present)
            for pt in ring[:-1] if ring[0] == ring[-1] else ring:
                all_points.append(pt)
            
            last_point = all_points[-1] if all_points else None
    
    return all_points


def _extract_rings_for_spiral(
    geom: Polygon | MultiPolygon,
    climb_milling: bool,
) -> list[list[tuple[float, float]]]:
    """Extract rings optimized for spiral connection."""
    rings = []
    
    if isinstance(geom, Polygon):
        polygons = [geom]
    elif isinstance(geom, MultiPolygon):
        polygons = list(geom.geoms)
    else:
        return []
    
    for poly in polygons:
        if poly.is_empty or poly.exterior is None:
            continue
        
        coords = list(poly.exterior.coords)
        if not climb_milling:
            coords = coords[::-1]
        rings.append(coords)
    
    return rings


def _create_arc_transition(
    start: tuple[float, float],
    end: tuple[float, float],
    stepover: float,
    climb_milling: bool,
) -> list[tuple[float, float]]:
    """Create a smooth arc transition between two points.
    
    This maintains engagement during the transition from one contour
    to the next, rather than making an abrupt direction change.
    """
    import math
    
    sx, sy = start
    ex, ey = end
    
    dx = ex - sx
    dy = ey - sy
    dist = math.sqrt(dx*dx + dy*dy)
    
    # If points are very close, just return direct connection
    if dist < stepover * 0.5:
        return []
    
    # If points are far, use linear blend (arc would be too large)
    if dist > stepover * 3:
        # Create intermediate points for smoother transition
        num_pts = max(2, int(dist / (stepover * 0.5)))
        points = []
        for i in range(1, num_pts):
            t = i / num_pts
            points.append((sx + dx * t, sy + dy * t))
        return points
    
    # Create arc transition
    # Calculate arc center offset perpendicular to the line
    mid_x = (sx + ex) / 2
    mid_y = (sy + ey) / 2
    
    # Perpendicular direction
    perp_x = -dy / dist
    perp_y = dx / dist
    
    # Arc bulge (positive for climb, negative for conventional)
    bulge = stepover * 0.3 * (1 if climb_milling else -1)
    
    # Arc midpoint
    arc_x = mid_x + perp_x * bulge
    arc_y = mid_y + perp_y * bulge
    
    # Generate arc points
    points = []
    num_arc_pts = 5
    for i in range(1, num_arc_pts):
        t = i / num_arc_pts
        # Quadratic bezier-like interpolation through arc point
        t1 = 1 - t
        px = t1*t1*sx + 2*t1*t*arc_x + t*t*ex
        py = t1*t1*sy + 2*t1*t*arc_y + t*t*ey
        points.append((px, py))
    
    return points


def _extract_rings(
    geom: Polygon | MultiPolygon,
    climb_milling: bool,
) -> list[list[tuple[float, float]]]:
    """Extract all rings from a geometry as coordinate lists.
    
    Returns exterior and interior rings with appropriate direction
    for climb or conventional milling.
    """
    rings = []
    
    if isinstance(geom, Polygon):
        polygons = [geom]
    elif isinstance(geom, MultiPolygon):
        polygons = list(geom.geoms)
    else:
        return []
    
    for poly in polygons:
        if poly.is_empty:
            continue
        
        # Exterior ring
        coords = list(poly.exterior.coords)
        # Climb milling = CCW for exterior (cutting on right side of tool)
        # Conventional = CW for exterior
        if not climb_milling:
            coords = coords[::-1]
        rings.append(coords)
        
        # Interior rings (holes) - opposite direction
        for interior in poly.interiors:
            coords = list(interior.coords)
            if climb_milling:
                coords = coords[::-1]
            rings.append(coords)
    
    return rings


def _reorder_ring_from_nearest(
    coords: list[tuple[float, float]],
    reference_point: tuple[float, float],
) -> list[tuple[float, float]]:
    """Reorder a closed ring to start from the point nearest to reference.
    
    This minimizes rapid travel between contours.
    """
    if len(coords) < 2:
        return coords
    
    # Check if ring is closed (first == last)
    is_closed = (coords[0][0] == coords[-1][0] and coords[0][1] == coords[-1][1])
    
    if is_closed:
        # Remove duplicate closing point for processing
        coords = coords[:-1]
    
    if len(coords) < 2:
        return coords
    
    # Find nearest point
    min_dist = float('inf')
    min_idx = 0
    ref_x, ref_y = reference_point
    
    for i, (x, y) in enumerate(coords):
        dist = (x - ref_x) ** 2 + (y - ref_y) ** 2
        if dist < min_dist:
            min_dist = dist
            min_idx = i
    
    # Reorder: start from nearest point
    reordered = coords[min_idx:] + coords[:min_idx]
    
    # Close the ring
    if is_closed:
        reordered.append(reordered[0])
    
    return reordered


def _generate_raster_pattern(
    polygon: Polygon | MultiPolygon,
    stepover: float,
    z_depth: float,
    climb_milling: bool,
) -> list[Toolpath]:
    """Generate raster (parallel line) pattern for pocket clearing.
    
    Creates horizontal lines across the pocket, alternating direction
    for efficient cutting.
    """
    toolpaths = []
    
    if polygon.is_empty:
        return []
    
    # Get bounds
    minx, miny, maxx, maxy = polygon.bounds
    
    # Generate horizontal lines
    y = miny + stepover / 2
    line_num = 0
    
    while y < maxy:
        # Create a horizontal line across the pocket
        line = LineString([(minx - 1, y), (maxx + 1, y)])
        
        # Intersect with pocket
        intersection = polygon.intersection(line)
        
        if not intersection.is_empty:
            segments = _extract_line_segments(intersection)
            
            # Alternate direction for each line
            if line_num % 2 == 1:
                segments = [seg[::-1] for seg in reversed(segments)]
            
            # Reverse for climb milling if needed
            if not climb_milling:
                segments = [seg[::-1] for seg in segments]
            
            for seg in segments:
                if len(seg) >= 2:
                    points = np.array(seg)
                    toolpaths.append(Toolpath(
                        points=points,
                        z_depth=z_depth,
                        is_rapid=False,
                    ))
        
        y += stepover
        line_num += 1
    
    # Add perimeter pass for cleaner walls
    perimeter_paths = _generate_perimeter_pass(polygon, z_depth, climb_milling)
    toolpaths.extend(perimeter_paths)
    
    return toolpaths


def _extract_line_segments(geom) -> list[list[tuple[float, float]]]:
    """Extract line segments from a geometry."""
    segments = []
    
    if isinstance(geom, LineString):
        segments.append(list(geom.coords))
    elif isinstance(geom, MultiLineString):
        for line in geom.geoms:
            segments.append(list(line.coords))
    elif hasattr(geom, 'geoms'):
        for g in geom.geoms:
            segments.extend(_extract_line_segments(g))
    
    return segments


def _generate_perimeter_pass(
    polygon: Polygon | MultiPolygon,
    z_depth: float,
    climb_milling: bool,
) -> list[Toolpath]:
    """Generate a perimeter cleanup pass."""
    toolpaths = []
    
    def process_ring(ring, reverse: bool):
        coords = list(ring.coords)
        if reverse:
            coords = coords[::-1]
        if len(coords) >= 2:
            return Toolpath(
                points=np.array(coords),
                z_depth=z_depth,
                is_rapid=False,
            )
        return None
    
    if isinstance(polygon, Polygon):
        polygons = [polygon]
    elif isinstance(polygon, MultiPolygon):
        polygons = list(polygon.geoms)
    else:
        return []
    
    for poly in polygons:
        # Exterior ring - climb = CCW, conventional = CW
        exterior = poly.exterior
        tp = process_ring(exterior, not climb_milling)
        if tp:
            toolpaths.append(tp)
        
        # Interior rings (holes) - opposite direction
        for interior in poly.interiors:
            tp = process_ring(interior, climb_milling)
            if tp:
                toolpaths.append(tp)
    
    return toolpaths


@dataclass
class VBitParams:
    """Parameters for v-bit finishing pass."""
    tool: VBit
    flat_depth: float | None = None  # Fixed depth (if None, calculate from target_width)
    target_width: float | None = None  # Target carve width (calculates depth from v-bit angle)
    max_depth: float = 10.0  # Maximum allowed depth
    roughing_diameter: float | None = None  # Roughing end mill diameter for rest machining
    climb_milling: bool = True


def generate_vbit_toolpaths(
    paths: list[CarvePath],
    params: VBitParams,
) -> list[Toolpath]:
    """Generate v-bit toolpaths for finishing pass.
    
    If roughing_diameter is specified, generates paths that clean up
    areas the roughing pass couldn't reach (inside corners).
    
    Args:
        paths: Paths to v-carve (can be open or closed)
        params: V-bit parameters
        
    Returns:
        List of toolpaths for v-carving
    """
    tool = params.tool
    toolpaths = []
    
    # Calculate base cutting depth
    if params.flat_depth is not None:
        base_depth = -abs(params.flat_depth)
    elif params.target_width is not None:
        base_depth = -tool.depth_for_width(params.target_width)
    else:
        # Default to a reasonable depth based on tool diameter
        base_depth = -tool.depth_for_width(tool.diameter / 4)
    
    # Clamp to max depth
    base_depth = max(base_depth, -params.max_depth)
    
    for path in paths:
        if len(path.points) < 2:
            continue
        
        if path.is_closed and params.roughing_diameter is not None:
            # Generate corner cleanup paths for rest machining
            corner_paths = _generate_vbit_corner_cleanup(
                path, tool, params.roughing_diameter, 
                base_depth, params.max_depth, params.climb_milling
            )
            toolpaths.extend(corner_paths)
        else:
            # Simple outline trace for open paths or no roughing info
            # Offset inward so V-bit edge aligns with model boundary
            import math
            from shapely.geometry import Polygon
            
            points = path.points
            
            if path.is_closed and len(points) >= 3:
                # Calculate inward offset: at depth d, edge is at d * tan(angle/2) from center
                half_angle = math.radians(tool.angle / 2)
                offset = abs(base_depth) * math.tan(half_angle)
                
                # Offset polygon inward
                try:
                    polygon = Polygon(points)
                    if not polygon.is_valid:
                        polygon = polygon.buffer(0)
                    
                    offset_poly = polygon.buffer(-offset, join_style=1, resolution=16)
                    
                    if not offset_poly.is_empty and isinstance(offset_poly, Polygon):
                        coords = list(offset_poly.exterior.coords)
                        if not params.climb_milling:
                            coords = coords[::-1]
                        points = np.array(coords)
                    else:
                        # Offset too large, use original
                        if not params.climb_milling:
                            points = points[::-1]
                        points = np.vstack([points, points[0:1]])
                except Exception:
                    if not params.climb_milling:
                        points = points[::-1]
                    points = np.vstack([points, points[0:1]])
            elif path.is_closed:
                if not params.climb_milling:
                    points = points[::-1]
                points = np.vstack([points, points[0:1]])
            
            toolpaths.append(Toolpath(
                points=points,
                z_depth=base_depth,
                is_rapid=False,
            ))
    
    return toolpaths


def _generate_vbit_corner_cleanup(
    path: CarvePath,
    tool: VBit,
    roughing_diameter: float,
    base_depth: float,
    max_depth: float,
    climb_milling: bool,
) -> list[Toolpath]:
    """Generate V-bit paths to clean up corners left by roughing pass.
    
    Pattern based on professional V-carving:
    1. Progressive depth passes with arc-rounded corners
    2. Diagonal ramp-in/out moves at deep corners
    """
    import math
    from shapely.geometry import Polygon, Point
    
    toolpaths = []
    points = path.points
    
    if len(points) < 3:
        return toolpaths
    
    roughing_radius = roughing_diameter / 2
    half_angle = math.radians(tool.angle / 2)
    tan_half = math.tan(half_angle)
    
    # Create polygon from path
    try:
        polygon = Polygon(points)
        if not polygon.is_valid:
            polygon = polygon.buffer(0)
    except Exception:
        return toolpaths
    
    # Calculate where roughing pass cleared
    roughing_cleared = polygon.buffer(-roughing_radius, join_style=1, resolution=16)
    
    # Find corners and their depths
    coords = list(polygon.exterior.coords)[:-1]  # Remove duplicate end
    corner_info = []
    
    for i, (x, y) in enumerate(coords):
        pt = Point(x, y)
        if roughing_cleared.is_empty:
            dist_to_cleared = roughing_radius
        else:
            dist_to_cleared = roughing_cleared.exterior.distance(pt)
        
        # Calculate corner depth
        if dist_to_cleared < 0.01:
            corner_depth = abs(base_depth)
        else:
            extra_width = dist_to_cleared * 2
            extra_depth = tool.depth_for_width(extra_width)
            corner_depth = abs(base_depth) + extra_depth
        
        corner_depth = min(corner_depth, max_depth)
        
        # Calculate corner angle
        prev_idx = (i - 1) % len(coords)
        next_idx = (i + 1) % len(coords)
        
        v1 = (coords[prev_idx][0] - x, coords[prev_idx][1] - y)
        v2 = (coords[next_idx][0] - x, coords[next_idx][1] - y)
        
        len1 = math.sqrt(v1[0]**2 + v1[1]**2)
        len2 = math.sqrt(v2[0]**2 + v2[1]**2)
        
        if len1 > 0.01 and len2 > 0.01:
            v1 = (v1[0]/len1, v1[1]/len1)
            v2 = (v2[0]/len2, v2[1]/len2)
            dot = v1[0]*v2[0] + v1[1]*v2[1]
            corner_angle = math.acos(max(-1, min(1, dot)))
        else:
            corner_angle = math.pi
        
        # Diagonal direction for ramp (bisector pointing outward)
        bisect_x = v1[0] + v2[0]
        bisect_y = v1[1] + v2[1]
        bisect_len = math.sqrt(bisect_x**2 + bisect_y**2)
        if bisect_len > 0.01:
            bisect_x, bisect_y = bisect_x/bisect_len, bisect_y/bisect_len
        else:
            bisect_x, bisect_y = v1[0], v1[1]
        
        corner_info.append({
            'x': x, 'y': y,
            'depth': corner_depth,
            'angle': corner_angle,
            'bisect': (bisect_x, bisect_y),
            'is_sharp': corner_angle < math.pi * 0.7,  # < 126 degrees
        })
    
    # Determine depth passes
    min_corner_depth = min(c['depth'] for c in corner_info)
    max_corner_depth = max(c['depth'] for c in corner_info)
    
    # Create depth levels (0.75mm steps like reference)
    depth_step = 0.75
    depth_levels = []
    d = depth_step
    while d <= max_corner_depth:
        depth_levels.append(d)
        d += depth_step
    if not depth_levels or depth_levels[-1] < max_corner_depth:
        depth_levels.append(max_corner_depth)
    
    # For each depth level, generate perimeter trace with rounded corners
    for level_depth in depth_levels:
        # Calculate offset for this depth
        offset = level_depth * tan_half
        
        # Offset polygon inward
        offset_poly = polygon.buffer(-offset, join_style=1, resolution=16)
        
        if offset_poly.is_empty:
            continue
        
        if isinstance(offset_poly, Polygon):
            offset_coords = list(offset_poly.exterior.coords)
        else:
            continue
        
        if not climb_milling:
            offset_coords = offset_coords[::-1]
        
        # Generate arc-cornered perimeter
        arc_toolpaths = _generate_arc_perimeter(
            offset_coords, level_depth, tan_half, climb_milling
        )
        toolpaths.extend(arc_toolpaths)
    
    # Generate corner ramps at deep corners (final cleanup)
    for corner in corner_info:
        if corner['depth'] > abs(base_depth) * 1.3 and corner['is_sharp']:
            ramp_paths = _create_diagonal_corner_ramp(
                corner['x'], corner['y'],
                corner['depth'], corner['bisect'],
                tan_half, max_depth
            )
            toolpaths.extend(ramp_paths)
    
    return toolpaths


def _generate_arc_perimeter(
    coords: list[tuple[float, float]],
    depth: float,
    tan_half_angle: float,
    climb_milling: bool,
) -> list[Toolpath]:
    """Generate perimeter toolpath with G2/G3 arcs at corners."""
    import math
    
    toolpaths = []
    n = len(coords) - 1  # Exclude duplicate end point
    
    if n < 3:
        return toolpaths
    
    # Arc radius based on depth
    arc_radius = depth * tan_half_angle * 0.8
    
    # Collect points and arcs
    path_segments = []
    
    for i in range(n):
        curr = coords[i]
        next_pt = coords[(i + 1) % n]
        next_next = coords[(i + 2) % n]
        
        # Vector to next point
        dx = next_pt[0] - curr[0]
        dy = next_pt[1] - curr[1]
        edge_len = math.sqrt(dx*dx + dy*dy)
        
        if edge_len < 0.01:
            continue
        
        # Calculate corner angle at next_pt
        dx2 = next_next[0] - next_pt[0]
        dy2 = next_next[1] - next_pt[1]
        edge2_len = math.sqrt(dx2*dx2 + dy2*dy2)
        
        if edge2_len < 0.01:
            continue
        
        # Normalize
        dx, dy = dx/edge_len, dy/edge_len
        dx2, dy2 = dx2/edge2_len, dy2/edge2_len
        
        # Corner angle
        dot = dx*dx2 + dy*dy2
        cross = dx*dy2 - dy*dx2
        
        # For now, just add linear segments
        # (Full arc implementation would be more complex)
        path_segments.append(curr)
    
    # Close path
    if path_segments:
        path_segments.append(path_segments[0])
        toolpaths.append(Toolpath(
            points=np.array(path_segments),
            z_depth=-depth,
            is_rapid=False,
        ))
    
    return toolpaths


def _create_diagonal_corner_ramp(
    corner_x: float,
    corner_y: float, 
    corner_depth: float,
    bisect_dir: tuple[float, float],
    tan_half_angle: float,
    max_depth: float,
) -> list[Toolpath]:
    """Create diagonal ramp-in/out move at a corner.
    
    Pattern from reference:
    - Start at corner at depth
    - Ramp OUT diagonally while rising to surface
    - Ramp back IN while plunging to depth
    """
    toolpaths = []
    
    # Ramp distance based on depth
    ramp_distance = corner_depth / tan_half_angle if tan_half_angle > 0.01 else corner_depth * 2
    ramp_distance = min(ramp_distance, 5.0)  # Cap at 5mm
    
    # Calculate offset for this depth
    offset = corner_depth * tan_half_angle
    
    # Corner position (offset inward)
    cx = corner_x - bisect_dir[0] * offset
    cy = corner_y - bisect_dir[1] * offset
    
    # Ramp-out position (further along bisector)
    rx = corner_x + bisect_dir[0] * ramp_distance
    ry = corner_y + bisect_dir[1] * ramp_distance
    
    # Create ramp path with Z interpolation
    # Points: [corner at depth] -> [ramp-out at surface] -> [corner at depth]
    ramp_points = np.array([
        [cx, cy, -corner_depth],      # Start at corner, deep
        [rx, ry, 0.0],                 # Ramp out to surface
        [cx, cy, -corner_depth],       # Ramp back in to depth
    ])
    
    toolpaths.append(Toolpath(
        points=ramp_points,
        z_depth=-corner_depth,
        is_rapid=False,
    ))
    
    return toolpaths


def _smooth_corners(
    points: list[tuple[float, float]], 
    radius: float
) -> list[tuple[float, float]]:
    """Smooth sharp corners by adding fillet arcs.
    
    Stays on the path, just rounds the corners with small arcs.
    """
    import math
    
    if len(points) < 3:
        return points
    
    result = []
    n = len(points)
    
    for i in range(n):
        p0 = points[(i - 1) % n]
        p1 = points[i]
        p2 = points[(i + 1) % n]
        
        # Calculate vectors
        v1x, v1y = p0[0] - p1[0], p0[1] - p1[1]
        v2x, v2y = p2[0] - p1[0], p2[1] - p1[1]
        
        len1 = math.sqrt(v1x*v1x + v1y*v1y)
        len2 = math.sqrt(v2x*v2x + v2y*v2y)
        
        if len1 < 0.01 or len2 < 0.01:
            result.append(p1)
            continue
        
        # Normalize
        v1x, v1y = v1x/len1, v1y/len1
        v2x, v2y = v2x/len2, v2y/len2
        
        # Calculate angle between edges
        dot = v1x*v2x + v1y*v2y
        angle = math.acos(max(-1, min(1, dot)))
        
        # Skip if angle is very shallow (nearly straight)
        if angle > math.pi * 0.9:
            result.append(p1)
            continue
        
        # Calculate fillet distance (how far from corner to start arc)
        # Limit to half the edge length
        fillet_dist = min(radius, len1 * 0.4, len2 * 0.4)
        
        if fillet_dist < 0.1:
            result.append(p1)
            continue
        
        # Points where fillet starts/ends
        start_x = p1[0] + v1x * fillet_dist
        start_y = p1[1] + v1y * fillet_dist
        end_x = p1[0] + v2x * fillet_dist
        end_y = p1[1] + v2y * fillet_dist
        
        # Add arc points
        result.append((start_x, start_y))
        
        # Interpolate arc (3 intermediate points)
        for t in [0.25, 0.5, 0.75]:
            # Linear interpolation along arc
            ax = start_x + t * (end_x - start_x)
            ay = start_y + t * (end_y - start_y)
            
            # Pull toward corner slightly for smoother curve
            pull = 0.2 * (1 - abs(t - 0.5) * 2)
            ax += pull * (p1[0] - ax)
            ay += pull * (p1[1] - ay)
            
            result.append((ax, ay))
        
        result.append((end_x, end_y))
    
    # Close the path
    if result and result[0] != result[-1]:
        result.append(result[0])
    
    return result


def _create_vertical_ramp(
    corner_pt: tuple[float, float],
    start_depth: float,
    end_depth: float,
) -> list[Toolpath]:
    """Create a vertical ramping plunge into a corner.
    
    The tool moves in Z while staying at the corner XY position.
    """
    toolpaths = []
    
    depth_change = abs(end_depth) - abs(start_depth)
    if depth_change <= 0.1:
        return toolpaths
    
    # Number of plunge steps
    n_steps = max(3, int(depth_change * 2))
    
    # Create ramping plunge at corner position
    for i in range(n_steps):
        t = (i + 1) / n_steps
        z = start_depth - t * depth_change
        
        # Small movement at each depth (tool stays near corner)
        toolpaths.append(Toolpath(
            points=np.array([
                [corner_pt[0], corner_pt[1]],
                [corner_pt[0], corner_pt[1]]  # Same point - vertical move
            ]),
            z_depth=z,
            is_rapid=False,
        ))
    
    return toolpaths


def _group_by_depth(
    points: list[tuple[float, float]], 
    depths: list[float],
    tolerance: float = 0.1,
) -> list[tuple[list[tuple[float, float]], float]]:
    """Group consecutive points with similar depths into segments."""
    if not points:
        return []
    
    segments = []
    current_points = [points[0]]
    current_depth = depths[0]
    
    for i in range(1, len(points)):
        if abs(depths[i] - current_depth) <= tolerance:
            # Same depth group - add to current segment
            current_points.append(points[i])
        else:
            # Depth changed - save current segment and start new one
            if len(current_points) >= 2:
                segments.append((current_points, current_depth))
            # Start new segment with overlap point for continuity
            current_points = [points[i-1], points[i]]
            current_depth = depths[i]
    
    # Save final segment
    if len(current_points) >= 2:
        segments.append((current_points, current_depth))
    
    return segments


def toolpaths_to_gcode(
    toolpaths: list[Toolpath],
    tool: EndMill | VBit,
    safe_z: float = 5.0,
    operation_name: str | None = None,
    origin_offset: tuple[float, float] = (0.0, 0.0),
) -> str:
    """Convert toolpaths to G-code string.
    
    Args:
        toolpaths: List of toolpaths to convert
        tool: Tool being used
        safe_z: Safe retract height
        operation_name: Name for G-code header comment
        origin_offset: (X, Y) offset to subtract from all coordinates
    """
    from .gcode import GCodeBuilder, GCodeSettings
    
    settings = GCodeSettings(safe_z=safe_z)
    builder = GCodeBuilder(settings)
    
    offset_x, offset_y = origin_offset
    
    if operation_name:
        header_comment = f"{operation_name} with {tool.name}"
    else:
        header_comment = f"Cutting with {tool.name}"
    
    builder.header(header_comment)
    builder.spindle_on(tool.spindle_rpm)
    
    current_depth = None
    
    for tp in toolpaths:
        # Check if points have Z values (Nx3 array)
        has_z = tp.points.shape[1] >= 3 if len(tp.points.shape) > 1 else False
        
        # Comment for depth changes
        if not has_z and tp.z_depth != current_depth:
            builder.comment(f"Depth: {tp.z_depth:.3f}mm")
            current_depth = tp.z_depth
        
        # Apply origin offset to coordinates
        start_x = tp.points[0, 0] - offset_x
        start_y = tp.points[0, 1] - offset_y
        start_z = tp.points[0, 2] if has_z else tp.z_depth
        
        # Rapid to start position
        builder.rapid_z(safe_z)
        builder.rapid_xy(start_x, start_y)
        
        # Plunge to starting depth
        builder.plunge(start_z, tool.plunge_rate)
        
        # Handle arc moves
        if tp.is_arc and tp.arc_center is not None and len(tp.points) == 2:
            end_x = tp.points[1, 0] - offset_x
            end_y = tp.points[1, 1] - offset_y
            end_z = tp.points[1, 2] if has_z else tp.z_depth
            i, j = tp.arc_center
            
            cmd = "G2" if tp.clockwise else "G3"
            if has_z and abs(end_z - start_z) > 0.001:
                builder.add_line(f"{cmd}X{end_x:.4f}Y{end_y:.4f}Z{end_z:.4f}I{i:.4f}J{j:.4f}F{tool.feed_rate}")
            else:
                builder.add_line(f"{cmd}X{end_x:.4f}Y{end_y:.4f}I{i:.4f}J{j:.4f}F{tool.feed_rate}")
        else:
            # Cut along path (linear or with Z interpolation)
            for point in tp.points[1:]:
                px = point[0] - offset_x
                py = point[1] - offset_y
                
                if has_z:
                    pz = point[2]
                    builder.add_line(f"G1X{px:.4f}Y{py:.4f}Z{pz:.4f}F{tool.feed_rate}")
                else:
                    builder.linear_xy(px, py, tool.feed_rate)
        
        # Retract
        builder.rapid_z(safe_z)
    
    builder.footer()
    return builder.get_gcode()
