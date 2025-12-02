"""Inlay generation for pocket and plug toolpaths."""

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np
from shapely.geometry import Polygon
from shapely.affinity import scale

from .geometry import CarvePath
from .tools import EndMill, VBit
from .toolpaths import (
    Toolpath, RoughingParams, VBitParams, ClearingStrategy,
    generate_roughing_toolpaths, generate_vbit_toolpaths
)


class InlayPart(Enum):
    """Which part of the inlay to generate."""
    POCKET = "pocket"  # Female part (carved into base)
    PLUG = "plug"      # Male part (the inlay piece)
    BOTH = "both"      # Generate both


@dataclass
class InlayParams:
    """Parameters for inlay generation."""
    # Tools
    roughing_tool: EndMill
    vbit_tool: VBit
    
    # Depths
    pocket_depth: float       # Total pocket depth
    plug_depth: float         # Total plug depth (usually pocket_depth + glue_gap)
    glue_gap: float = 0.5     # Extra depth on plug for glue space
    
    # Options
    stock_to_leave: float = 0.1
    climb_milling: bool = True
    clearing_strategy: ClearingStrategy = ClearingStrategy.CONTOUR
    
    # Mirror settings for plug
    mirror_axis: str = "x"    # "x" or "y" - axis to mirror around
    plug_border: float = 5.0  # Border around plug to clear (mm)
    
    # What to generate
    part: InlayPart = InlayPart.BOTH


@dataclass
class InlayResult:
    """Result of inlay toolpath generation."""
    # Pocket toolpaths
    pocket_roughing: list[Toolpath]
    pocket_vbit: list[Toolpath]
    
    # Plug toolpaths
    plug_roughing: list[Toolpath]
    plug_vbit: list[Toolpath]
    
    # Mirrored paths (for visualization)
    mirrored_paths: list[CarvePath]
    
    # Parameters used
    params: InlayParams


def generate_inlay_toolpaths(
    paths: list[CarvePath],
    params: InlayParams,
) -> InlayResult:
    """Generate complete inlay toolpaths for pocket and/or plug.
    
    Args:
        paths: Design paths (closed shapes)
        params: Inlay parameters
        
    Returns:
        InlayResult with all generated toolpaths
    """
    pocket_roughing = []
    pocket_vbit = []
    plug_roughing = []
    plug_vbit = []
    mirrored_paths = []
    
    # Filter to closed paths only
    closed_paths = [p for p in paths if p.is_closed]
    
    if not closed_paths:
        return InlayResult(
            pocket_roughing=[], pocket_vbit=[],
            plug_roughing=[], plug_vbit=[],
            mirrored_paths=[], params=params
        )
    
    # Generate pocket toolpaths
    if params.part in (InlayPart.POCKET, InlayPart.BOTH):
        pocket_roughing, pocket_vbit = _generate_pocket(closed_paths, params)
    
    # Generate plug toolpaths (mirrored)
    if params.part in (InlayPart.PLUG, InlayPart.BOTH):
        mirrored = _mirror_paths(closed_paths, params.mirror_axis)
        mirrored_paths = mirrored
        plug_roughing, plug_vbit = _generate_plug(mirrored, params)
    
    return InlayResult(
        pocket_roughing=pocket_roughing,
        pocket_vbit=pocket_vbit,
        plug_roughing=plug_roughing,
        plug_vbit=plug_vbit,
        mirrored_paths=mirrored_paths,
        params=params,
    )


def _generate_pocket(
    paths: list[CarvePath],
    params: InlayParams,
) -> tuple[list[Toolpath], list[Toolpath]]:
    """Generate pocket (female) toolpaths."""
    
    # Roughing pass
    roughing_params = RoughingParams(
        tool=params.roughing_tool,
        target_depth=params.pocket_depth,
        stock_to_leave=params.stock_to_leave,
        climb_milling=params.climb_milling,
        strategy=params.clearing_strategy,
    )
    roughing_toolpaths = generate_roughing_toolpaths(paths, roughing_params)
    
    # V-bit finishing pass
    vbit_params = VBitParams(
        tool=params.vbit_tool,
        max_depth=params.pocket_depth,
        roughing_diameter=params.roughing_tool.diameter,
        climb_milling=params.climb_milling,
    )
    vbit_toolpaths = generate_vbit_toolpaths(paths, vbit_params)
    
    return roughing_toolpaths, vbit_toolpaths


def _generate_plug(
    mirrored_paths: list[CarvePath],
    params: InlayParams,
) -> tuple[list[Toolpath], list[Toolpath]]:
    """Generate plug (male) toolpaths - clears area AROUND the design.
    
    The plug is the inverse of the pocket - we clear material around
    the design, leaving it standing proud to fit into the pocket.
    """
    from shapely.geometry import Polygon, box
    from shapely.ops import unary_union
    
    # Plug depth includes glue gap
    plug_total_depth = params.plug_depth
    
    # Get bounding box of all paths
    all_points = np.vstack([p.points for p in mirrored_paths])
    min_x, min_y = all_points.min(axis=0)
    max_x, max_y = all_points.max(axis=0)
    
    # Create outer boundary with border
    border = params.plug_border
    outer_box = box(min_x - border, min_y - border, max_x + border, max_y + border)
    
    # Create polygons from mirrored paths (the design to keep)
    design_polys = []
    for path in mirrored_paths:
        if path.is_closed and len(path.points) >= 3:
            try:
                poly = Polygon(path.points)
                if poly.is_valid:
                    design_polys.append(poly)
                else:
                    design_polys.append(poly.buffer(0))
            except Exception:
                pass
    
    if not design_polys:
        return [], []
    
    # Union of all design shapes
    design_union = unary_union(design_polys)
    
    # Subtract design from outer box = area to clear
    clearance_area = outer_box.difference(design_union)
    
    if clearance_area.is_empty:
        return [], []
    
    # Generate plug clearing toolpaths with TAPERED profile
    # The plug must be narrower at top, wider at bottom to match pocket V-groove
    # At each depth, account for how wide the V-bit will cut
    
    import math
    
    tool_diameter = params.roughing_tool.diameter
    stepover = tool_diameter * 0.4  # 40% stepover
    tool_radius = tool_diameter / 2
    
    # V-bit geometry - determines taper angle
    vbit_half_angle = math.radians(params.vbit_tool.angle / 2)
    tan_half = math.tan(vbit_half_angle)
    
    # Calculate depth passes  
    depth_per_pass = 1.5  # mm per pass
    num_passes = max(1, int(np.ceil(plug_total_depth / depth_per_pass)))
    
    roughing_toolpaths = []
    
    for pass_num in range(num_passes):
        z_depth = -min((pass_num + 1) * depth_per_pass, plug_total_depth)
        current_depth = abs(z_depth)
        
        # At this depth, the V-bit will cut this far from the design edge
        # So the roughing must stay further from the design at deeper cuts
        vbit_width_at_depth = current_depth / tan_half
        
        # Minimum distance from design edge at this depth
        min_clearance = tool_radius + params.stock_to_leave + vbit_width_at_depth
        
        # Start from outer boundary, work inward toward the design
        current_inset = tool_radius
        
        while True:
            # Inset from outer box
            inner_box = outer_box.buffer(-current_inset)
            if inner_box.is_empty:
                break
            
            # The cutting path is the inner_box minus the design (with depth-dependent clearance)
            design_with_clearance = design_union.buffer(min_clearance)
            ring = inner_box.difference(design_with_clearance)
            
            if ring.is_empty:
                break
            
            def add_ring_path(geom):
                if hasattr(geom, 'exterior'):
                    coords = np.array(geom.exterior.coords)
                    # Reverse for climb milling
                    if params.climb_milling:
                        coords = coords[::-1]
                    roughing_toolpaths.append(Toolpath(
                        points=coords,
                        z_depth=z_depth,
                        is_rapid=False,
                    ))
            
            if hasattr(ring, 'geoms'):
                for g in ring.geoms:
                    add_ring_path(g)
            elif hasattr(ring, 'exterior'):
                add_ring_path(ring)
            
            current_inset += stepover
            
            # Stop when we've reached the design boundary
            if current_inset > border:
                break
    
    # V-bit finishing for plug - creates tapered edge on OUTSIDE of design
    # The V-bit starts at design edge (Z=0) and cuts outward with increasing depth
    # At offset distance d from edge, Z depth = -d * tan(half_angle)
    
    vbit_toolpaths = []
    
    # V-bit step size (how far to move outward between passes)
    vbit_stepover = 0.3  # mm - small steps for smooth finish
    
    # Maximum offset based on max depth
    max_offset = plug_total_depth * tan_half
    
    current_offset = 0
    while current_offset <= max_offset:
        # Calculate Z depth at this offset
        if current_offset == 0:
            z_depth = 0  # Start at surface on the design edge
        else:
            z_depth = -current_offset / tan_half
        
        # Don't go deeper than plug depth
        if abs(z_depth) > plug_total_depth:
            z_depth = -plug_total_depth
        
        # Create offset polygon from design
        if current_offset == 0:
            offset_poly = design_union
        else:
            offset_poly = design_union.buffer(current_offset)
        
        if offset_poly.is_empty:
            break
        
        def add_vbit_path(geom):
            if hasattr(geom, 'exterior'):
                coords = np.array(geom.exterior.coords)
                # For plug V-bit cutting outward, use CW for climb milling
                if not params.climb_milling:
                    coords = coords[::-1]
                vbit_toolpaths.append(Toolpath(
                    points=coords,
                    z_depth=z_depth,
                    is_rapid=False,
                ))
        
        if hasattr(offset_poly, 'geoms'):
            for g in offset_poly.geoms:
                add_vbit_path(g)
        elif hasattr(offset_poly, 'exterior'):
            add_vbit_path(offset_poly)
        
        current_offset += vbit_stepover
    
    return roughing_toolpaths, vbit_toolpaths


def _mirror_paths(
    paths: list[CarvePath],
    axis: str,
) -> list[CarvePath]:
    """Mirror paths around the specified axis.
    
    For X axis: mirror horizontally (flip left-right)
    For Y axis: mirror vertically (flip top-bottom)
    
    The mirror is centered on the bounding box of all paths.
    """
    if not paths:
        return []
    
    # Find bounding box center of all paths
    all_points = np.vstack([p.points for p in paths])
    min_x, min_y = all_points.min(axis=0)
    max_x, max_y = all_points.max(axis=0)
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2
    
    mirrored = []
    for path in paths:
        # Mirror points
        if axis.lower() == "x":
            # Mirror around vertical axis (flip X)
            new_points = path.points.copy()
            new_points[:, 0] = 2 * center_x - new_points[:, 0]
        else:
            # Mirror around horizontal axis (flip Y)
            new_points = path.points.copy()
            new_points[:, 1] = 2 * center_y - new_points[:, 1]
        
        # Reverse winding order to maintain correct inside/outside
        new_points = new_points[::-1]
        
        mirrored.append(CarvePath(
            id=f"{path.id}_mirrored",
            points=new_points,
            is_closed=path.is_closed,
            layer=path.layer,
            color=path.color,
        ))
    
    return mirrored


def inlay_to_gcode(
    result: InlayResult,
    safe_z: float = 5.0,
    origin_offset: tuple[float, float] = (0.0, 0.0),
) -> dict[str, str]:
    """Convert inlay result to G-code strings.
    
    Returns:
        Dictionary with keys:
        - 'pocket_roughing': G-code for pocket roughing
        - 'pocket_vbit': G-code for pocket V-bit
        - 'pocket_combined': Combined pocket G-code
        - 'plug_roughing': G-code for plug roughing
        - 'plug_vbit': G-code for plug V-bit
        - 'plug_combined': Combined plug G-code
    """
    from .toolpaths import toolpaths_to_gcode
    
    gcode = {}
    params = result.params
    
    # Pocket G-code
    if result.pocket_roughing:
        gcode['pocket_roughing'] = toolpaths_to_gcode(
            result.pocket_roughing,
            params.roughing_tool,
            safe_z=safe_z,
            operation_name="Pocket Roughing",
            origin_offset=origin_offset,
        )
    
    if result.pocket_vbit:
        gcode['pocket_vbit'] = toolpaths_to_gcode(
            result.pocket_vbit,
            params.vbit_tool,
            safe_z=safe_z,
            operation_name="Pocket V-Bit Finishing",
            origin_offset=origin_offset,
        )
    
    # Combined pocket
    if result.pocket_roughing or result.pocket_vbit:
        gcode['pocket_combined'] = _combine_gcode(
            result.pocket_roughing, result.pocket_vbit,
            params.roughing_tool, params.vbit_tool,
            safe_z, origin_offset, "Pocket"
        )
    
    # Plug G-code
    if result.plug_roughing:
        gcode['plug_roughing'] = toolpaths_to_gcode(
            result.plug_roughing,
            params.roughing_tool,
            safe_z=safe_z,
            operation_name="Plug Roughing",
            origin_offset=origin_offset,
        )
    
    if result.plug_vbit:
        gcode['plug_vbit'] = toolpaths_to_gcode(
            result.plug_vbit,
            params.vbit_tool,
            safe_z=safe_z,
            operation_name="Plug V-Bit Finishing",
            origin_offset=origin_offset,
        )
    
    # Combined plug
    if result.plug_roughing or result.plug_vbit:
        gcode['plug_combined'] = _combine_gcode(
            result.plug_roughing, result.plug_vbit,
            params.roughing_tool, params.vbit_tool,
            safe_z, origin_offset, "Plug"
        )
    
    return gcode


def _combine_gcode(
    roughing: list[Toolpath],
    vbit: list[Toolpath],
    roughing_tool: EndMill,
    vbit_tool: VBit,
    safe_z: float,
    origin_offset: tuple[float, float],
    part_name: str,
) -> str:
    """Combine roughing and V-bit into single G-code with tool change."""
    from .gcode import GCodeBuilder, GCodeSettings
    
    settings = GCodeSettings(safe_z=safe_z)
    builder = GCodeBuilder(settings)
    offset_x, offset_y = origin_offset
    
    builder.header(f"{part_name} - Combined Roughing and V-Bit")
    
    # === Roughing pass ===
    if roughing:
        builder.add_line(f"; === {part_name} Roughing with {roughing_tool.name} ===")
        builder.add_line("T1 M6 ; Tool 1 - End Mill")
        builder.spindle_on(roughing_tool.spindle_rpm)
        
        for tp in roughing:
            has_z = tp.points.shape[1] >= 3 if len(tp.points.shape) > 1 else False
            
            start_x = tp.points[0, 0] - offset_x
            start_y = tp.points[0, 1] - offset_y
            start_z = tp.points[0, 2] if has_z else tp.z_depth
            
            builder.rapid_z(safe_z)
            builder.rapid_xy(start_x, start_y)
            builder.plunge(start_z, roughing_tool.plunge_rate)
            
            for point in tp.points[1:]:
                px = point[0] - offset_x
                py = point[1] - offset_y
                if has_z:
                    builder.add_line(f"G1X{px:.4f}Y{py:.4f}Z{point[2]:.4f}F{roughing_tool.feed_rate}")
                else:
                    builder.linear_xy(px, py, roughing_tool.feed_rate)
        
        builder.rapid_z(safe_z)
        builder.spindle_off()
        builder.add_line("")
    
    # === V-bit finishing pass ===
    if vbit:
        builder.add_line(f"; === {part_name} V-Bit Finishing with {vbit_tool.name} ===")
        builder.add_line("T2 M6 ; Tool 2 - V-Bit")
        builder.add_line("M0 ; Pause for tool change")
        builder.spindle_on(vbit_tool.spindle_rpm)
        
        for tp in vbit:
            has_z = tp.points.shape[1] >= 3 if len(tp.points.shape) > 1 else False
            
            start_x = tp.points[0, 0] - offset_x
            start_y = tp.points[0, 1] - offset_y
            start_z = tp.points[0, 2] if has_z else tp.z_depth
            
            builder.rapid_z(safe_z)
            builder.rapid_xy(start_x, start_y)
            builder.plunge(start_z, vbit_tool.plunge_rate)
            
            if tp.is_arc and tp.arc_center is not None and len(tp.points) == 2:
                end_x = tp.points[1, 0] - offset_x
                end_y = tp.points[1, 1] - offset_y
                end_z = tp.points[1, 2] if has_z else tp.z_depth
                i, j = tp.arc_center
                cmd = "G2" if tp.clockwise else "G3"
                if has_z and abs(end_z - start_z) > 0.001:
                    builder.add_line(f"{cmd}X{end_x:.4f}Y{end_y:.4f}Z{end_z:.4f}I{i:.4f}J{j:.4f}F{vbit_tool.feed_rate}")
                else:
                    builder.add_line(f"{cmd}X{end_x:.4f}Y{end_y:.4f}I{i:.4f}J{j:.4f}F{vbit_tool.feed_rate}")
            else:
                for point in tp.points[1:]:
                    px = point[0] - offset_x
                    py = point[1] - offset_y
                    if has_z:
                        builder.add_line(f"G1X{px:.4f}Y{py:.4f}Z{point[2]:.4f}F{vbit_tool.feed_rate}")
                    else:
                        builder.linear_xy(px, py, vbit_tool.feed_rate)
        
        builder.rapid_z(safe_z)
    
    builder.footer()
    return builder.get_gcode()
