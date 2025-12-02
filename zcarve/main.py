"""Main application window for ZCarve."""

import sys
from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt, QPointF
from PySide6.QtGui import QAction, QPainter, QPen, QColor, QBrush
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QFormLayout,
    QToolBar,
    QFileDialog,
    QStatusBar,
    QGraphicsScene,
    QGraphicsView,
    QGraphicsPathItem,
    QGraphicsRectItem,
    QDockWidget,
    QListWidget,
    QListWidgetItem,
    QLabel,
    QDoubleSpinBox,
    QComboBox,
    QMessageBox,
    QGroupBox,
    QPushButton,
    QTabWidget,
)
from PySide6.QtGui import QPainterPath

from .geometry import SVGLoader, CarvePath
from .tools import ToolLibrary
from .dialogs import RoughingDialog, VBitDialog, ToolpathDialog
from .toolpaths import generate_roughing_toolpaths, generate_vbit_toolpaths, toolpaths_to_gcode
from .inlay import generate_inlay_toolpaths, inlay_to_gcode, InlayPart

import numpy as np
import pyqtgraph as pg
import pyqtgraph.opengl as gl


class GLToolpathView(gl.GLViewWidget):
    """Single 3D view for toolpaths."""
    
    def __init__(self):
        super().__init__()
        self.setBackgroundColor('#2d2d2d')
        self.setCameraPosition(distance=100, elevation=30, azimuth=45)
        
        # Enable mouse panning
        self.opts['center'] = pg.Vector(0, 0, 0)
        self._last_pos = None
        
        # Add grid
        self.grid = gl.GLGridItem()
        self.grid.setSize(100, 100)
        self.grid.setSpacing(10, 10)
        self.grid.setColor((80, 80, 80, 100))
        self.addItem(self.grid)
        
        # Store current toolpaths
        self._roughing: list = []
        self._vbit: list = []
        self._design_paths: list = []
        self._center = (0, 0)
        self._stock_size = None  # (width, height, depth)
    
    def set_stock_size(self, width: float, height: float, depth: float):
        """Set stock dimensions for visualization."""
        self._stock_size = (width, height, depth)
        self._redraw()
    
    def mousePressEvent(self, ev):
        """Handle mouse press - track position."""
        self._last_pos = ev.pos()
        ev.accept()
    
    def mouseMoveEvent(self, ev):
        """Handle mouse drag - left=rotate, right=pan, wheel=zoom."""
        if self._last_pos is None:
            self._last_pos = ev.pos()
            return
        
        diff = ev.pos() - self._last_pos
        self._last_pos = ev.pos()
        
        if ev.buttons() == Qt.MouseButton.LeftButton:
            # Rotate
            self.orbit(-diff.x(), diff.y())
        elif ev.buttons() == Qt.MouseButton.RightButton:
            # Pan - scale by distance so point follows cursor
            scale = self.opts['distance'] * 0.002
            self.pan(diff.x() * scale, -diff.y() * scale, 0)
        elif ev.buttons() == Qt.MouseButton.MiddleButton:
            # Also pan with middle button
            scale = self.opts['distance'] * 0.002
            self.pan(diff.x() * scale, -diff.y() * scale, 0)
        
        ev.accept()
    
    def wheelEvent(self, ev):
        """Handle mouse wheel for zoom."""
        delta = ev.angleDelta().y()
        self.opts['distance'] *= 0.999 ** delta
        self.update()
        ev.accept()
    
    def set_toolpaths(self, roughing: list, vbit: list, design_paths: list = None):
        """Set toolpaths to display."""
        self._roughing = roughing or []
        self._vbit = vbit or []
        self._design_paths = design_paths or []
        self._redraw()
    
    def clear(self):
        """Clear all toolpaths."""
        self._roughing = []
        self._vbit = []
        self._design_paths = []
        self._redraw()
    
    def _redraw(self):
        """Redraw all toolpath lines."""
        # Remove all items except grid
        for item in list(self.items):
            if item is not self.grid:
                self.removeItem(item)
        
        # Calculate center from all points
        all_pts = []
        for path in self._design_paths:
            if hasattr(path, 'points'):
                all_pts.extend(path.points[:, :2])
        for tp in self._roughing + self._vbit:
            all_pts.extend(tp.points[:, :2])
        
        if all_pts:
            all_pts = np.array(all_pts)
            cx = (all_pts[:, 0].min() + all_pts[:, 0].max()) / 2
            cy = (all_pts[:, 1].min() + all_pts[:, 1].max()) / 2
            width = max(all_pts[:, 0].max() - all_pts[:, 0].min(),
                       all_pts[:, 1].max() - all_pts[:, 1].min())
            self._center = (cx, cy)
            
            # Resize grid
            self.grid.setSize(width * 1.5, width * 1.5)
            self.grid.setSpacing(width / 10, width / 10)
            
            # Adjust camera
            self.setCameraPosition(distance=width * 1.2)
        else:
            self._center = (0, 0)
        
        cx, cy = self._center
        
        # Draw design outline (cyan)
        for path in self._design_paths:
            if hasattr(path, 'points') and len(path.points) >= 2:
                pts = path.points
                line_pts = np.column_stack([
                    pts[:, 0] - cx,
                    pts[:, 1] - cy,
                    np.zeros(len(pts))
                ])
                self.addItem(gl.GLLinePlotItem(
                    pos=line_pts, color=(0, 1, 1, 0.6), width=2, antialias=True
                ))
        
        # Draw roughing (blue)
        for tp in self._roughing:
            self._draw_toolpath(tp, (0.3, 0.5, 1, 0.9), cx, cy)
        
        # Draw v-bit (orange)
        for tp in self._vbit:
            self._draw_toolpath(tp, (1, 0.5, 0.2, 0.9), cx, cy)
        
        # Draw stock boundary (if set)
        if self._stock_size:
            self._draw_stock_boundary(cx, cy)
    
    def _draw_toolpath(self, tp, color: tuple, cx: float, cy: float):
        """Draw a single toolpath."""
        pts = tp.points
        if len(pts) < 2:
            return
        
        if pts.shape[1] >= 3:
            z_vals = pts[:, 2]
        else:
            z_vals = np.full(len(pts), tp.z_depth)
        
        line_pts = np.column_stack([
            pts[:, 0] - cx,
            pts[:, 1] - cy,
            z_vals
        ])
        
        self.addItem(gl.GLLinePlotItem(
            pos=line_pts, color=color, width=1.5, antialias=True
        ))
    
    def _draw_stock_boundary(self, cx: float, cy: float):
        """Draw stock boundary as a 3D wireframe box."""
        w, h, d = self._stock_size
        hw, hh = w / 2, h / 2  # Half dimensions
        
        # Stock center is at (w/2, h/2) in world coords
        # Toolpaths are offset by -cx, -cy to center them
        # So stock center after same offset is at (w/2 - cx, h/2 - cy)
        stock_cx = hw - cx
        stock_cy = hh - cy
        
        top = np.array([
            [stock_cx - hw, stock_cy - hh, 0],
            [stock_cx + hw, stock_cy - hh, 0],
            [stock_cx + hw, stock_cy + hh, 0],
            [stock_cx - hw, stock_cy + hh, 0],
            [stock_cx - hw, stock_cy - hh, 0],  # Close the loop
        ])
        
        # Draw top edge (brown)
        self.addItem(gl.GLLinePlotItem(
            pos=top, color=(0.6, 0.4, 0.2, 0.8), width=2, antialias=True
        ))


class DesignView3D(QWidget):
    """Tabbed 3D view for pocket and plug toolpaths."""
    
    def __init__(self):
        super().__init__()
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Tab widget
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)
        
        # Pocket view
        self.pocket_view = GLToolpathView()
        self.tabs.addTab(self.pocket_view, "Pocket")
        
        # Plug view (initially hidden)
        self.plug_view = GLToolpathView()
        
        # Controls info
        info = QLabel("🖱 Left: Rotate | Right: Pan | Scroll: Zoom")
        info.setStyleSheet("color: #888; font-size: 10px; padding: 2px;")
        layout.addWidget(info)
        
        # Store data
        self._design_paths: list[CarvePath] = []
        self._pocket_roughing: list = []
        self._pocket_vbit: list = []
        self._plug_roughing: list = []
        self._plug_vbit: list = []
        self._stock_size = (100, 100, 20)  # Default
    
    def update_stock(self, width: float, height: float, depth: float = 20.0):
        """Update stock dimensions."""
        self._stock_size = (width, height, depth)
        # Update plug view with stock size
        self.plug_view.set_stock_size(width, height, depth)
    
    def display_paths(self, paths: list[CarvePath]):
        """Display design paths."""
        self._design_paths = paths
        self._update_views()
    
    def clear_design(self):
        """Clear design paths."""
        self._design_paths = []
        self._update_views()
    
    def clear_toolpaths(self):
        """Clear all toolpaths."""
        self._pocket_roughing = []
        self._pocket_vbit = []
        self._plug_roughing = []
        self._plug_vbit = []
        self._update_views()
    
    def display_toolpaths(self, toolpaths, tool_type: str = "roughing", clear_existing: bool = True):
        """Display toolpaths (legacy interface)."""
        if clear_existing:
            self._pocket_roughing = []
            self._pocket_vbit = []
        
        if tool_type == "roughing":
            self._pocket_roughing = toolpaths
        else:
            self._pocket_vbit = toolpaths
        
        self._update_views()
    
    def display_all_toolpaths(self, pocket_roughing, pocket_vbit, plug_roughing, plug_vbit):
        """Display all toolpaths (pocket and plug)."""
        self._pocket_roughing = pocket_roughing or []
        self._pocket_vbit = pocket_vbit or []
        self._plug_roughing = plug_roughing or []
        self._plug_vbit = plug_vbit or []
        self._update_views()
    
    def _update_views(self):
        """Update both pocket and plug views."""
        # Update pocket view
        self.pocket_view.set_toolpaths(
            self._pocket_roughing, 
            self._pocket_vbit,
            self._design_paths
        )
        
        # Handle plug tab
        has_plug = bool(self._plug_roughing or self._plug_vbit)
        
        if has_plug:
            # Add plug tab if not present
            if self.tabs.count() < 2:
                self.tabs.addTab(self.plug_view, "Plug")
            
            # Update plug view
            self.plug_view.set_toolpaths(
                self._plug_roughing,
                self._plug_vbit,
                []  # Don't show design outline on plug (it's mirrored)
            )
        else:
            # Remove plug tab if present
            if self.tabs.count() > 1:
                self.tabs.removeTab(1)


class MainWindow(QMainWindow):
    """Main application window."""
    
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("ZCarve - V-Carve Inlay G-code Generator")
        self.setMinimumSize(1024, 768)
        
        # Initialize components
        self.svg_loader = SVGLoader()
        self.tool_library = ToolLibrary()
        self.current_file: Optional[Path] = None
        
        # Stock dimensions (mm)
        self.stock_x: float = 100.0
        self.stock_y: float = 100.0
        self.stock_z: float = 20.0
        self.origin_position: str = "center"  # lower_left, lower_right, upper_left, upper_right, center
        
        # Toolpath cache
        self._roughing_toolpaths: list = []
        self._roughing_params = None
        self._vbit_toolpaths: list = []
        self._vbit_params = None
        
        # Set up UI
        self._setup_ui()
        self._setup_menus()
        self._setup_toolbar()
        self._setup_docks()
        self._setup_statusbar()
    
    def _setup_ui(self):
        """Set up the main UI layout."""
        # Central widget - 3D design view
        self.design_view = DesignView3D()
        self.design_view.update_stock(self.stock_x, self.stock_y, self.stock_z)
        self.setCentralWidget(self.design_view)
    
    def _setup_menus(self):
        """Set up menu bar."""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("&File")
        
        open_action = QAction("&Open SVG...", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self.open_svg)
        file_menu.addAction(open_action)
        
        file_menu.addSeparator()
        
        export_menu = file_menu.addMenu("&Export G-code")
        
        export_roughing = QAction("&Roughing Pass...", self)
        export_roughing.triggered.connect(self.export_roughing)
        export_menu.addAction(export_roughing)
        
        export_finishing = QAction("&V-Bit Finishing...", self)
        export_finishing.triggered.connect(self.export_finishing)
        export_menu.addAction(export_finishing)
        
        file_menu.addSeparator()
        
        preview_menu = file_menu.addMenu("&Preview Toolpaths")
        
        preview_roughing = QAction("Preview &Roughing", self)
        preview_roughing.triggered.connect(self.preview_roughing)
        preview_menu.addAction(preview_roughing)
        
        preview_finishing = QAction("Preview &V-Bit", self)
        preview_finishing.triggered.connect(self.preview_finishing)
        preview_menu.addAction(preview_finishing)
        
        clear_preview = QAction("&Clear Preview", self)
        clear_preview.triggered.connect(self.clear_preview)
        preview_menu.addAction(clear_preview)
        
        file_menu.addSeparator()
        
        quit_action = QAction("&Quit", self)
        quit_action.setShortcut("Ctrl+Q")
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)
        
        # Tools menu
        tools_menu = menubar.addMenu("&Tools")
        
        tool_library_action = QAction("&Tool Library...", self)
        tool_library_action.triggered.connect(self.show_tool_library)
        tools_menu.addAction(tool_library_action)
        
        # View menu
        view_menu = menubar.addMenu("&View")
        
        zoom_fit = QAction("Zoom to &Fit", self)
        zoom_fit.setShortcut("F")
        zoom_fit.triggered.connect(self.zoom_fit)
        view_menu.addAction(zoom_fit)
        
        zoom_reset = QAction("&Reset Zoom", self)
        zoom_reset.setShortcut("R")
        zoom_reset.triggered.connect(self.zoom_reset)
        view_menu.addAction(zoom_reset)
    
    def _setup_toolbar(self):
        """Set up the toolbar."""
        toolbar = QToolBar("Main Toolbar")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)
        
        open_action = QAction("Open", self)
        open_action.triggered.connect(self.open_svg)
        toolbar.addAction(open_action)
        
        toolbar.addSeparator()
        
        zoom_fit_action = QAction("Fit", self)
        zoom_fit_action.triggered.connect(self.zoom_fit)
        toolbar.addAction(zoom_fit_action)
    
    def _setup_docks(self):
        """Set up dock widgets."""
        # Stock dock (top of right side)
        self.stock_dock = QDockWidget("Stock Material", self)
        self.stock_dock.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea
        )
        
        stock_widget = QWidget()
        stock_layout = QFormLayout(stock_widget)
        stock_layout.setContentsMargins(10, 10, 10, 10)
        
        self.stock_x_spin = QDoubleSpinBox()
        self.stock_x_spin.setRange(1.0, 2000.0)
        self.stock_x_spin.setValue(self.stock_x)
        self.stock_x_spin.setSuffix(" mm")
        self.stock_x_spin.setDecimals(1)
        self.stock_x_spin.valueChanged.connect(self._on_stock_changed)
        stock_layout.addRow("Width (X):", self.stock_x_spin)
        
        self.stock_y_spin = QDoubleSpinBox()
        self.stock_y_spin.setRange(1.0, 2000.0)
        self.stock_y_spin.setValue(self.stock_y)
        self.stock_y_spin.setSuffix(" mm")
        self.stock_y_spin.setDecimals(1)
        self.stock_y_spin.valueChanged.connect(self._on_stock_changed)
        stock_layout.addRow("Height (Y):", self.stock_y_spin)
        
        self.stock_z_spin = QDoubleSpinBox()
        self.stock_z_spin.setRange(0.5, 200.0)
        self.stock_z_spin.setValue(self.stock_z)
        self.stock_z_spin.setSuffix(" mm")
        self.stock_z_spin.setDecimals(1)
        self.stock_z_spin.valueChanged.connect(self._on_stock_changed)
        stock_layout.addRow("Thickness (Z):", self.stock_z_spin)
        
        # Origin position selector
        self.origin_combo = QComboBox()
        self.origin_combo.addItem("Center", "center")
        self.origin_combo.addItem("Lower Left", "lower_left")
        self.origin_combo.addItem("Lower Right", "lower_right")
        self.origin_combo.addItem("Upper Left", "upper_left")
        self.origin_combo.addItem("Upper Right", "upper_right")
        self.origin_combo.currentIndexChanged.connect(self._on_origin_changed)
        stock_layout.addRow("Origin:", self.origin_combo)
        
        self.stock_dock.setWidget(stock_widget)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.stock_dock)
        
        # Paths dock (below stock on right side)
        self.paths_dock = QDockWidget("Paths", self)
        self.paths_dock.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea
        )
        
        paths_widget = QWidget()
        paths_layout = QVBoxLayout(paths_widget)
        paths_layout.setContentsMargins(5, 5, 5, 5)
        
        self.paths_list = QListWidget()
        paths_layout.addWidget(self.paths_list)
        
        # Main toolpath generation button
        self.generate_btn = QPushButton("⚙ Generate Toolpaths...")
        self.generate_btn.setToolTip("Generate roughing + V-bit toolpaths")
        self.generate_btn.clicked.connect(self.generate_toolpaths)
        paths_layout.addWidget(self.generate_btn)
        
        # Save G-Code button
        self.save_gcode_btn = QPushButton("💾 Save G-Code...")
        self.save_gcode_btn.setToolTip("Export generated toolpaths to G-code file")
        self.save_gcode_btn.clicked.connect(self.save_gcode)
        self.save_gcode_btn.setEnabled(False)  # Disabled until toolpaths generated
        paths_layout.addWidget(self.save_gcode_btn)
        
        paths_layout.addSpacing(5)
        
        # Clear preview button
        self.clear_preview_btn = QPushButton("Clear Preview")
        self.clear_preview_btn.clicked.connect(self.clear_preview)
        paths_layout.addWidget(self.clear_preview_btn)
        
        self.paths_dock.setWidget(paths_widget)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.paths_dock)
    
    def _on_stock_changed(self):
        """Handle stock dimension changes."""
        self.stock_x = self.stock_x_spin.value()
        self.stock_y = self.stock_y_spin.value()
        self.stock_z = self.stock_z_spin.value()
        self.design_view.update_stock(self.stock_x, self.stock_y, self.stock_z)
    
    def _on_origin_changed(self):
        """Handle origin position change."""
        self.origin_position = self.origin_combo.currentData()
    
    def get_origin_offset(self) -> tuple[float, float]:
        """Get the X, Y offset to apply based on origin position.
        
        Returns offset to subtract from coordinates to move origin.
        """
        if self.origin_position == "lower_left":
            return (0.0, 0.0)
        elif self.origin_position == "lower_right":
            return (self.stock_x, 0.0)
        elif self.origin_position == "upper_left":
            return (0.0, self.stock_y)
        elif self.origin_position == "upper_right":
            return (self.stock_x, self.stock_y)
        elif self.origin_position == "center":
            return (self.stock_x / 2, self.stock_y / 2)
        return (0.0, 0.0)
    
    def _setup_statusbar(self):
        """Set up the status bar."""
        self.statusbar = QStatusBar()
        self.setStatusBar(self.statusbar)
        self.statusbar.showMessage("Ready")
    
    def _update_paths_list(self):
        """Update the paths list from loaded SVG."""
        self.paths_list.clear()
        for path in self.svg_loader.paths:
            label = path.id
            if path.layer:
                label = f"{path.layer}/{path.id}"
            item = QListWidgetItem(label)
            item.setData(Qt.ItemDataRole.UserRole, path.id)
            item.setCheckState(Qt.CheckState.Checked)
            self.paths_list.addItem(item)
    
    def _get_selected_paths(self) -> list:
        """Get paths that are checked in the paths list."""
        selected_ids = set()
        for i in range(self.paths_list.count()):
            item = self.paths_list.item(i)
            if item.checkState() == Qt.CheckState.Checked:
                selected_ids.add(item.data(Qt.ItemDataRole.UserRole))
        
        return [p for p in self.svg_loader.paths if p.id in selected_ids]
    
    def open_svg(self):
        """Open an SVG file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open SVG File",
            "",
            "SVG Files (*.svg);;All Files (*)",
        )
        
        if file_path:
            try:
                path = Path(file_path)
                self.svg_loader.load(path)
                self.current_file = path
                
                # Clear toolpath cache
                self._roughing_toolpaths = []
                self._roughing_params = None
                self._vbit_toolpaths = []
                self._vbit_params = None
                
                # Update UI
                self.design_view.display_paths(self.svg_loader.paths)
                self._update_paths_list()
                
                self.statusbar.showMessage(
                    f"Loaded {path.name}: {len(self.svg_loader.paths)} paths"
                )
                self.setWindowTitle(f"ZCarve - {path.name}")
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load SVG: {e}")
    
    def export_roughing(self):
        """Export roughing pass G-code."""
        if not self.svg_loader.paths:
            QMessageBox.warning(self, "Warning", "No design loaded")
            return
        
        # Get only checked/closed paths
        selected_paths = self._get_selected_paths()
        paths_to_cut = [p for p in selected_paths if p.is_closed]
        
        if not paths_to_cut:
            QMessageBox.warning(
                self, "Warning", 
                "No closed paths selected. Roughing requires closed shapes."
            )
            return
        
        # Check if we have cached toolpaths
        if self._roughing_toolpaths and self._roughing_params:
            # Use cached - just need output file
            from PySide6.QtWidgets import QFileDialog
            file_path, _ = QFileDialog.getSaveFileName(
                self,
                "Save Roughing G-code",
                "roughing.nc",
                "G-code Files (*.nc *.gcode *.ngc);;All Files (*)",
            )
            if not file_path:
                return
            
            toolpaths = self._roughing_toolpaths
            params = self._roughing_params
            output_path = Path(file_path)
        else:
            # No cache - show full dialog
            dialog = RoughingDialog(self.tool_library, self)
            if dialog.exec() != RoughingDialog.DialogCode.Accepted:
                return
            
            params = dialog.get_params()
            output_path = dialog.output_path
            
            try:
                # Generate toolpaths
                self.statusbar.showMessage("Generating roughing toolpaths...")
                toolpaths = generate_roughing_toolpaths(paths_to_cut, params)
                
                if not toolpaths:
                    QMessageBox.warning(
                        self, "Warning",
                        "No toolpaths generated. The pocket may be too small for the selected tool."
                    )
                    return
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to generate toolpaths: {e}")
                return
        
        try:
            # Generate G-code with origin offset
            gcode = toolpaths_to_gcode(
                toolpaths, 
                params.tool,
                operation_name="Roughing",
                origin_offset=self.get_origin_offset()
            )
            
            # Save to file
            output_path.write_text(gcode)
            
            self.statusbar.showMessage(
                f"Exported {len(toolpaths)} toolpaths to {output_path.name}"
            )
            
            QMessageBox.information(
                self, "Export Complete",
                f"Roughing G-code saved to:\n{output_path}"
            )
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to generate G-code: {e}")
    
    def export_finishing(self):
        """Export v-bit finishing G-code."""
        if not self.svg_loader.paths:
            QMessageBox.warning(self, "Warning", "No design loaded")
            return
        
        # V-carving can work on any paths (open or closed)
        paths_to_cut = self._get_selected_paths()
        
        if not paths_to_cut:
            QMessageBox.warning(self, "Warning", "No paths selected")
            return
        
        # Check if we have cached toolpaths
        if self._vbit_toolpaths and self._vbit_params:
            # Use cached - just need output file
            from PySide6.QtWidgets import QFileDialog
            file_path, _ = QFileDialog.getSaveFileName(
                self,
                "Save V-Bit G-code",
                "vbit_finishing.nc",
                "G-code Files (*.nc *.gcode *.ngc);;All Files (*)",
            )
            if not file_path:
                return
            
            toolpaths = self._vbit_toolpaths
            params = self._vbit_params
            output_path = Path(file_path)
        else:
            # No cache - show full dialog
            dialog = VBitDialog(self.tool_library, self)
            if dialog.exec() != VBitDialog.DialogCode.Accepted:
                return
            
            params = dialog.get_params()
            output_path = dialog.output_path
            
            try:
                # Generate toolpaths
                self.statusbar.showMessage("Generating v-bit toolpaths...")
                toolpaths = generate_vbit_toolpaths(paths_to_cut, params)
                
                if not toolpaths:
                    QMessageBox.warning(
                        self, "Warning",
                        "No toolpaths generated."
                    )
                    return
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to generate toolpaths: {e}")
                return
        
        try:
            # Generate G-code with origin offset
            gcode = toolpaths_to_gcode(
                toolpaths, 
                params.tool,
                operation_name="V-Bit Finishing",
                origin_offset=self.get_origin_offset()
            )
            
            # Save to file
            output_path.write_text(gcode)
            
            self.statusbar.showMessage(
                f"Exported {len(toolpaths)} toolpaths to {output_path.name}"
            )
            
            QMessageBox.information(
                self, "Export Complete",
                f"V-bit G-code saved to:\n{output_path}"
            )
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to generate G-code: {e}")
    
    def show_tool_library(self):
        """Show tool library editor."""
        # TODO: Implement tool library editor dialog
        QMessageBox.information(
            self, "Not Implemented", "Tool library editor coming soon"
        )
    
    def preview_roughing(self):
        """Preview roughing toolpaths without exporting."""
        if not self.svg_loader.paths:
            QMessageBox.warning(self, "Warning", "No design loaded")
            return
        
        selected_paths = self._get_selected_paths()
        paths_to_cut = [p for p in selected_paths if p.is_closed]
        
        if not paths_to_cut:
            QMessageBox.warning(
                self, "Warning", 
                "No closed paths selected. Roughing requires closed shapes."
            )
            return
        
        dialog = RoughingDialog(self.tool_library, self, preview_mode=True)
        if dialog.exec() != RoughingDialog.DialogCode.Accepted:
            return
        
        params = dialog.get_params()
        
        try:
            self.statusbar.showMessage("Generating roughing preview...")
            toolpaths = generate_roughing_toolpaths(paths_to_cut, params)
            
            if not toolpaths:
                QMessageBox.warning(
                    self, "Warning",
                    "No toolpaths generated. The pocket may be too small."
                )
                return
            
            # Cache the toolpaths
            self._roughing_toolpaths = toolpaths
            self._roughing_params = params
            
            self.design_view.display_toolpaths(toolpaths, "roughing")
            self._update_preview_status()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to generate preview: {e}")
    
    def preview_finishing(self):
        """Preview v-bit toolpaths without exporting."""
        if not self.svg_loader.paths:
            QMessageBox.warning(self, "Warning", "No design loaded")
            return
        
        paths_to_cut = self._get_selected_paths()
        
        if not paths_to_cut:
            QMessageBox.warning(self, "Warning", "No paths selected")
            return
        
        # Check if we have cached roughing to use as reference
        roughing_diameter = None
        roughing_depth = None
        if self._roughing_params:
            roughing_diameter = self._roughing_params.tool.diameter
            roughing_depth = self._roughing_params.target_depth
        
        dialog = VBitDialog(
            self.tool_library, self, 
            preview_mode=True,
            auto_roughing_diameter=roughing_diameter,
            auto_roughing_depth=roughing_depth
        )
        if dialog.exec() != VBitDialog.DialogCode.Accepted:
            return
        
        params = dialog.get_params()
        
        try:
            self.statusbar.showMessage("Generating v-bit preview...")
            toolpaths = generate_vbit_toolpaths(paths_to_cut, params)
            
            if not toolpaths:
                QMessageBox.warning(self, "Warning", "No toolpaths generated.")
                return
            
            # Cache the toolpaths
            self._vbit_toolpaths = toolpaths
            self._vbit_params = params
            
            self.design_view.display_toolpaths(toolpaths, "vbit", clear_existing=False)
            self._update_preview_status()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to generate preview: {e}")
    
    def _update_preview_status(self):
        """Update status bar with current preview info."""
        parts = []
        if self._roughing_toolpaths:
            parts.append(f"{len(self._roughing_toolpaths)} roughing")
        if self._vbit_toolpaths:
            parts.append(f"{len(self._vbit_toolpaths)} v-bit")
        
        if parts:
            self.statusbar.showMessage(f"Preview: {' + '.join(parts)} toolpaths")
        else:
            self.statusbar.showMessage("Ready")
    
    def clear_preview(self):
        """Clear toolpath preview from display."""
        self.design_view.clear_toolpaths()
        self._roughing_toolpaths = []
        self._roughing_params = None
        self._vbit_toolpaths = []
        self._vbit_params = None
        self._toolpath_result = None
        self.save_gcode_btn.setEnabled(False)
        self.statusbar.showMessage("Preview cleared")
    
    def generate_toolpaths(self):
        """Generate combined roughing + V-bit toolpaths."""
        if not self.svg_loader.paths:
            QMessageBox.warning(self, "Warning", "No design loaded")
            return
        
        selected_paths = self._get_selected_paths()
        paths_to_cut = [p for p in selected_paths if p.is_closed]
        
        if not paths_to_cut:
            QMessageBox.warning(
                self, "Warning", 
                "No closed paths selected. Toolpath generation requires closed shapes."
            )
            return
        
        dialog = ToolpathDialog(self.tool_library, self)
        if dialog.exec() != ToolpathDialog.DialogCode.Accepted:
            return
        
        params = dialog.get_params()
        
        try:
            self.statusbar.showMessage("Generating toolpaths...")
            result = generate_inlay_toolpaths(paths_to_cut, params)
            
            # Store result for later export
            self._toolpath_result = result
            
            # Show preview
            self._preview_toolpath_result(result)
            
            # Enable save button
            self.save_gcode_btn.setEnabled(True)
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to generate toolpaths: {e}")
    
    def save_gcode(self):
        """Save generated toolpaths to G-code file."""
        if not hasattr(self, '_toolpath_result') or self._toolpath_result is None:
            QMessageBox.warning(self, "Warning", "No toolpaths generated. Generate toolpaths first.")
            return
        
        self._export_toolpath_result(self._toolpath_result)
    
    def _preview_toolpath_result(self, result):
        """Display toolpath preview."""
        # Cache toolpaths
        self._roughing_toolpaths = result.pocket_roughing
        self._vbit_toolpaths = result.pocket_vbit
        
        # Display all toolpaths (pocket and plug) in 3D view
        self.design_view.display_all_toolpaths(
            result.pocket_roughing,
            result.pocket_vbit,
            result.plug_roughing,
            result.plug_vbit
        )
        
        total = (len(result.pocket_roughing) + len(result.pocket_vbit) + 
                 len(result.plug_roughing) + len(result.plug_vbit))
        
        parts = []
        if result.pocket_roughing or result.pocket_vbit:
            parts.append("pocket")
        if result.plug_roughing or result.plug_vbit:
            parts.append("plug")
        
        msg = f"Preview: {total} toolpaths"
        if parts:
            msg += f" ({', '.join(parts)})"
        self.statusbar.showMessage(msg)
    
    def _export_toolpath_result(self, result):
        """Export G-code files."""
        from pathlib import Path
        
        # Get base filename from loaded SVG
        default_name = "toolpath"
        if self.svg_loader.paths and hasattr(self.svg_loader, 'file_path'):
            default_name = Path(self.svg_loader.file_path).stem
        
        # Determine if we're generating plug (inlay mode)
        has_plug = result.params.part in (InlayPart.PLUG, InlayPart.BOTH)
        
        # Ask for save location
        if has_plug:
            filename = f"{default_name}_pocket.gcode"
        else:
            filename = f"{default_name}.gcode"
        
        save_path, _ = QFileDialog.getSaveFileName(
            self, "Export G-Code", 
            str(Path.home() / filename),
            "G-Code Files (*.gcode *.nc *.ngc);;All Files (*)"
        )
        
        if not save_path:
            return
        
        save_path = Path(save_path)
        base_name = save_path.stem.replace("_pocket", "").replace("_plug", "")
        save_dir = save_path.parent
        
        origin_offset = self.get_origin_offset()
        
        # Generate G-code
        gcode = inlay_to_gcode(result, safe_z=5.0, origin_offset=origin_offset)
        
        files_saved = []
        
        # Save main/pocket file
        if result.params.part in (InlayPart.POCKET, InlayPart.BOTH):
            if 'pocket_combined' in gcode:
                if has_plug:
                    out_path = save_dir / f"{base_name}_pocket.gcode"
                else:
                    out_path = save_dir / f"{base_name}.gcode"
                out_path.write_text(gcode['pocket_combined'])
                files_saved.append(out_path.name)
        
        # Save plug file (if generating inlay)
        if has_plug and 'plug_combined' in gcode:
            plug_path = save_dir / f"{base_name}_plug.gcode"
            plug_path.write_text(gcode['plug_combined'])
            files_saved.append(plug_path.name)
        
        if files_saved:
            QMessageBox.information(
                self, "Export Complete",
                f"Saved G-code:\n• " + "\n• ".join(files_saved)
            )
            self.statusbar.showMessage(f"Exported: {', '.join(files_saved)}")
    
    def zoom_fit(self):
        """Zoom to fit all content."""
        self.design_view.fitInView(
            self.design_view.scene.sceneRect(),
            Qt.AspectRatioMode.KeepAspectRatio,
        )
    
    def zoom_reset(self):
        """Reset zoom to 100%."""
        self.design_view.resetTransform()


def main():
    """Application entry point."""
    app = QApplication(sys.argv)
    app.setApplicationName("ZCarve")
    app.setApplicationVersion("0.1.0")
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
