"""3D preview dialog for visualizing toolpaths as wireframes."""

import numpy as np
from typing import Optional

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QCheckBox, 
    QPushButton, QLabel, QGroupBox, QSplitter, QWidget
)
from PySide6.QtCore import Qt

import matplotlib
matplotlib.use('QtAgg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from .toolpaths import Toolpath


class ToolpathPreviewDialog(QDialog):
    """Dialog showing 3D wireframe view of toolpaths (pocket and plug side by side)."""
    
    def __init__(
        self,
        pocket_roughing: list[Toolpath] = None,
        pocket_vbit: list[Toolpath] = None,
        plug_roughing: list[Toolpath] = None,
        plug_vbit: list[Toolpath] = None,
        parent=None
    ):
        super().__init__(parent)
        self.setWindowTitle("3D Toolpath Preview")
        self.setMinimumSize(1000, 500)
        self.resize(1200, 600)
        
        self.pocket_roughing = pocket_roughing or []
        self.pocket_vbit = pocket_vbit or []
        self.plug_roughing = plug_roughing or []
        self.plug_vbit = plug_vbit or []
        
        self._setup_ui()
        self._update_plots()
    
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        
        # Check if we have plug data
        has_plug = bool(self.plug_roughing or self.plug_vbit)
        
        if has_plug:
            # Side by side splitter
            splitter = QSplitter(Qt.Orientation.Horizontal)
            
            # Pocket view
            pocket_widget = self._create_view_widget("Pocket", is_pocket=True)
            splitter.addWidget(pocket_widget)
            
            # Plug view
            plug_widget = self._create_view_widget("Plug", is_pocket=False)
            splitter.addWidget(plug_widget)
            
            splitter.setSizes([600, 600])
            layout.addWidget(splitter)
        else:
            # Single pocket view
            pocket_widget = self._create_view_widget("Toolpaths", is_pocket=True)
            layout.addWidget(pocket_widget)
        
        # Controls
        controls = QHBoxLayout()
        
        # Show options
        show_group = QGroupBox("Show")
        show_layout = QHBoxLayout(show_group)
        
        self.show_roughing = QCheckBox("Roughing")
        self.show_roughing.setChecked(True)
        self.show_roughing.stateChanged.connect(self._update_plots)
        show_layout.addWidget(self.show_roughing)
        
        self.show_vbit = QCheckBox("V-Bit")
        self.show_vbit.setChecked(True)
        self.show_vbit.stateChanged.connect(self._update_plots)
        show_layout.addWidget(self.show_vbit)
        
        controls.addWidget(show_group)
        controls.addStretch()
        
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        controls.addWidget(close_btn)
        
        layout.addLayout(controls)
    
    def _create_view_widget(self, title: str, is_pocket: bool) -> QWidget:
        """Create a 3D view widget."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(2, 2, 2, 2)
        
        label = QLabel(f"<b>{title}</b>")
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(label)
        
        figure = Figure(figsize=(5, 4))
        canvas = FigureCanvas(figure)
        layout.addWidget(canvas)
        
        if is_pocket:
            self.pocket_figure = figure
            self.pocket_canvas = canvas
        else:
            self.plug_figure = figure
            self.plug_canvas = canvas
        
        return widget
    
    def _update_plots(self):
        """Update all 3D plots."""
        self._plot_toolpaths(
            self.pocket_figure, 
            self.pocket_canvas,
            self.pocket_roughing if self.show_roughing.isChecked() else [],
            self.pocket_vbit if self.show_vbit.isChecked() else [],
            "Pocket"
        )
        
        if hasattr(self, 'plug_figure'):
            self._plot_toolpaths(
                self.plug_figure,
                self.plug_canvas,
                self.plug_roughing if self.show_roughing.isChecked() else [],
                self.plug_vbit if self.show_vbit.isChecked() else [],
                "Plug"
            )
    
    def _plot_toolpaths(
        self, 
        figure: Figure, 
        canvas: FigureCanvas,
        roughing: list[Toolpath],
        vbit: list[Toolpath],
        title: str
    ):
        """Plot toolpaths as 3D wireframes."""
        figure.clear()
        ax = figure.add_subplot(111, projection='3d')
        
        all_x, all_y, all_z = [], [], []
        
        # Plot roughing paths (blue)
        for tp in roughing:
            pts = tp.points
            n = len(pts)
            if n < 2:
                continue
            
            # Get Z values
            if pts.shape[1] >= 3:
                z_vals = pts[:, 2]
            else:
                z_vals = np.full(n, tp.z_depth)
            
            ax.plot(pts[:, 0], pts[:, 1], z_vals, 
                   color='blue', linewidth=0.8, alpha=0.7)
            
            all_x.extend(pts[:, 0])
            all_y.extend(pts[:, 1])
            all_z.extend(z_vals)
        
        # Plot v-bit paths (red)
        for tp in vbit:
            pts = tp.points
            n = len(pts)
            if n < 2:
                continue
            
            # Get Z values
            if pts.shape[1] >= 3:
                z_vals = pts[:, 2]
            else:
                z_vals = np.full(n, tp.z_depth)
            
            ax.plot(pts[:, 0], pts[:, 1], z_vals,
                   color='red', linewidth=0.8, alpha=0.7)
            
            all_x.extend(pts[:, 0])
            all_y.extend(pts[:, 1])
            all_z.extend(z_vals)
        
        # Set axis labels
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_zlabel('Z (mm)')
        
        # Set equal aspect and limits
        if all_x:
            x_range = max(all_x) - min(all_x)
            y_range = max(all_y) - min(all_y)
            z_range = max(all_z) - min(all_z) if max(all_z) != min(all_z) else 1
            
            max_range = max(x_range, y_range, z_range)
            
            ax.set_box_aspect([
                x_range / max_range if max_range > 0 else 1,
                y_range / max_range if max_range > 0 else 1,
                z_range / max_range if max_range > 0 else 0.3
            ])
        
        # Title with counts
        parts = []
        if roughing:
            parts.append(f"Roughing: {len(roughing)}")
        if vbit:
            parts.append(f"V-Bit: {len(vbit)}")
        
        ax.set_title(f"{title}\n{', '.join(parts)}" if parts else title)
        
        canvas.draw()


# Keep old class name for backward compatibility
class Preview3DDialog(ToolpathPreviewDialog):
    """Legacy wrapper for old interface."""
    
    def __init__(
        self, 
        stock_dims: tuple[float, float, float],
        roughing_toolpaths: list[Toolpath],
        vbit_toolpaths: list[Toolpath],
        roughing_diameter: float = 3.0,
        vbit_angle: float = 60.0,
        parent=None
    ):
        super().__init__(
            pocket_roughing=roughing_toolpaths,
            pocket_vbit=vbit_toolpaths,
            parent=parent
        )
