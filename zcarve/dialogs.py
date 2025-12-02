"""Dialog windows for ZCarve."""

from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QFormLayout,
    QGroupBox,
    QComboBox,
    QDoubleSpinBox,
    QSpinBox,
    QCheckBox,
    QPushButton,
    QFileDialog,
    QLabel,
    QDialogButtonBox,
    QMessageBox,
)

from .tools import ToolLibrary, EndMill, VBit
from .toolpaths import RoughingParams, VBitParams, ClearingStrategy


class RoughingDialog(QDialog):
    """Dialog for configuring roughing pass parameters."""
    
    def __init__(self, tool_library: ToolLibrary, parent=None, preview_mode: bool = False):
        super().__init__(parent)
        self.tool_library = tool_library
        self.selected_tool: Optional[EndMill] = None
        self.output_path: Optional[Path] = None
        self.preview_mode = preview_mode
        
        title = "Preview Roughing Pass" if preview_mode else "Export Roughing Pass"
        self.setWindowTitle(title)
        self.setMinimumWidth(400)
        
        self._setup_ui()
        self._load_tools()
    
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        
        # Tool selection
        tool_group = QGroupBox("Tool Selection")
        tool_layout = QFormLayout(tool_group)
        
        self.tool_combo = QComboBox()
        self.tool_combo.currentIndexChanged.connect(self._on_tool_changed)
        tool_layout.addRow("End Mill:", self.tool_combo)
        
        self.tool_info = QLabel()
        self.tool_info.setStyleSheet("color: gray;")
        tool_layout.addRow("", self.tool_info)
        
        layout.addWidget(tool_group)
        
        # Cut parameters
        cut_group = QGroupBox("Cut Parameters")
        cut_layout = QFormLayout(cut_group)
        
        self.depth_spin = QDoubleSpinBox()
        self.depth_spin.setRange(0.1, 50.0)
        self.depth_spin.setValue(3.0)
        self.depth_spin.setSuffix(" mm")
        self.depth_spin.setDecimals(2)
        cut_layout.addRow("Target Depth:", self.depth_spin)
        
        self.stock_spin = QDoubleSpinBox()
        self.stock_spin.setRange(0.0, 5.0)
        self.stock_spin.setValue(0.1)
        self.stock_spin.setSuffix(" mm")
        self.stock_spin.setDecimals(2)
        self.stock_spin.setToolTip("Material to leave for finishing pass")
        cut_layout.addRow("Stock to Leave:", self.stock_spin)
        
        self.climb_check = QCheckBox("Climb Milling")
        self.climb_check.setChecked(True)
        self.climb_check.setToolTip("Climb milling gives better finish but requires rigid setup")
        cut_layout.addRow("", self.climb_check)
        
        self.strategy_combo = QComboBox()
        self.strategy_combo.addItem("Adaptive HSM", ClearingStrategy.ADAPTIVE)
        self.strategy_combo.addItem("Contour", ClearingStrategy.CONTOUR)
        self.strategy_combo.addItem("Raster (Zigzag)", ClearingStrategy.RASTER)
        self.strategy_combo.setToolTip("Adaptive: constant engagement, smooth corners; Contour: follows shape; Raster: zigzag lines")
        cut_layout.addRow("Clearing Strategy:", self.strategy_combo)
        
        layout.addWidget(cut_group)
        
        # Feed/speed overrides
        override_group = QGroupBox("Feed && Speed Overrides (optional)")
        override_layout = QFormLayout(override_group)
        
        self.feed_spin = QSpinBox()
        self.feed_spin.setRange(0, 5000)
        self.feed_spin.setValue(0)
        self.feed_spin.setSuffix(" mm/min")
        self.feed_spin.setSpecialValueText("Use tool default")
        override_layout.addRow("Feed Rate:", self.feed_spin)
        
        self.plunge_spin = QSpinBox()
        self.plunge_spin.setRange(0, 2000)
        self.plunge_spin.setValue(0)
        self.plunge_spin.setSuffix(" mm/min")
        self.plunge_spin.setSpecialValueText("Use tool default")
        override_layout.addRow("Plunge Rate:", self.plunge_spin)
        
        self.rpm_spin = QSpinBox()
        self.rpm_spin.setRange(0, 30000)
        self.rpm_spin.setValue(0)
        self.rpm_spin.setSuffix(" RPM")
        self.rpm_spin.setSpecialValueText("Use tool default")
        override_layout.addRow("Spindle RPM:", self.rpm_spin)
        
        layout.addWidget(override_group)
        
        # Output file (hidden in preview mode)
        if not self.preview_mode:
            output_group = QGroupBox("Output")
            output_layout = QHBoxLayout(output_group)
            
            self.output_label = QLabel("No file selected")
            self.output_label.setStyleSheet("color: gray;")
            output_layout.addWidget(self.output_label, 1)
            
            browse_btn = QPushButton("Browse...")
            browse_btn.clicked.connect(self._browse_output)
            output_layout.addWidget(browse_btn)
            
            layout.addWidget(output_group)
        
        # Dialog buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | 
            QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
    
    def _load_tools(self):
        """Load end mills into combo box."""
        self.tool_combo.clear()
        for tool in self.tool_library.get_endmills():
            self.tool_combo.addItem(tool.name, tool.id)
        
        if self.tool_combo.count() > 0:
            self._on_tool_changed(0)
    
    def _on_tool_changed(self, index: int):
        """Handle tool selection change."""
        if index < 0:
            return
        
        tool_id = self.tool_combo.currentData()
        tool = self.tool_library.get(tool_id)
        
        if isinstance(tool, EndMill):
            self.selected_tool = tool
            self.tool_info.setText(
                f"Ø{tool.diameter}mm, {tool.stepover_percent}% stepover, "
                f"{tool.max_depth_per_pass}mm/pass"
            )
    
    def _browse_output(self):
        """Browse for output file location."""
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Roughing G-code",
            "roughing.nc",
            "G-code Files (*.nc *.gcode *.ngc);;All Files (*)",
        )
        
        if file_path:
            self.output_path = Path(file_path)
            self.output_label.setText(self.output_path.name)
            self.output_label.setStyleSheet("")
    
    def _on_accept(self):
        """Validate and accept dialog."""
        if self.selected_tool is None:
            QMessageBox.warning(self, "Warning", "Please select an end mill")
            return
        
        if not self.preview_mode and self.output_path is None:
            QMessageBox.warning(self, "Warning", "Please select an output file")
            return
        
        self.accept()
    
    def get_params(self) -> RoughingParams:
        """Get the configured roughing parameters."""
        tool = self.selected_tool
        
        # Apply overrides if specified
        if self.feed_spin.value() > 0:
            tool = EndMill(
                id=tool.id,
                name=tool.name,
                diameter=tool.diameter,
                feed_rate=float(self.feed_spin.value()),
                plunge_rate=tool.plunge_rate if self.plunge_spin.value() == 0 
                           else float(self.plunge_spin.value()),
                spindle_rpm=tool.spindle_rpm if self.rpm_spin.value() == 0 
                           else self.rpm_spin.value(),
                flute_count=tool.flute_count,
                stepover_percent=tool.stepover_percent,
                max_depth_per_pass=tool.max_depth_per_pass,
            )
        elif self.plunge_spin.value() > 0 or self.rpm_spin.value() > 0:
            tool = EndMill(
                id=tool.id,
                name=tool.name,
                diameter=tool.diameter,
                feed_rate=tool.feed_rate,
                plunge_rate=tool.plunge_rate if self.plunge_spin.value() == 0 
                           else float(self.plunge_spin.value()),
                spindle_rpm=tool.spindle_rpm if self.rpm_spin.value() == 0 
                           else self.rpm_spin.value(),
                flute_count=tool.flute_count,
                stepover_percent=tool.stepover_percent,
                max_depth_per_pass=tool.max_depth_per_pass,
            )
        
        return RoughingParams(
            tool=tool,
            target_depth=self.depth_spin.value(),
            stock_to_leave=self.stock_spin.value(),
            climb_milling=self.climb_check.isChecked(),
            strategy=self.strategy_combo.currentData(),
        )


class VBitDialog(QDialog):
    """Dialog for configuring v-bit finishing pass parameters."""
    
    def __init__(
        self, 
        tool_library: ToolLibrary, 
        parent=None, 
        preview_mode: bool = False,
        auto_roughing_diameter: float | None = None,
        auto_roughing_depth: float | None = None
    ):
        super().__init__(parent)
        self.tool_library = tool_library
        self.selected_tool: Optional[VBit] = None
        self.output_path: Optional[Path] = None
        self.preview_mode = preview_mode
        self.auto_roughing_diameter = auto_roughing_diameter
        self.auto_roughing_depth = auto_roughing_depth
        
        title = "Preview V-Bit Pass" if preview_mode else "Export V-Bit Finishing Pass"
        self.setWindowTitle(title)
        self.setMinimumWidth(400)
        
        self._setup_ui()
        self._load_tools()
    
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        
        # Tool selection
        tool_group = QGroupBox("Tool Selection")
        tool_layout = QFormLayout(tool_group)
        
        self.tool_combo = QComboBox()
        self.tool_combo.currentIndexChanged.connect(self._on_tool_changed)
        tool_layout.addRow("V-Bit:", self.tool_combo)
        
        self.tool_info = QLabel()
        self.tool_info.setStyleSheet("color: gray;")
        tool_layout.addRow("", self.tool_info)
        
        layout.addWidget(tool_group)
        
        # Cut parameters
        cut_group = QGroupBox("Cut Parameters")
        cut_layout = QFormLayout(cut_group)
        
        # Depth mode selection
        self.depth_mode_combo = QComboBox()
        self.depth_mode_combo.addItem("Target Width", "width")
        self.depth_mode_combo.addItem("Fixed Depth", "depth")
        self.depth_mode_combo.currentIndexChanged.connect(self._on_depth_mode_changed)
        cut_layout.addRow("Depth Mode:", self.depth_mode_combo)
        
        self.width_spin = QDoubleSpinBox()
        self.width_spin.setRange(0.1, 20.0)
        self.width_spin.setValue(2.0)
        self.width_spin.setSuffix(" mm")
        self.width_spin.setDecimals(2)
        self.width_spin.setToolTip("Target carved width (depth calculated from v-bit angle)")
        self.width_spin.valueChanged.connect(self._update_calculated_depth)
        cut_layout.addRow("Target Width:", self.width_spin)
        
        self.depth_spin = QDoubleSpinBox()
        self.depth_spin.setRange(0.1, 20.0)
        self.depth_spin.setValue(2.0)
        self.depth_spin.setSuffix(" mm")
        self.depth_spin.setDecimals(2)
        self.depth_spin.setEnabled(False)
        cut_layout.addRow("Fixed Depth:", self.depth_spin)
        
        self.calculated_label = QLabel()
        self.calculated_label.setStyleSheet("color: gray;")
        cut_layout.addRow("Calculated:", self.calculated_label)
        
        self.max_depth_spin = QDoubleSpinBox()
        self.max_depth_spin.setRange(0.5, 50.0)
        # Default to roughing depth if available
        default_depth = self.auto_roughing_depth if self.auto_roughing_depth else 10.0
        self.max_depth_spin.setValue(default_depth)
        self.max_depth_spin.setSuffix(" mm")
        self.max_depth_spin.setDecimals(2)
        self.max_depth_spin.setToolTip("Maximum allowed cutting depth")
        cut_layout.addRow("Max Depth:", self.max_depth_spin)
        
        self.climb_check = QCheckBox("Climb Milling")
        self.climb_check.setChecked(True)
        cut_layout.addRow("", self.climb_check)
        
        layout.addWidget(cut_group)
        
        # Roughing tool reference (for corner cleanup)
        roughing_group = QGroupBox("Roughing Reference")
        roughing_layout = QFormLayout(roughing_group)
        
        if self.auto_roughing_diameter is not None:
            # Auto-detected from cached roughing preview
            self.roughing_combo = None  # Not using combo
            self._auto_roughing = self.auto_roughing_diameter
            roughing_label = QLabel(f"Ø{self.auto_roughing_diameter:.1f}mm (from roughing preview)")
            roughing_label.setStyleSheet("font-weight: bold; color: #2a7d2a;")
            roughing_layout.addRow("Roughing Tool:", roughing_label)
            
            roughing_info = QLabel("Auto-detected from roughing preview.\nV-bit will clean corners the roughing pass couldn't reach.")
            roughing_info.setStyleSheet("color: gray; font-size: 10px;")
            roughing_layout.addRow("", roughing_info)
        else:
            # Manual selection
            self._auto_roughing = None
            self.roughing_combo = QComboBox()
            self.roughing_combo.addItem("None (full depth trace)", None)
            for tool in self.tool_library.get_endmills():
                self.roughing_combo.addItem(f"{tool.name} (Ø{tool.diameter}mm)", tool.diameter)
            roughing_layout.addRow("Roughing Tool:", self.roughing_combo)
            
            roughing_info = QLabel("Optional: Select roughing tool to clean corners.\nLeave as 'None' for full depth V-carve.")
            roughing_info.setStyleSheet("color: gray; font-size: 10px;")
            roughing_layout.addRow("", roughing_info)
        
        layout.addWidget(roughing_group)
        
        # Feed/speed overrides
        override_group = QGroupBox("Feed && Speed Overrides (optional)")
        override_layout = QFormLayout(override_group)
        
        self.feed_spin = QSpinBox()
        self.feed_spin.setRange(0, 5000)
        self.feed_spin.setValue(0)
        self.feed_spin.setSuffix(" mm/min")
        self.feed_spin.setSpecialValueText("Use tool default")
        override_layout.addRow("Feed Rate:", self.feed_spin)
        
        self.plunge_spin = QSpinBox()
        self.plunge_spin.setRange(0, 2000)
        self.plunge_spin.setValue(0)
        self.plunge_spin.setSuffix(" mm/min")
        self.plunge_spin.setSpecialValueText("Use tool default")
        override_layout.addRow("Plunge Rate:", self.plunge_spin)
        
        self.rpm_spin = QSpinBox()
        self.rpm_spin.setRange(0, 30000)
        self.rpm_spin.setValue(0)
        self.rpm_spin.setSuffix(" RPM")
        self.rpm_spin.setSpecialValueText("Use tool default")
        override_layout.addRow("Spindle RPM:", self.rpm_spin)
        
        layout.addWidget(override_group)
        
        # Output file (hidden in preview mode)
        if not self.preview_mode:
            output_group = QGroupBox("Output")
            output_layout = QHBoxLayout(output_group)
            
            self.output_label = QLabel("No file selected")
            self.output_label.setStyleSheet("color: gray;")
            output_layout.addWidget(self.output_label, 1)
            
            browse_btn = QPushButton("Browse...")
            browse_btn.clicked.connect(self._browse_output)
            output_layout.addWidget(browse_btn)
            
            layout.addWidget(output_group)
        
        # Dialog buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | 
            QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
    
    def _load_tools(self):
        """Load v-bits into combo box."""
        self.tool_combo.clear()
        for tool in self.tool_library.get_vbits():
            self.tool_combo.addItem(tool.name, tool.id)
        
        if self.tool_combo.count() > 0:
            self._on_tool_changed(0)
    
    def _on_tool_changed(self, index: int):
        """Handle tool selection change."""
        if index < 0:
            return
        
        tool_id = self.tool_combo.currentData()
        tool = self.tool_library.get(tool_id)
        
        if isinstance(tool, VBit):
            self.selected_tool = tool
            tip_info = f", {tool.tip_diameter}mm tip" if tool.tip_diameter > 0 else ""
            self.tool_info.setText(
                f"{tool.angle}° angle, Ø{tool.diameter}mm{tip_info}"
            )
            self._update_calculated_depth()
    
    def _on_depth_mode_changed(self, index: int):
        """Handle depth mode change."""
        mode = self.depth_mode_combo.currentData()
        self.width_spin.setEnabled(mode == "width")
        self.depth_spin.setEnabled(mode == "depth")
        self._update_calculated_depth()
    
    def _update_calculated_depth(self):
        """Update the calculated depth display."""
        if self.selected_tool is None:
            return
        
        mode = self.depth_mode_combo.currentData()
        if mode == "width":
            depth = self.selected_tool.depth_for_width(self.width_spin.value())
            self.calculated_label.setText(f"Depth: {depth:.3f}mm")
        else:
            width = self._width_for_depth(self.depth_spin.value())
            self.calculated_label.setText(f"Width: {width:.3f}mm")
    
    def _width_for_depth(self, depth: float) -> float:
        """Calculate carved width for a given depth."""
        import math
        if self.selected_tool is None:
            return 0.0
        half_angle = math.radians(self.selected_tool.angle / 2)
        return 2 * depth * math.tan(half_angle) + self.selected_tool.tip_diameter
    
    def _browse_output(self):
        """Browse for output file location."""
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save V-Bit G-code",
            "vbit_finishing.nc",
            "G-code Files (*.nc *.gcode *.ngc);;All Files (*)",
        )
        
        if file_path:
            self.output_path = Path(file_path)
            self.output_label.setText(self.output_path.name)
            self.output_label.setStyleSheet("")
    
    def _on_accept(self):
        """Validate and accept dialog."""
        if self.selected_tool is None:
            QMessageBox.warning(self, "Warning", "Please select a v-bit")
            return
        
        if not self.preview_mode and self.output_path is None:
            QMessageBox.warning(self, "Warning", "Please select an output file")
            return
        
        self.accept()
    
    def get_params(self) -> VBitParams:
        """Get the configured v-bit parameters."""
        tool = self.selected_tool
        
        # Apply overrides if specified
        if self.feed_spin.value() > 0 or self.plunge_spin.value() > 0 or self.rpm_spin.value() > 0:
            tool = VBit(
                id=tool.id,
                name=tool.name,
                diameter=tool.diameter,
                feed_rate=tool.feed_rate if self.feed_spin.value() == 0 
                         else float(self.feed_spin.value()),
                plunge_rate=tool.plunge_rate if self.plunge_spin.value() == 0 
                           else float(self.plunge_spin.value()),
                spindle_rpm=tool.spindle_rpm if self.rpm_spin.value() == 0 
                           else self.rpm_spin.value(),
                angle=tool.angle,
                tip_diameter=tool.tip_diameter,
            )
        
        mode = self.depth_mode_combo.currentData()
        
        # Get roughing diameter from auto-detected or combo selection
        if self._auto_roughing is not None:
            roughing_diameter = self._auto_roughing
        elif self.roughing_combo is not None:
            roughing_diameter = self.roughing_combo.currentData()
        else:
            roughing_diameter = None
        
        return VBitParams(
            tool=tool,
            flat_depth=self.depth_spin.value() if mode == "depth" else None,
            target_width=self.width_spin.value() if mode == "width" else None,
            max_depth=self.max_depth_spin.value(),
            roughing_diameter=roughing_diameter,
            climb_milling=self.climb_check.isChecked(),
        )


class ToolpathDialog(QDialog):
    """Combined dialog for roughing + V-bit toolpath generation."""
    
    def __init__(self, tool_library: ToolLibrary, parent=None):
        super().__init__(parent)
        self.tool_library = tool_library
        self.setWindowTitle("Generate Toolpaths")
        self.setMinimumWidth(450)
        
        self._setup_ui()
        self._load_tools()
    
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        
        # === Roughing Tool ===
        roughing_group = QGroupBox("Roughing Tool")
        roughing_layout = QFormLayout(roughing_group)
        
        self.roughing_combo = QComboBox()
        self.roughing_combo.currentIndexChanged.connect(self._on_roughing_changed)
        roughing_layout.addRow("End Mill:", self.roughing_combo)
        
        self.roughing_info = QLabel()
        self.roughing_info.setStyleSheet("color: gray; font-size: 11px;")
        roughing_layout.addRow("", self.roughing_info)
        
        layout.addWidget(roughing_group)
        
        # === V-Bit Tool ===
        vbit_group = QGroupBox("V-Bit Finishing Tool")
        vbit_layout = QFormLayout(vbit_group)
        
        self.vbit_combo = QComboBox()
        self.vbit_combo.currentIndexChanged.connect(self._on_vbit_changed)
        vbit_layout.addRow("V-Bit:", self.vbit_combo)
        
        self.vbit_info = QLabel()
        self.vbit_info.setStyleSheet("color: gray; font-size: 11px;")
        vbit_layout.addRow("", self.vbit_info)
        
        layout.addWidget(vbit_group)
        
        # === Cut Parameters ===
        cut_group = QGroupBox("Cut Parameters")
        cut_layout = QFormLayout(cut_group)
        
        self.depth_spin = QDoubleSpinBox()
        self.depth_spin.setRange(0.1, 50.0)
        self.depth_spin.setValue(3.0)
        self.depth_spin.setSuffix(" mm")
        self.depth_spin.setDecimals(2)
        self.depth_spin.setToolTip("Total cutting depth")
        cut_layout.addRow("Depth:", self.depth_spin)
        
        self.stock_spin = QDoubleSpinBox()
        self.stock_spin.setRange(0.0, 5.0)
        self.stock_spin.setValue(0.1)
        self.stock_spin.setSuffix(" mm")
        self.stock_spin.setDecimals(2)
        self.stock_spin.setToolTip("Material to leave for finishing pass")
        cut_layout.addRow("Stock to Leave:", self.stock_spin)
        
        self.climb_check = QCheckBox("Climb Milling")
        self.climb_check.setChecked(True)
        cut_layout.addRow("", self.climb_check)
        
        from .toolpaths import ClearingStrategy
        self.strategy_combo = QComboBox()
        self.strategy_combo.addItem("Contour", ClearingStrategy.CONTOUR)
        self.strategy_combo.addItem("Adaptive HSM", ClearingStrategy.ADAPTIVE)
        self.strategy_combo.addItem("Raster (Zigzag)", ClearingStrategy.RASTER)
        cut_layout.addRow("Clearing Strategy:", self.strategy_combo)
        
        layout.addWidget(cut_group)
        
        # === Plug Generation (for inlays) ===
        plug_group = QGroupBox("Plug Generation (for Inlays)")
        plug_group.setCheckable(True)
        plug_group.setChecked(False)
        self.plug_group = plug_group
        plug_layout = QFormLayout(plug_group)
        
        self.glue_gap_spin = QDoubleSpinBox()
        self.glue_gap_spin.setRange(0.0, 5.0)
        self.glue_gap_spin.setValue(0.5)
        self.glue_gap_spin.setSuffix(" mm")
        self.glue_gap_spin.setDecimals(2)
        self.glue_gap_spin.setToolTip("Extra depth on plug for glue space")
        plug_layout.addRow("Glue Gap:", self.glue_gap_spin)
        
        self.plug_border_spin = QDoubleSpinBox()
        self.plug_border_spin.setRange(1.0, 50.0)
        self.plug_border_spin.setValue(5.0)
        self.plug_border_spin.setSuffix(" mm")
        self.plug_border_spin.setDecimals(1)
        self.plug_border_spin.setToolTip("Border around plug to clear (material removed around design)")
        plug_layout.addRow("Clear Border:", self.plug_border_spin)
        
        self.mirror_combo = QComboBox()
        self.mirror_combo.addItem("Mirror X (Horizontal)", "x")
        self.mirror_combo.addItem("Mirror Y (Vertical)", "y")
        self.mirror_combo.setToolTip("Axis to mirror plug around")
        plug_layout.addRow("Mirror Axis:", self.mirror_combo)
        
        layout.addWidget(plug_group)
        
        # === Buttons ===
        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)
    
    def _load_tools(self):
        """Load tools into combo boxes."""
        self.roughing_combo.clear()
        for tool in self.tool_library.get_endmills():
            self.roughing_combo.addItem(tool.name, tool)
        
        self.vbit_combo.clear()
        for tool in self.tool_library.get_vbits():
            self.vbit_combo.addItem(tool.name, tool)
        
        self._on_roughing_changed()
        self._on_vbit_changed()
    
    def _on_roughing_changed(self):
        """Update roughing tool info."""
        tool = self.roughing_combo.currentData()
        if tool:
            self.roughing_info.setText(
                f"Ø{tool.diameter}mm, Feed: {tool.feed_rate}mm/min, {tool.spindle_rpm}rpm"
            )
    
    def _on_vbit_changed(self):
        """Update V-bit tool info."""
        tool = self.vbit_combo.currentData()
        if tool:
            self.vbit_info.setText(
                f"{tool.angle}° angle, Feed: {tool.feed_rate}mm/min, {tool.spindle_rpm}rpm"
            )
    
    def get_params(self):
        """Get combined parameters from dialog."""
        from .inlay import InlayParams, InlayPart
        
        generate_plug = self.plug_group.isChecked()
        
        return InlayParams(
            roughing_tool=self.roughing_combo.currentData(),
            vbit_tool=self.vbit_combo.currentData(),
            pocket_depth=self.depth_spin.value(),
            plug_depth=self.depth_spin.value() + self.glue_gap_spin.value(),
            glue_gap=self.glue_gap_spin.value(),
            stock_to_leave=self.stock_spin.value(),
            climb_milling=self.climb_check.isChecked(),
            clearing_strategy=self.strategy_combo.currentData(),
            mirror_axis=self.mirror_combo.currentData(),
            plug_border=self.plug_border_spin.value(),
            part=InlayPart.BOTH if generate_plug else InlayPart.POCKET,
        )


# Keep InlayDialog as alias for backward compatibility
InlayDialog = ToolpathDialog
