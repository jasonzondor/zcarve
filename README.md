# ZCarve

G-code generator for v-carving inlays on GRBL-based CNC machines.

## Features

- **V-Carve Inlay Generation** – Complete pocket and plug toolpath generation for inlays
- **Tool Library** – SQLite-based tool database with full CRUD support for end mills and v-bits
- **3D Toolpath Preview** – Interactive OpenGL visualization with rotation, pan, and zoom
- **Dual Preview Tabs** – Side-by-side pocket and plug preview with stock boundary display
- **Roughing Pass** – Contour clearing with configurable stepover and depth passes
- **V-Bit Finishing** – Progressive depth passes with proper V-profile geometry
- **Plug Generation** – Tapered plug profiles that match pocket V-grooves
- **GRBL Output** – Clean G-code with tool change pauses

## Installation

```bash
# Install dependencies with Poetry
poetry install
```

## Usage

```bash
poetry run zcarve
```

## Workflow

1. **Design** – Create your inlay design in Inkscape and export as SVG
2. **Open** – Load the SVG in ZCarve (File → Open SVG)
3. **Configure Tools** – Set up your end mill and v-bit in the Tool Library (Tools → Tool Library)
4. **Generate** – Click "Generate Toolpaths" and configure:
   - Roughing tool and V-bit selection
   - Cut depth and glue gap
   - Enable "Generate Plug" for inlay work
5. **Preview** – Inspect toolpaths in the 3D view (rotate, pan, zoom)
6. **Export** – Save G-code files for pocket and plug

## Controls

- **Left-click + drag** – Rotate 3D view
- **Right-click + drag** – Pan view
- **Scroll wheel** – Zoom

## Tool Library

Tools are stored in `~/.config/zcarve/tools.db` (SQLite).

### End Mill Properties
- Diameter, feed rate, plunge rate, spindle RPM
- Flute count, stepover %, max depth per pass

### V-Bit Properties
- Diameter, feed rate, plunge rate, spindle RPM
- Included angle, tip diameter

## Requirements

- Python 3.10+
- PySide6
- pyqtgraph + PyOpenGL
- Shapely
- svgpathtools
- numpy
