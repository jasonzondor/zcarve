# ZCarve

G-code generator for v-carving inlays on GRBL-based CNC machines.

## Features

- **Tool Library** – Manage end mills and v-bits with feeds, speeds, and geometry
- **Roughing Pass** – Clear material with an end mill before v-carving
- **V-Bit Finishing** – Generate precise v-carve toolpaths for inlay work
- **GRBL Output** – Clean, compatible G-code for GRBL controllers

## Installation

```bash
# Install dependencies with Poetry a
poetry install
```

## Usage

```bash
poetry run zcarve
```

Or activate the virtual environment first:

```bash
poetry shell
zcarve
```

## Workflow

1. Design your inlay in Inkscape and export as SVG
2. Open the SVG in ZCarve
3. Select tools from your library (end mill for roughing, v-bit for finishing)
4. Configure cut parameters (depth, step-over, feeds/speeds)
5. Preview toolpaths
6. Export G-code files for your GRBL controller
