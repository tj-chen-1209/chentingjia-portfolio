# Assets Directory

This directory contains visual assets for the portfolio documentation.

## Structure

```
assets/
├── figures/          # System diagrams, architecture diagrams, result plots
│   ├── optitrack_ros2_arch.png          [TODO: Create]
│   ├── id_mapping_flow.png              [TODO: Create]
│   ├── ik_discontinuity.png             [TODO: Create]
│   ├── libero_pipeline.png              [TODO: Create]
│   └── libero_curve.png                 [TODO: Create]
└── gifs/             # Demo videos and animations
    └── optitrack_demo.gif               [TODO: Create]
```

## Creating Figures

### System Diagrams

**Recommended Tools:**
- [draw.io](https://app.diagrams.net/) - Free, export to PNG
- [Excalidraw](https://excalidraw.com/) - Hand-drawn style diagrams
- Mermaid (embedded in Markdown) - For simple flowcharts

**Requirements:**
- Resolution: 1200-1600px width for clarity
- Format: PNG with transparent background (where applicable)
- Naming: Use descriptive, lowercase names with underscores
- File size: Optimize to <500KB using [TinyPNG](https://tinypng.com/)

### Learning Curves & Plots

**Generate using Python:**

```python
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
plt.figure(figsize=(10, 6), dpi=150)
# ... plot your data ...
plt.savefig("libero_curve.png", bbox_inches='tight', dpi=150)
```

**Style Guidelines:**
- Use colorblind-friendly palettes (e.g., seaborn "colorblind")
- Label axes clearly with units
- Include legend with descriptive labels
- Export at 150 DPI minimum

### Demo GIFs

**Recording Workflow:**

1. Screen record using OBS Studio or built-in tools
2. Convert to GIF with reduced frame rate for file size:
   ```bash
   ffmpeg -i demo.mp4 -vf "fps=10,scale=800:-1:flags=lanczos" \
          -loop 0 optitrack_demo.gif
   ```
3. Optimize file size:
   ```bash
   gifsicle -O3 --colors 256 optitrack_demo.gif -o optitrack_demo_opt.gif
   ```

**Requirements:**
- Duration: 5-15 seconds (keep concise)
- Frame rate: 10 FPS (sufficient for demonstrations)
- File size: <5MB (preferably <2MB)
- Resolution: 800px width recommended

## Placeholder Status

All figure paths are currently referenced in documentation but files do not exist yet. Create them following the guidelines above and replace the placeholder references.

When figures are added, remove this README or update it to reflect completion status.
