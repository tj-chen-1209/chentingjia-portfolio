# Contributing to Portfolio Documentation

This guide is for maintaining and updating your portfolio as you complete new projects or collect experimental data.

## Quick Reference

**Add new project:**
1. Create `docs/projects/new-project.md`
2. Add Chinese TL;DR at top
3. Update `nav` in `mkdocs.yml`
4. Follow structure: Context → System → Challenges → Contributions → Results → Reproducibility

**Update placeholders:**
1. Find placeholder: `grep -r "{{VARIABLE}}" docs/`
2. Replace with measured value
3. Commit and push (auto-deploys)

**Add figures:**
1. Create diagram (draw.io, Matplotlib)
2. Save to `docs/assets/figures/`
3. Remove `.placeholder` file
4. Reference in Markdown: `![Alt](../assets/figures/name.png)`

## Documentation Standards

### Writing Style

**DO:**
- ✅ Use precise, technical language
- ✅ Include concrete numbers and measurements
- ✅ Report failures and limitations honestly
- ✅ Emphasize reproducibility (configs, versions, seeds)
- ✅ Use placeholders for unknown values: `{{VARIABLE}}`
- ✅ Add Chinese TL;DR (3-5 lines) at top of project pages

**DON'T:**
- ❌ Use marketing language or superlatives
- ❌ Fabricate numbers or exaggerate results
- ❌ Include proprietary code or internal details
- ❌ Make unverifiable claims
- ❌ Use emojis (except in grid cards if already present)

### Project Page Structure

Every project page should follow this template:

```markdown
# Project Title

**中文简介：** [3-5 lines summarizing project in Chinese]

---

**Organization:** [Company/Lab]  
**Duration:** [Year]  
**Role:** [Your role]  
**Stack:** [Key technologies]

## Context & Goal
[Why this project matters, what problem it solves]

## System Overview
[Architecture diagram + component descriptions]

## Key Challenges
[Technical problems encountered, root causes, solutions]

## My Contributions
[Verifiable engineering deliverables you built]

## Results
[Quantitative metrics, tables, figures]

## Failure Modes & Fixes
[What didn't work, why, and how you addressed it]

## Reproducibility Notes
[Software versions, hardware requirements, configuration steps]

## Next Steps
[Future improvements, open problems]

---

**References:** [Links to papers, docs, tools]
```

### Code Snippet Guidelines

**When to include code:**
- Illustrate a specific technical solution
- Show API usage or configuration
- Demonstrate performance-critical implementation

**How to format:**

```cpp
// Good: Concise, focused snippet with context
static void DataHandler(sFrameOfMocapData* data, void* pUserData) {
    MotiveStreamer* streamer = static_cast<MotiveStreamer*>(pUserData);
    std::lock_guard<std::mutex> lock(streamer->data_mutex_);
    streamer->latest_frame_ = *data;
}
```

**Avoid:**
- Full file dumps (link to private repo or describe architecture instead)
- Proprietary algorithms
- Code with company/lab-specific names or paths

## Figure Creation Standards

### System Architecture Diagrams

**Required Elements:**
- Component boxes with clear labels
- Data flow arrows with protocol/format labels
- Network connections with ports/addresses
- Color coding: Blue (data flow), Orange (processing), Green (output)

**Tools:**
- draw.io (free, exports clean PNGs)
- Excalidraw (hand-drawn style)
- Lucidchart (professional templates)

**Specifications:**
- Resolution: 1200-1600px width
- Format: PNG with transparent background
- File size: <500KB (use TinyPNG to compress)
- Style: Clean, professional, colorblind-friendly

### Learning Curves & Plots

**Required Elements:**
- Axes labeled with units
- Legend with descriptive labels
- Grid for readability
- Confidence intervals (±1 std) if multi-seed
- Baseline comparisons (e.g., pretrained 0-shot)

**Python Template:**

```python
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
sns.set_palette("colorblind")

fig, ax = plt.subplots(figsize=(10, 6), dpi=150)

# Plot your data with labels
ax.plot(x, y, label='Method Name', linewidth=2)
ax.fill_between(x, y_lower, y_upper, alpha=0.3)

ax.set_xlabel('X Label (units)', fontsize=12)
ax.set_ylabel('Y Label (units)', fontsize=12)
ax.set_title('Plot Title', fontsize=14, fontweight='bold')
ax.legend(fontsize=10, loc='best')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('output.png', dpi=150, bbox_inches='tight')
```

**Specifications:**
- Figure size: 10×6 inches at 150 DPI (1500×900px)
- Format: PNG
- Style: Seaborn "whitegrid" with colorblind palette
- Font: 12pt for labels, 10pt for legend

## Adding New Projects

### Step 1: Create Project File

```bash
# Create new project Markdown file
touch docs/projects/new-project.md

# Copy template structure from existing project
head -n 20 docs/projects/teleop-mocap.md > docs/projects/new-project.md
```

### Step 2: Write Content

Follow the standard structure (see template above). Key principles:

- Start with Chinese TL;DR (3-5 lines)
- Use concrete, verifiable statements
- Include placeholders for unknown values
- Add "Placeholder Checklist" table at end

### Step 3: Update Navigation

Edit `mkdocs.yml`:

```yaml
nav:
  - Home: index.md
  - Projects:
      - OptiTrack ROS2 Streaming Node: projects/teleop-mocap.md
      - GR00T-N1.6 SFT on LIBERO: projects/groot-libero-reproduction.md
      - RDT SFT & Evaluation on LIBERO: projects/rdt-libero.md
      - RLinf × RDT Integration: projects/rlinf-rdt-integration.md
      - New Project Title: projects/new-project.md  # Add here
```

### Step 4: Update Index Page

Edit `docs/index.md` to add new project card:

```markdown
-   :material-icon: **Project Title**

    ---

    Brief description (1-2 sentences). Key technology stack.

    **Stack:** Technologies used

    [:octicons-arrow-right-24: Technical Details](projects/new-project.md)
```

### Step 5: Test Locally

```bash
mkdocs serve
# Open http://127.0.0.1:8000
# Verify new project appears in navigation and renders correctly
```

### Step 6: Commit and Deploy

```bash
git add docs/projects/new-project.md docs/index.md mkdocs.yml
git commit -m "Add new project: [Project Title]"
git push origin main
# GitHub Actions auto-deploys
```

## Updating Existing Projects

### Adding Experimental Results

When you have new measurements:

1. **Locate placeholders:**
   ```bash
   grep "{{" docs/projects/teleop-mocap.md
   ```

2. **Replace with actual values:**
   ```bash
   # Use your editor's find-and-replace
   # Or sed:
   sed -i 's/{{MOCAP_HZ}}/120/g' docs/projects/teleop-mocap.md
   ```

3. **Remove from placeholder checklist:**
   - Delete the row from "Placeholder Checklist" table
   - Or add checkmark: ✅ Filled

4. **Commit with descriptive message:**
   ```bash
   git add docs/projects/teleop-mocap.md
   git commit -m "Add measured latency and throughput metrics for OptiTrack node"
   git push origin main
   ```

### Adding Figures

1. **Create figure** following standards above
2. **Save to assets directory:**
   ```bash
   cp ~/diagrams/new_figure.png docs/assets/figures/
   ```

3. **Remove placeholder file:**
   ```bash
   rm docs/assets/figures/new_figure.png.placeholder
   ```

4. **Verify reference in Markdown:**
   ```markdown
   ![Alt text](../assets/figures/new_figure.png)
   ```

5. **Commit:**
   ```bash
   git add docs/assets/figures/new_figure.png
   git rm docs/assets/figures/new_figure.png.placeholder
   git commit -m "Add system architecture diagram for OptiTrack project"
   git push origin main
   ```

## Quality Checklist Before Deployment

Use this checklist before major portfolio updates:

### Content Quality

- [ ] All project pages have Chinese TL;DR at top
- [ ] No fabricated numbers (use placeholders if unknown)
- [ ] No proprietary code or internal details
- [ ] All claims are verifiable or clearly marked as placeholders
- [ ] Failure modes and limitations honestly documented
- [ ] References and links are valid

### Technical Accuracy

- [ ] Software versions specified (or placeholders used)
- [ ] Hardware requirements clearly stated
- [ ] Configuration examples are accurate
- [ ] Code snippets compile/run (if provided)
- [ ] Performance metrics include measurement methodology

### Formatting

- [ ] All Markdown renders correctly (`mkdocs serve`)
- [ ] No broken internal links
- [ ] Images have alt text
- [ ] Tables are well-formatted
- [ ] Code blocks have language tags
- [ ] Math equations use correct delimiters (`\(` `\)`)

### Reproducibility

- [ ] Dependencies listed with versions (or version ranges)
- [ ] Configuration steps are complete
- [ ] Measurement protocols described
- [ ] Random seeds documented
- [ ] Hardware requirements specified

### Visual Assets

- [ ] All referenced figures exist (or have `.placeholder` files)
- [ ] Images are optimized (<500KB for PNGs)
- [ ] GIFs are optimized (<5MB, preferably <2MB)
- [ ] Figures have descriptive captions

## Maintenance Schedule

**Quarterly (Every 3 months):**
- Review and update project results with new data
- Fill placeholders as measurements become available
- Add new projects completed in the quarter
- Update software versions in reproducibility sections

**Before PhD Applications:**
- Ensure all high-priority placeholders filled
- Add any missing figures or demos
- Proofread all content for accuracy
- Verify all links work (PDF, demos, external references)

**Before Sharing with PIs:**
- Double-check no proprietary information included
- Verify all performance metrics are measured (not estimated)
- Ensure reproducibility sections are complete
- Test deployment on GitHub Pages

## Getting Help

**Build Issues:**
- Check MkDocs documentation: https://www.mkdocs.org/
- Material theme docs: https://squidfunk.github.io/mkdocs-material/

**Content Questions:**
- Review other project pages in `docs/projects/` for examples
- Check `PLACEHOLDERS.md` for measurement instructions
- See `ZHIHU_ADAPTATION.md` for writing guidance

**Deployment Issues:**
- See `DEPLOYMENT.md` for troubleshooting
- Check GitHub Actions logs for errors

---

**Remember:** Quality over quantity. Better to have 2-3 deeply documented, verifiable projects than 10 shallow summaries.
