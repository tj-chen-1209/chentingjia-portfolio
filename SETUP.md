# Portfolio Setup Guide

This guide explains how to set up the portfolio documentation locally and deploy to GitHub Pages.

## Prerequisites

- Python 3.9 or higher
- Git
- GitHub account with Pages enabled

## Local Development

### 1. Clone Repository

```bash
git clone https://github.com/chentingjia/chentingjia-portfolio.git
cd chentingjia-portfolio
```

### 2. Install Dependencies

```bash
# Using pip
pip install -r requirements.txt

# Or using pip with user flag
pip install --user -r requirements.txt
```

### 3. Serve Locally

```bash
mkdocs serve
```

The documentation will be available at `http://127.0.0.1:8000/`

**Live Reload:** MkDocs automatically reloads when you edit files.

### 4. Build Static Site

```bash
mkdocs build
```

Output will be generated in `site/` directory.

## Adding Content

### Project Pages

1. Create new Markdown file in `docs/projects/`
2. Add Chinese TL;DR at top (3-5 lines)
3. Follow structure: Context → System → Challenges → Contributions → Results → Reproducibility
4. Use placeholders `{{VARIABLE_NAME}}` for unknown values
5. Update `nav` section in `mkdocs.yml`

### Figures & Assets

1. Create diagrams using draw.io, Excalidraw, or Mermaid
2. Export to PNG (1200-1600px width, <500KB file size)
3. Save to `docs/assets/figures/`
4. Reference in Markdown: `![Alt text](../assets/figures/filename.png)`

### Demo GIFs

1. Screen record demo (OBS Studio, QuickTime, etc.)
2. Convert to optimized GIF:
   ```bash
   ffmpeg -i demo.mp4 -vf "fps=10,scale=800:-1:flags=lanczos" demo.gif
   gifsicle -O3 --colors 256 demo.gif -o demo_optimized.gif
   ```
3. Save to `docs/assets/gifs/`

## Deployment to GitHub Pages

### Automatic Deployment (Recommended)

Create `.github/workflows/deploy-docs.yml`:

```yaml
name: Deploy Documentation

on:
  push:
    branches: [main]

permissions:
  contents: write

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: 3.x
      - run: pip install -r requirements.txt
      - run: mkdocs gh-deploy --force
```

**First-time Setup:**

1. Go to repository Settings → Pages
2. Set Source to "gh-pages branch"
3. Push to main branch → workflow deploys automatically

### Manual Deployment

```bash
mkdocs gh-deploy
```

This builds the site and pushes to `gh-pages` branch.

## Updating Placeholders

Placeholders use `{{VARIABLE_NAME}}` format. To fill them:

1. Locate placeholder in Markdown files (search for `{{`)
2. Run experiments/measurements to obtain actual values
3. Replace placeholder with measured value
4. Commit changes

**Placeholder Tracking:**  
Each project page includes a "Placeholder Checklist" table listing all placeholders and measurement instructions.

## Validation Checklist

Before deploying, verify:

- [ ] All Markdown files render correctly (`mkdocs serve`)
- [ ] No broken internal links (check MkDocs build warnings)
- [ ] No proprietary code, private repo names, or sensitive data included
- [ ] Figures referenced in Markdown exist in `docs/assets/` (or placeholders documented)
- [ ] Chinese TL;DR present at top of each project page
- [ ] Placeholder tables included for unknown values
- [ ] Links in README.md updated (PDF, demo URLs)

## Troubleshooting

**Issue:** `ModuleNotFoundError: No module named 'mkdocs'`  
**Fix:** Run `pip install -r requirements.txt`

**Issue:** `Config value 'plugins': The "glightbox" plugin is not installed`  
**Fix:** Run `pip install mkdocs-glightbox`

**Issue:** Math equations not rendering  
**Fix:** Verify `mathjax.js` is in `docs/javascripts/` and referenced in `mkdocs.yml`

**Issue:** Images not loading in deployed site  
**Fix:** Use relative paths (`../assets/figures/`) not absolute paths

## Additional Resources

- [MkDocs Documentation](https://www.mkdocs.org/)
- [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/)
- [GitHub Pages Documentation](https://docs.github.com/en/pages)
