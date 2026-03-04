# Deployment Guide

Step-by-step instructions for deploying this portfolio to GitHub Pages.

## Prerequisites Checklist

Before deploying, ensure:

- [ ] Git repository initialized and connected to GitHub remote
- [ ] All Markdown files render correctly locally (`mkdocs serve`)
- [ ] No proprietary code, private repo names, or sensitive data included
- [ ] Contact email and placeholder URLs updated (see `PLACEHOLDERS.md`)
- [ ] GitHub repository is **public** (required for GitHub Pages on free plan)
- [ ] Repository name matches expected format: `chentingjia-portfolio`

## Deployment Methods

### Method 1: Automatic Deployment via GitHub Actions (Recommended)

GitHub Actions workflow is already configured at `.github/workflows/deploy-docs.yml`.

**One-Time Setup:**

1. **Enable GitHub Pages in repository settings:**
   ```
   Go to: Settings → Pages
   Source: Deploy from a branch
   Branch: gh-pages / (root)
   Save
   ```

2. **Push to main branch:**
   ```bash
   git add .
   git commit -m "Initial portfolio deployment"
   git push origin main
   ```

3. **Verify deployment:**
   - Go to Actions tab in GitHub repository
   - Watch "Deploy Documentation" workflow run (takes ~2-3 minutes)
   - Once complete, visit: `https://chentingjia.github.io/chentingjia-portfolio/`

**Subsequent Updates:**

Every push to `main` branch automatically triggers redeployment. No manual intervention needed.

### Method 2: Manual Deployment

If you prefer manual control or need to deploy from local machine:

```bash
# Build and deploy to gh-pages branch
mkdocs gh-deploy --force --clean

# Visit: https://chentingjia.github.io/chentingjia-portfolio/
```

**What this does:**
- Builds static site in temporary directory
- Pushes to `gh-pages` branch
- GitHub Pages automatically serves from `gh-pages`

## Verification Steps

After deployment, verify:

1. **Site loads:** Visit `https://chentingjia.github.io/chentingjia-portfolio/`
2. **Navigation works:** Click through all nav items (Home, Projects)
3. **No broken links:** Check MkDocs build output for warnings
4. **Images load:** Verify all figure references (or show as broken if placeholder)
5. **Math renders:** Check LaTeX equations display correctly
6. **Mobile responsive:** Test on mobile device or browser dev tools

## Troubleshooting

### Issue: "404 - Page Not Found" after deployment

**Causes:**
- GitHub Pages not enabled in repository settings
- Wrong branch selected (should be `gh-pages`)
- Deployment workflow failed

**Fix:**
1. Check Actions tab for workflow errors
2. Verify Settings → Pages → Source is set to `gh-pages` branch
3. Wait 2-3 minutes for DNS propagation

### Issue: "Site loads but shows generic README"

**Cause:** GitHub Pages serving from `main` branch instead of `gh-pages`

**Fix:**
1. Go to Settings → Pages
2. Change Source from "main" to "gh-pages"
3. Save and wait 1-2 minutes

### Issue: Images not loading (broken links)

**Causes:**
- Incorrect relative paths in Markdown
- Files not committed to repository
- Case-sensitive filenames on Linux servers

**Fix:**
1. Use relative paths: `../assets/figures/image.png` (not absolute)
2. Verify files exist: `ls docs/assets/figures/`
3. Check case: GitHub Pages is case-sensitive even if your local OS isn't
4. Rebuild: `mkdocs gh-deploy --force --clean`

### Issue: Math equations not rendering

**Cause:** MathJax configuration not loaded

**Fix:**
1. Verify `docs/javascripts/mathjax.js` exists
2. Check `mkdocs.yml` includes `extra_javascript` section
3. Clear browser cache and reload

### Issue: "Permission denied" during `mkdocs gh-deploy`

**Cause:** Git credentials not configured or insufficient permissions

**Fix:**
1. Configure Git credentials:
   ```bash
   git config --global user.name "Your Name"
   git config --global user.email "your.email@example.com"
   ```
2. Authenticate with GitHub:
   ```bash
   # Using GitHub CLI (recommended)
   gh auth login
   
   # Or using SSH keys
   ssh-keygen -t ed25519 -C "your.email@example.com"
   # Add key to GitHub: Settings → SSH and GPG keys
   ```

## Custom Domain (Optional)

To use a custom domain like `tingjia-chen.com`:

1. **Add CNAME file:**
   ```bash
   echo "tingjia-chen.com" > docs/CNAME
   ```

2. **Configure DNS:**
   - Add A records pointing to GitHub Pages IPs:
     - 185.199.108.153
     - 185.199.109.153
     - 185.199.110.153
     - 185.199.111.153
   - Or add CNAME record: `chentingjia.github.io`

3. **Update `mkdocs.yml`:**
   ```yaml
   site_url: https://tingjia-chen.com
   ```

4. **Enable HTTPS in GitHub:**
   - Go to Settings → Pages
   - Check "Enforce HTTPS"

## Monitoring Deployments

**GitHub Actions:**
- View all deployments: `https://github.com/chentingjia/chentingjia-portfolio/actions`
- Check workflow logs for errors

**Build Status Badge:**

Add to README.md:
```markdown
[![Deploy Docs](https://github.com/chentingjia/chentingjia-portfolio/actions/workflows/deploy-docs.yml/badge.svg)](https://github.com/chentingjia/chentingjia-portfolio/actions/workflows/deploy-docs.yml)
```

## Redeployment After Updates

**Scenario 1: Content Updates**

```bash
# Edit Markdown files
vim docs/projects/teleop-mocap.md

# Test locally
mkdocs serve

# Commit and push (triggers auto-deploy)
git add docs/
git commit -m "Update OptiTrack project metrics"
git push origin main
```

**Scenario 2: Fill Placeholders**

```bash
# Use find-and-replace to update placeholders
sed -i 's/{{MOCAP_HZ}}/120/g' docs/projects/teleop-mocap.md

# Commit and push
git add .
git commit -m "Fill performance placeholders with measured values"
git push origin main
```

**Scenario 3: Add New Figures**

```bash
# Add figure files
cp ~/diagrams/optitrack_arch.png docs/assets/figures/optitrack_ros2_arch.png

# Remove .placeholder files
rm docs/assets/figures/*.placeholder

# Commit and push
git add docs/assets/
git commit -m "Add system architecture diagrams"
git push origin main
```

## Performance Optimization

**Build Time:** ~15-30 seconds on GitHub Actions  
**Page Load Time:** Should be <2s for initial load

**Optimization Tips:**
- Compress images: Use TinyPNG or `optipng`
- Optimize GIFs: Use `gifsicle -O3`
- Enable caching: Already configured in `mkdocs.yml`
- Minimize external dependencies

## Backup & Version Control

**Recommended Practice:**

- Keep raw figure sources (`.drawio`, `.sketch`) in separate branch or external storage
- Tag releases for major portfolio updates:
  ```bash
  git tag -a v1.0 -m "Initial public release"
  git push origin v1.0
  ```
- Export PDF snapshots periodically:
  ```bash
  # Print to PDF from browser or use pandoc
  pandoc docs/projects/teleop-mocap.md -o teleop-mocap.pdf
  ```

---

**Next Steps:**

1. Verify local build: `mkdocs serve`
2. Push to GitHub: `git push origin main`
3. Monitor deployment: Check Actions tab
4. Share URL: `https://chentingjia.github.io/chentingjia-portfolio/`
