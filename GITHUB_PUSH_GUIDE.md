# GitHub Push Guide - PPE Detection Project

This guide will help you push your PPE Detection project to GitHub.

## Prerequisites

1. **GitHub Account**: Make sure you have a GitHub account at https://github.com
2. **Git Installed**: Git should be installed on your system
3. **Git Configured**: Your name and email should be configured

---

## Step 1: Configure Git (One-time setup)

If you haven't configured Git yet, run these commands:

```bash
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
```

**Verify configuration:**
```bash
git config --global user.name
git config --global user.email
```

---

## Step 2: Create a New Repository on GitHub

1. Go to https://github.com
2. Click the **"+"** icon in the top right corner
3. Select **"New repository"**
4. Fill in the details:
   - **Repository name**: `ppe-detection` (or your preferred name)
   - **Description**: "PPE Detection System using YOLOv8 for helmet and vest detection"
   - **Visibility**: Choose **Public** or **Private**
   - **DO NOT** initialize with README, .gitignore, or license (we already have these)
5. Click **"Create repository"**

> **Note**: After creating, GitHub will show you the repository URL. It will look like:
> - HTTPS: `https://github.com/your-username/ppe-detection.git`
> - SSH: `git@github.com:your-username/ppe-detection.git`

---

## Step 3: Prepare Your Local Repository

Open PowerShell/Terminal in your project directory:

```bash
cd C:\Users\hashi\Downloads\PPE2\PPE2\PPE-Detection
```

### Check Git Status

```bash
git status
```

### Add Remote Repository

Replace `YOUR_USERNAME` and `REPO_NAME` with your GitHub username and repository name:

```bash
git remote add origin https://github.com/YOUR_USERNAME/REPO_NAME.git
```

**Example:**
```bash
git remote add origin https://github.com/johndoe/ppe-detection.git
```

### Verify Remote

```bash
git remote -v
```

---

## Step 4: Stage Your Files

### Add All Files

```bash
git add .
```

### Check What Will Be Committed

```bash
git status
```

**Important Files to Check:**
- ✅ Code files (.py)
- ✅ Configuration (config.yaml)
- ✅ Documentation (README.md)
- ✅ Requirements (requirements.txt)
- ❌ Model files (.pt) - Should be excluded by .gitignore
- ❌ Video files (.mp4) - Should be excluded by .gitignore
- ❌ Large output files - Should be excluded

---

## Step 5: Commit Your Changes

```bash
git commit -m "Phase 1: System stabilization and POC delivery complete"
```

**Or use a more detailed commit message:**

```bash
git commit -m "Phase 1: PPE Detection System - Complete

- Removed legacy files and cleaned up codebase
- Added GPU utilities with CUDA detection
- Implemented health check endpoints
- Created comprehensive test infrastructure
- Added detection validation and benchmark scripts
- Improved tracking for ID persistence
- Updated documentation with setup and usage guides
- Configured for custom PPE model (helmet + vest detection)
"
```

---

## Step 6: Push to GitHub

### First Push (Set Upstream)

```bash
git push -u origin master
```

**Or if your default branch is 'main':**

```bash
git push -u origin main
```

> **Note**: You may be prompted for your GitHub username and password/token.
> - For HTTPS, you'll need a **Personal Access Token** (not password)
> - Generate token at: https://github.com/settings/tokens

### Subsequent Pushes (After First Push)

```bash
git push
```

---

## Step 7: Verify on GitHub

1. Go to your repository on GitHub: `https://github.com/YOUR_USERNAME/REPO_NAME`
2. Check that all files are uploaded
3. Verify README.md is displayed on the main page

---

## Common Issues & Solutions

### Issue 1: "fatal: remote origin already exists"

**Solution:** Remove the existing remote and add again:
```bash
git remote remove origin
git remote add origin https://github.com/YOUR_USERNAME/REPO_NAME.git
```

### Issue 2: Authentication Failed (HTTPS)

**Solution:** Use a Personal Access Token instead of password:
1. Go to: https://github.com/settings/tokens
2. Click "Generate new token (classic)"
3. Select scopes: `repo` (full control)
4. Copy the token
5. Use the token as your password when pushing

### Issue 3: Large Files Rejected

**Solution:** Files over 100MB are rejected by GitHub.
```bash
# Check file sizes
git ls-files -s | awk '{if ($4 > 100000000) print $4, $2}'

# Remove large files from git
git rm --cached large_file.pt
git commit -m "Remove large files"
```

### Issue 4: Wrong Branch Name

**Solution:** Rename your branch to 'main':
```bash
git branch -M main
git push -u origin main
```

---

## Adding Model Files (Alternative Approach)

Since `.pt` model files are large (5-6 MB), you have options:

### Option 1: Use Git LFS (Large File Storage)
```bash
# Install Git LFS first
git lfs install

# Track .pt files
git lfs track "*.pt"

# Add and commit
git add .gitattributes
git add best.pt
git commit -m "Add model with Git LFS"
git push
```

### Option 2: Link to External Storage
Add a note in README.md:
```markdown
## Model Files

Download the trained model:
- [best.pt](https://drive.google.com/...) - PPE Detection Model (5.96 MB)
```

### Option 3: Exclude from Git (Current Setup)
Model files are in `.gitignore` and won't be pushed. Users download separately.

---

## Quick Reference

### Daily Workflow

```bash
# 1. Check status
git status

# 2. Add changes
git add .

# 3. Commit
git commit -m "Your descriptive message"

# 4. Push
git push
```

### Useful Commands

```bash
# View commit history
git log --oneline

# View remote URL
git remote -v

# Create a new branch
git checkout -b feature-name

# Switch branches
git checkout main

# View differences
git diff

# Undo last commit (keep changes)
git reset --soft HEAD~1
```

---

## Next Steps After Pushing

1. **Add a License**: Go to your repo > Add file > Create new file > Name it `LICENSE`
2. **Add Topics**: Repository Settings > Topics > Add: `yolov8`, `ppe-detection`, `computer-vision`, `object-detection`
3. **Enable Issues**: Settings > Features > Issues (for bug tracking)
4. **Add Collaborators**: Settings > Collaborators (if working in a team)
5. **Create Releases**: Releases > Create a new release (for versioning)

---

## Repository Best Practices

✅ **DO:**
- Write clear commit messages
- Update README.md regularly
- Keep .gitignore updated
- Commit frequently with small, logical changes
- Test before pushing
- Use branches for new features

❌ **DON'T:**
- Commit sensitive data (API keys, passwords)
- Push large binary files (>50MB)
- Use vague commit messages ("fixed stuff")
- Commit directly to main/master without testing
- Include auto-generated files

---

## Support

If you encounter issues:
1. Check GitHub documentation: https://docs.github.com
2. Search Stack Overflow
3. Ask in GitHub Community: https://github.community

---

**Your project is ready to push! Follow the steps above and your PPE Detection system will be on GitHub. 🚀**
