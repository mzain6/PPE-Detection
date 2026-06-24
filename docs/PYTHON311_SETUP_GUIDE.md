# Python 3.11 Installation & face-recognition Setup Guide

## Step 1: Download Python 3.11

**Manual Download:**
1. Visit: https://www.python.org/downloads/release/python-31110/
2. Scroll down to "Files"
3. Click: **Windows installer (64-bit)**
4. Download: `python-3.11.10-amd64.exe`

**Or use this direct link:**
```
https://www.python.org/ftp/python/3.11.10/python-3.11.10-amd64.exe
```

---

## Step 2: Install Python 3.11

1. **Run the downloaded installer** (`python-3.11.10-amd64.exe`)

2. **IMPORTANT:** On the first screen:
   - ✅ Check "Add python.exe to PATH"
   - ✅ Check "Install for all users" (optional)
   - Click "Install Now"

3. Wait for installation to complete (~5 minutes)

4. Click "Close"

---

## Step 3: Verify Python 3.11 Installation

Open a **NEW PowerShell window** (important!) and run:

```powershell
python --version
```

**Expected output:** `Python 3.11.10`

If you see `Python 3.13.7`, the old version is still default. Try:

```powershell
py -3.11 --version
```

This should show Python 3.11.

---

## Step 4: Navigate to Project Directory

```powershell
cd c:\Users\ST\Desktop\PPE_Phas2_05Feb\Phase_Detection_05Feb\PPE-Detection-2
```

---

## Step 5: Create Virtual Environment with Python 3.11

```powershell
# Create virtual environment
py -3.11 -m venv venv_face

# Activate it
.\venv_face\Scripts\Activate.ps1
```

**Note:** If you get a security error, run:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Then try activating again.

---

## Step 6: Install face-recognition (EASY NOW!)

**Once virtual environment is activated**, you'll see `(venv_face)` in your prompt:

```powershell
# Upgrade pip first
python -m pip install --upgrade pip

# Install dlib pre-built wheel for Python 3.11
pip install https://github.com/z-mahmud22/Dlib_Windows_Python3.x/raw/main/dlib-19.24.1-cp311-cp311-win_amd64.whl

# Install face-recognition
pip install face-recognition

# Install other project dependencies
pip install ultralytics opencv-python numpy pyyaml fastapi uvicorn torch torchvision
```

---

## Step 7: Verify Installation

```powershell
python -c "import face_recognition; print('✅ face-recognition installed successfully!')"
```

**Expected output:** `✅ face-recognition installed successfully!`

---

## Step 8: Run Face Detection Test

```powershell
python test_face_detection.py
```

**What to expect:**
- Webcam opens
- Green boxes around persons with faces
- Person IDs shown
- **IDs persist when you leave and return!** ✅

---

## 🚀 Quick Commands (After Python 3.11 Installed)

```powershell
# 1. Navigate to project
cd c:\Users\ST\Desktop\PPE_Phas2_05Feb\Phase_Detection_05Feb\PPE-Detection-2

# 2. Create & activate virtual environment
py -3.11 -m venv venv_face
.\venv_face\Scripts\Activate.ps1

# 3. Install everything
python -m pip install --upgrade pip
pip install https://github.com/z-mahmud22/Dlib_Windows_Python3.x/raw/main/dlib-19.24.1-cp311-cp311-win_amd64.whl
pip install face-recognition
pip install -r requirements.txt

# 4. Test it
python test_face_detection.py
```

---

## 📋 Future Use

**Every time you want to use face detection:**

```powershell
# 1. Navigate to project
cd c:\Users\ST\Desktop\PPE_Phas2_05Feb\Phase_Detection_05Feb\PPE-Detection-2

# 2. Activate virtual environment
.\venv_face\Scripts\Activate.ps1

# 3. Run your scripts
python test_face_detection.py
# or
python multi_camera_ppe_demo.py
```

---

## ⚠️ Troubleshooting

**"py -3.11: command not found"**
- Python 3.11 not installed or not in PATH
- Reinstall Python 3.11 with "Add to PATH" checked

**"cannot be loaded because running scripts is disabled"**
- Run: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`

**"No module named 'face_recognition'"**
- Make sure virtual environment is activated (you should see `(venv_face)` in prompt)

**Virtual environment activation doesn't work**
- Try: `.\\venv_face\\Scripts\\activate.bat` instead

---

## ✅ Summary

1. Download Python 3.11.10 installer
2. Install with "Add to PATH" checked
3. Create virtual environment: `py -3.11 -m venv venv_face`
4. Activate: `.\venv_face\Scripts\Activate.ps1`
5. Install face-recognition with pre-built wheel
6. Test with `python test_face_detection.py`

**Total time:** ~15 minutes
