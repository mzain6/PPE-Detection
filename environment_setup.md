# Environment Setup Guide (NVIDIA T500)

This guide is tailored for deploying the PPE Detection system on a machine with an **NVIDIA T500** GPU (Turing Architecture).

## 1. Prerequisite: NVIDIA Drivers
Ensure you have the latest NVIDIA Studio or Data Center drivers installed.
- **Minimum Version**: Driver 531.x or later (supports CUDA 12.1).
- **Check**: Run `nvidia-smi` in PowerShell. You should see your T500 listed.

## 2. Python Environment with Anaconda
We recommend using Conda to manage CUDA dependencies cleanly.

```powershell
# Create environment
conda create -n ppe_env python=3.10 -y

# Activate
conda activate ppe_env
```

## 3. Install PyTorch with CUDA 12.1 Support
The NVIDIA T500 is a modern card (Turing) and works best with the latest stable CUDA.

```powershell
# Install PyTorch, torchvision, and torchaudio with CUDA 12.1 support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

> **Verification**:
> Run `python -c "import torch; print(torch.cuda.is_available())"`
> It **must** return `True`.

## 4. Install Project Dependencies
Install the remaining requirements from the project root.

```powershell
pip install -r requirements.txt
```

## 5. Optimization for T500
The T500 typically has 4GB VRAM. To avoid "Out of Memory" errors:
- **Model**: Use `yolov8n.pt` (Nano) or `yolov8s.pt` (Small). Avoid `yolov8m` or larger.
- **Input Size**: Keep `input_size: 640` in `config.yaml`.
- **FP16**: The code automatically enables FP16 if supported.

## Troubleshooting
- **DLL Load Failed**: Usually means Visual C++ Redistributable is missing. Install "VC_redist.x64".
- **OOM (Out Of Memory)**: Reduce `batch` size or check if other apps are using the GPU via Task Manager.
