#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# Netryx Astra V2 — One-command setup
# Run: chmod +x setup.sh && ./setup.sh
# ═══════════════════════════════════════════════════════════════
set -e

# Save the directory this script lives in (works even if called from elsewhere)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo ""
echo "🔱 Netryx Astra V2 — Setup"
echo "═══════════════════════════════════════"
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] Python not found."
    echo "        Download Python from https://python.org"
    echo "        Make sure to check \"Add Python to PATH\" during install."
    exit 1
fi

if ! python3 -c "import sys; exit(0 if sys.version_info >= (3,10) and sys.version_info < (3,13) else 1)"; then
    echo "[ERROR] Unsupported Python version detected."
    python3 --version
    echo ""
    echo "Astra requires Python 3.10 - 3.12."
    echo "Please install a compatible Python version."
    exit 1
fi

echo "[OK] Compatible Python installation found."

# Check Git
if ! command -v git &> /dev/null; then
    echo "[ERROR] Git not found. Download from https://git-scm.com"
    exit 1
fi
echo "[OK] Git found"

# Check for CUDA Toolkit
if command -v nvcc &> /dev/null; then
    echo "[OK] CUDA Toolkit detected"
else
    echo "[WARNING] CUDA Toolkit was not detected."
    echo "[WARNING] Astra can still run using PyTorch CUDA, but MASt3R will use a slower RoPE2D fallback."
    echo "[WARNING] Download CUDA Toolkit 12.4 from https://developer.nvidia.com/cuda-12-4-0-download-archive"
fi

# Create venv
cd "$SCRIPT_DIR"
if [ ! -d "venv" ]; then
    echo "[SETUP] Creating virtual environment..."
    python3 -m venv venv
fi

source venv/bin/activate 2>installer.log || source venv/Scripts/activate 2>installer.log
echo "[OK] Virtual environment activated"

# Install dependencies
echo ""
echo "[SETUP] Installing Python dependencies (this takes a few minutes)..."
python3 -m pip install --upgrade pip -q
python3 -m pip install -r requirements.txt -q

echo "Checking for NVIDIA GPU..."
if command -v nvidia-smi &> /dev/null; then
    echo "[OK] NVIDIA GPU detected"
    echo "Installing CUDA PyTorch..."
    python3 -m pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
else
    echo "[INFO] No NVIDIA GPU detected"
    echo "Installing CPU PyTorch..."
    python3 -m pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0
fi
echo "[OK] Dependencies installed"

# Clone MASt3R
echo ""
MAST3R_DIR="$SCRIPT_DIR/../mast3r"
if [ -f "$MAST3R_DIR/mast3r/model.py" ]; then
    echo "✅ MASt3R already cloned at $MAST3R_DIR"
else
    echo "[SETUP] Cloning MASt3R (this may take a few minutes)..."
    cd "$SCRIPT_DIR/.."
    git clone --recursive https://github.com/naver/mast3r.git
    cd "$SCRIPT_DIR/../mast3r"
    python3 -m pip install -r requirements.txt -q
    python3 -m pip install -r dust3r/requirements.txt -q
    cd "$SCRIPT_DIR"
    echo "[OK] MASt3R cloned and dependencies installed"
fi

# Return to Netryx directory
cd "$SCRIPT_DIR"

# Pre-download MegaLoc weights
echo ""
echo "[SETUP] Downloading MegaLoc model weights (first time only)..."
python3 -c "import torch; model = torch.hub.load('gmberton/MegaLoc', 'get_trained_model', trust_repo=True); print('[OK] MegaLoc ready')" 2>installer.log
if [ $? -ne 0 ]; then
    echo "[WARN] MegaLoc download failed - will retry on first run"
fi

# Pre-download MASt3R weights
echo ""
echo "[SETUP] Downloading MASt3R model weights (~1.2GB, first time only)..."
python3 -c "import sys,os; p=os.path.abspath(os.path.join('$SCRIPT_DIR','..','mast3r')); sys.path.insert(0,p); sys.path.insert(0,os.path.join(p,'dust3r')); from mast3r.model import AsymmetricMASt3R; m=AsymmetricMASt3R.from_pretrained('naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric'); print('[OK] MASt3R ready')" 2>installer.log
if [ $? -ne 0 ]; then
    echo "[WARN] MASt3R download failed - will retry on first run"
fi

# Create data dirs
cd "$SCRIPT_DIR"
mkdir -p netryx_data/megaloc_parts 2>installer.log
mkdir -p netryx_data/index 2>installer.log
echo "[OK] Data directories created"

# Done
echo ""
echo "===================================================="
echo "  Setup complete!"
echo ""
echo "  To run Netryx:"
echo "    Double-click run.bat"
echo "  Or:"
echo "    source venv/bin/activate"
echo "    python3 test_super.py"
echo "===================================================="
