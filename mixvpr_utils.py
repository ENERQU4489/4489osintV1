"""
mixvpr_utils.py — Optional MixVPR encoder, drop-in alternative to megaloc_utils.

MixVPR ("Feature Mixing for Visual Place Recognition", 2023) is a single-backbone
place-recognition model. Compared to MegaLoc it produces much smaller descriptors
(512 or 4096 vs 8448) and encodes faster, at a small recall cost. This module is
fully self-contained and does NOT modify or import megaloc_utils — the two encoders
can coexist. Descriptors from the two models are NOT interchangeable: an index built
with one must be searched with the same one.

Weights (ResNet50 backbone, from the official MixVPR release) are downloaded once
from Google Drive into ./models_mixvpr/.

API mirrors megaloc_utils:
    get_mixvpr_model, extract_mixvpr_descriptor, batch_extract_mixvpr,
    mixvpr_similarity, MIXVPR_RAW_DIM
"""

import os
import sys
import threading
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

# ── Locate the VPR-methods repo that ships the MixVPR architecture ──────────
# (cached by torch.hub the first time any VPR model is fetched)
_VPR_REPO = os.path.expanduser(
    "~/.cache/torch/hub/gmberton_VPR-methods-evaluation_master")
if os.path.isdir(_VPR_REPO) and _VPR_REPO not in sys.path:
    sys.path.insert(0, _VPR_REPO)

# Prefer the Expansion data drive (keeps large files off a small OS disk),
# fall back to a local folder next to this file.
_EXP = "/Volumes/Expansion/netryx"
if os.path.isdir(_EXP):
    MODELS_DIR = os.path.join(_EXP, "models_mixvpr")
else:
    MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models_mixvpr")

# descriptor_dim -> (google-drive file id, local filename)
_WEIGHTS = {
    512:  ("1khiTUNzZhfV2UUupZoIsPIbsMRBYVDqj", "mixvpr_512.ckpt"),
    4096: ("1vuz3PvnR7vxnDDLQrdHJaOA04SQrtk5L", "mixvpr_4096.ckpt"),
}

MIXVPR_INPUT_SIZE = 320  # ResNet50 layer3 -> 20x20 feature map, as MixVPR expects
MIXVPR_RAW_DIM = 512     # default option; 4096 also available

_model = None
_model_dim = None
_lock = threading.Lock()

if torch.backends.mps.is_available():
    _device = 'mps'
elif torch.cuda.is_available():
    _device = 'cuda'
else:
    _device = 'cpu'

# ImageNet normalization (MixVPR was trained with it)
_preprocess = T.Compose([
    T.Resize((MIXVPR_INPUT_SIZE, MIXVPR_INPUT_SIZE), antialias=True),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def _load_mixvpr_arch():
    """Load the MixVPRModel class + MODELS_INFO directly from the repo file.

    We load mixvpr.py by path (not `import vpr_models.mixvpr`) because the
    package __init__ eagerly imports every VPR method, some needing deps we
    don't have. mixvpr.py imports `gdown` only for its own downloader, which
    we don't use — so we stub it if absent.
    """
    import importlib.util
    if "gdown" not in sys.modules:
        try:
            import gdown  # noqa: F401
        except Exception:
            import types
            stub = types.ModuleType("gdown")
            stub.download = lambda *a, **k: (_ for _ in ()).throw(
                RuntimeError("gdown stubbed; use mixvpr_utils._download_weights"))
            sys.modules["gdown"] = stub
    path = os.path.join(_VPR_REPO, "vpr_models", "mixvpr.py")
    spec = importlib.util.spec_from_file_location("_mixvpr_arch", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.MixVPRModel, mod.MODELS_INFO


def _download_weights(dim):
    fid, fname = _WEIGHTS[dim]
    os.makedirs(MODELS_DIR, exist_ok=True)
    path = os.path.join(MODELS_DIR, fname)
    if os.path.exists(path) and os.path.getsize(path) > 1_000_000:
        return path
    import requests
    print(f"[MIXVPR] Downloading {dim}-dim weights from Google Drive...")
    s = requests.Session()
    r = s.get("https://drive.usercontent.google.com/download",
              params={"id": fid, "export": "download", "confirm": "t"},
              stream=True, timeout=120)
    r.raise_for_status()
    with open(path, "wb") as f:
        for chunk in r.iter_content(1 << 20):
            f.write(chunk)
    if open(path, "rb").read(2) != b"PK":
        os.remove(path)
        raise RuntimeError("MixVPR weight download did not return a valid checkpoint")
    print(f"[MIXVPR] Saved weights to {path} ({os.path.getsize(path)/1e6:.0f} MB)")
    return path


def get_mixvpr_model(dim=MIXVPR_RAW_DIM, device=None):
    """Load MixVPR (singleton, thread-safe). dim is 512 or 4096."""
    global _model, _model_dim, MIXVPR_RAW_DIM
    if _model is not None and _model_dim == dim:
        return _model
    with _lock:
        if _model is not None and _model_dim == dim:
            return _model
        if dim not in _WEIGHTS:
            raise ValueError(f"MixVPR dim must be one of {list(_WEIGHTS)}, got {dim}")
        dev = device or _device
        MixVPRModel, MODELS_INFO = _load_mixvpr_arch()
        _, _, out_channels, out_rows = MODELS_INFO[dim]
        cfg = {"in_channels": 1024, "in_h": 20, "in_w": 20,
               "out_channels": out_channels, "mix_depth": 4,
               "mlp_ratio": 1, "out_rows": out_rows}
        print(f"[MIXVPR] Loading MixVPR ({dim}-dim) on {dev}...")
        model = MixVPRModel(agg_config=cfg)
        state = torch.load(_download_weights(dim), map_location="cpu", weights_only=False)
        model.load_state_dict(state)
        model = model.eval().to(dev)
        for p in model.parameters():
            p.requires_grad_(False)
        _model, _model_dim, MIXVPR_RAW_DIM = model, dim, dim
        print(f"[MIXVPR] Model ready. Output dim: {dim}")
        return _model


def extract_mixvpr_descriptor(pil_img, dim=MIXVPR_RAW_DIM):
    """Extract a single L2-normalized MixVPR descriptor from a PIL image."""
    model = get_mixvpr_model(dim)
    t = _preprocess(pil_img.convert("RGB")).unsqueeze(0).to(_device)
    with torch.no_grad():
        desc = model(t)
    return desc.float().cpu().numpy().squeeze()


def batch_extract_mixvpr(pil_images, batch_size=32, dim=MIXVPR_RAW_DIM):
    """Batch extract MixVPR descriptors. Returns (N, dim) float32, L2-normalized."""
    model = get_mixvpr_model(dim)
    out = []
    for i in range(0, len(pil_images), batch_size):
        batch = pil_images[i:i + batch_size]
        t = torch.stack([_preprocess(im.convert("RGB")) for im in batch]).to(_device)
        with torch.no_grad():
            d = model(t)
        out.append(d.float().cpu().numpy())
        if _device == 'mps' and (i // batch_size) % 10 == 0:
            torch.mps.empty_cache()
    return np.vstack(out)


def mixvpr_similarity(d1, d2):
    """Cosine similarity between two L2-normalized descriptors."""
    return float(np.dot(d1, d2))


if __name__ == "__main__":
    m = get_mixvpr_model(512)
    img = Image.new("RGB", (400, 300), (120, 90, 60))
    d = extract_mixvpr_descriptor(img)
    print("descriptor shape:", d.shape, "| L2 norm:", round(float(np.linalg.norm(d)), 4))
