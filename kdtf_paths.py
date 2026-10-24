"""Locations of external assets used by stage-1 preprocessing. Override with environment variables."""
import os

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
# BEV / ROMP / SMPL files (BEV.pth, SMPLA_NEUTRAL.pth, smil_packed_info.pth, ...)
ROMP_DIR = os.environ.get('KDTF_ROMP_DIR', os.path.join(REPO_ROOT, 'romp', 'files', '.romp'))
# Scratch space for cropped agent images
TMP_DIR = os.environ.get('KDTF_TMP_DIR', os.path.join(REPO_ROOT, 'temp'))
# PLLaVA-13B weights (SIT captioning only)
PLLAVA_DIR = os.environ.get('PLLAVA_MODEL_DIR', os.path.join(REPO_ROOT, 'checkpoints', 'pllava-13b'))
