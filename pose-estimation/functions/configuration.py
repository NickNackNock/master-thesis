from pathlib import Path
import torch
from gazelle.model import get_gazelle_model

from rtmlib import RTMO
from boxmot.trackers.tracker_zoo import create_tracker

# -------- DIRECTORIES --------
INPUT_VIDEO  = "/home/neurolab/thesisProject/data/videos/output_ultra_cut.mp4"
OUTPUT_DIR_POSE = Path("./output/pose")
OUTPUT_DIR_GAZE = Path("./output/gaze")

# Check existance of directories and create them if they don't exist
OUTPUT_DIR_POSE.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR_GAZE.mkdir(parents=True, exist_ok=True)


# -------- MODELS --------
# Adapts on which device is available on the system
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"  


# --- Gaze ---
# After some trial and error, I found this combination to prouce satisfactory results
GAZELLE_CKPT = "/home/neurolab/repositories/gazelle/checkpoints/gazelle_dinov2_vitl14_inout_childplay.pt"

DETECTOR     = "yolov8l"
TRACKER_TYPE = "botsort"
REID         = "clip_market1501"

tracker = create_tracker(
    tracker_type = TRACKER_TYPE,
    tracker_config = None,
    reid_weights = Path(f"{REID}.pt"),
    device = DEVICE,
    half = False, # Gazelle's ReID model at full resolutio, slower but more accurate
)

gazelle, gazelle_transform = get_gazelle_model("gazelle_dinov2_vitl14_inout")
#gazelle = torch.compile(gazelle, mode="reduce-overhead")
gazelle.load_gazelle_state_dict(torch.load(GAZELLE_CKPT, weights_only=True))
gazelle.eval()
gazelle.to(DEVICE)
print("All models loaded.")

gazelle_variables = {
    "gazelle": gazelle,
    "gazelle_transform": gazelle_transform,
    "tracker": tracker,
    "DEVICE": DEVICE
    }

# --- Pose ---
BACKEND = 'onnxruntime'     
openpose_skeleton = False   

# This is the best model so far that can identify CHILD/PARENT Keypoints 
# by leveraging its one-stage efficiency
# It is possible to test other models, from custom to preset
# Check the github at the following link: https://github.com/Tau-J/rtmlib
# For a more detailed structure: https://deepwiki.com/Tau-J/rtmlib/4.2.2-rtmo
pose_model = RTMO(
    onnx_model ='https://download.openmmlab.com/mmpose/v1/projects/rtmo/onnx_sdk/rtmo-l_16xb16-600e_body7-640x640-b37118ce_20231211.zip',  
    backend = BACKEND,
    device = DEVICE
)


# -------- OUTPUT FILES --------
# REMINDER: change to a dynamic string
OUTPUT_POSE = OUTPUT_DIR_POSE / f"pose_RTMO-L.mp4"
OUTPUT_GAZE   = OUTPUT_DIR_GAZE / f"gaze_{DETECTOR}_{TRACKER_TYPE}_{REID}"
