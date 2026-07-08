from pathlib import Path
import torch

# -------- DIRECTORIES --------
VIDEO_NAME = "output_ultra_cut.mp4"
INPUT_VIDEO = f"/home/neurolab/thesisProject/data/videos/{VIDEO_NAME}"

VIDEO_STEM = VIDEO_NAME[:-4]           # "output_ultra_cut"
BASE_OUT   = Path(f"./output/{VIDEO_STEM}")

OUTPUT_DIR_RAW_POSE  = BASE_OUT / "raw"      / "pose"
OUTPUT_DIR_RAW_GAZE  = BASE_OUT / "raw"      / "gaze"
OUTPUT_DIR_FILT_POSE = BASE_OUT / "filtered" / "pose"
OUTPUT_DIR_FILT_GAZE = BASE_OUT / "filtered" / "gaze"

# Check existence of directories and create them if they don't exist
for _d in (OUTPUT_DIR_RAW_POSE, OUTPUT_DIR_RAW_GAZE,
           OUTPUT_DIR_FILT_POSE, OUTPUT_DIR_FILT_GAZE):
    _d.mkdir(parents=True, exist_ok=True)

# -------- STEREO CALIBRATION --------
# Example path setup in main.py
STEREO_CALIB_FILE = "path/to/your/calibration_folder"


# -------- DEVICE --------
# Adapts on which device is available on the system
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


# -------- OUTPUT FILES --------
# REMINDER: change VIDEO_NAME above to switch between video inputs

# Pose — raw outputs
OUTPUT_RAW_POSE_VIDEO   = OUTPUT_DIR_RAW_POSE / "pose_RTMO-L.mp4"           # no tracking
OUTPUT_RAW_POSE_TRACKED = OUTPUT_DIR_RAW_POSE / "pose_RTMO-L_tracked.mp4"   # with tracking
OUTPUT_RAW_POSE_XLSX    = OUTPUT_DIR_RAW_POSE / "pose_RTMO-L_tracked.xlsx"  # keypoint export

# Pose — filtered outputs
OUTPUT_FILT_POSE_VIDEO  = OUTPUT_DIR_FILT_POSE / "pose_RTMO-L_filtered.mp4"
OUTPUT_FILT_POSE_XLSX   = OUTPUT_DIR_FILT_POSE / "pose_RTMO-L_filtered.xlsx"

# Gaze — stem only (gaze.py appends _ID1.mp4, _ID2.mp4, _ID3.mp4)
OUTPUT_RAW_GAZE_STEM    = OUTPUT_DIR_RAW_GAZE / "gaze_from_pose"


# -------- MODEL CONFIGURATION --------
# --- Gaze ---
# After some trial and error, I found this combination to produce satisfactory results
GAZELLE_CKPT = "/home/neurolab/repositories/gazelle/checkpoints/gazelle_dinov2_vitl14_inout_childplay.pt"

DETECTOR     = "yolov8l"
TRACKER_TYPE = "botsort"
REID         = "clip_market1501"

# --- Pose ---
BACKEND           = 'onnxruntime'
openpose_skeleton = False

# This is the best model so far that can identify CHILD/PARENT Keypoints
# by leveraging its one-stage efficiency
# It is possible to test other models, from custom to preset
# Check the github at the following link: https://github.com/Tau-J/rtmlib
# For a more detailed structure: https://deepwiki.com/Tau-J/rtmlib/4.2.2-rtmo
RTMO_MODEL_URL = (
    "https://download.openmmlab.com/mmpose/v1/projects/rtmo/onnx_sdk/"
    "rtmo-l_16xb16-600e_body7-640x640-b37118ce_20231211.zip"
)


# -------- LAZY MODEL LOADERS --------
# Models are instantiated on demand so that importing this module never
# triggers heavy GPU allocations just to read a path constant.

def load_pose_model():
    """Instantiates and returns the RTMO pose model."""
    from rtmlib import RTMO
    model = RTMO(onnx_model=RTMO_MODEL_URL, backend=BACKEND, device=DEVICE)
    print("Pose model (RTMO-L) loaded.")
    return model


def load_tracker():
    """Instantiates and returns the BoxMOT tracker (used for pose tracking)."""
    from boxmot.trackers.tracker_zoo import create_tracker
    tracker = create_tracker(
        tracker_type   = TRACKER_TYPE,
        tracker_config = None,
        reid_weights   = Path(f"{REID}.pt"),
        device         = DEVICE,
        half           = False,  # Gazelle's ReID model at full resolution, slower but more accurate
    )
    print(f"Tracker ({TRACKER_TYPE} + {REID}) loaded.")
    return tracker


def load_gaze_models(with_tracker: bool = False) -> dict:
    """
    Loads Gazelle and returns the variables dict consumed by gaze.py.
    Pass with_tracker=True only when using run_gaze_estimation() (legacy full pipeline)
    which needs tracker inside the dict. The new run_gaze_from_pose_xlsx() doesn't.
    """
    from gazelle.model import get_gazelle_model

    gazelle, gazelle_transform = get_gazelle_model("gazelle_dinov2_vitl14_inout")
    gazelle.load_gazelle_state_dict(torch.load(GAZELLE_CKPT, weights_only=True))
    gazelle.eval()
    gazelle.to(DEVICE)
    print("Gazelle loaded.")

    result = {
        "gazelle":           gazelle,
        "gazelle_transform": gazelle_transform,
        "DEVICE":            DEVICE,
    }

    if with_tracker:
        result["tracker"] = load_tracker()

    return result
