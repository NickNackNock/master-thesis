from pathlib import Path
import torch
import os

# Just a check if the machine can use GPU or not
def device_availability():
    return "cuda:0" if torch.cuda.is_available() else "cpu"


# Creating the corisponding folders 
def check_output_folder(output_folder, PCI_n):

    # Creating sub directories in the output folder
    # of the current interactios
    output_folder_raw = output_folder / PCI_n / "raw"
    output_folder_filt = output_folder / PCI_n / "filt"

    output_folder_raw.mkdir(parents = True, exist_ok = True)
    output_folder_filt.mkdir(parents = True, exist_ok = True)   

    return output_folder_raw, output_folder_filt


def file_names(output_folder_raw, output_folder_filt):

    # --- Pose outputs ---
    # Raw
    output_video_pose_raw = output_folder_raw / "pose_RTMO-L.mp4"                   # no tracking
    output_video_poseTracking_raw = output_folder_raw / "pose_RTMO-L_tracked.mp4"   # with tracking
    output_xlsx_pose_raw = output_folder_raw / "pose_RTMO-L_tracked.xlsx"           # keypoint export

    # Filt
    output_video_pose_filt = output_folder_filt / "pose_RTMO-L_filtered.mp4"
    output_xlsx_pose_filt = output_folder_filt / "pose_RTMO-L_tracked.xlsx"   


    # --- Gaze output ---
    # Since it is uses the filtered pose we put it there
    output_video_gaze_filt = output_folder_filt / "Gaze"
    #output_xlsx_gaze_filt = output_folder_filt / "Gaze.xlsx"    To be decided

    # Putting everything into a variable 
    output_files = {
        "output_video_pose_raw" :        output_video_pose_raw,
        "output_video_poseTracking_raw": output_video_poseTracking_raw,
        "output_xlsx_pose_raw" :         output_xlsx_pose_raw,

        "output_video_pose_filt" :        output_video_pose_filt,
        "output_xlsx_pose_filt" :        output_xlsx_pose_filt,

        "output_video_gaze_filt" :        output_video_gaze_filt,
        #"output_xlsx_gaze_filt" :        output_xlsx_gaze_filt,
       
    }
    return output_files

def defined_thresholds():
    thesholds = {
        # Theshold to visualize in the videos produced
        # TODO: Why am I puttingdifferent thesholds?
        #       Doesn0t make much sense to me
        "visualization" : 0.7,
        "head_bbox"     : 0.5,
        "gaze_body_valid": 0.5,

    }

    return thesholds

# --------  MODEL LOADERS --------
# Models are instantiated on demand so that importing this module never
# triggers heavy GPU allocations just to read a path constant.

def load_pose_model():
    # This is the best model so far that can identify CHILD/PARENT Keypoints
    # by leveraging its one-stage efficiency
    # It is possible to test other models, from custom to preset
    # Check the github at the following link: https://github.com/Tau-J/rtmlib
    # For a more detailed structure: https://deepwiki.com/Tau-J/rtmlib/4.2.2-rtmo

    RTMO_MODEL_URL = (
        "https://download.openmmlab.com/mmpose/v1/projects/rtmo/onnx_sdk/"
        "rtmo-l_16xb16-600e_body7-640x640-b37118ce_20231211.zip"
    )

    BACKEND = 'onnxruntime'

    DEVICE = device_availability()

    from rtmlib import RTMO
    model = RTMO(onnx_model=RTMO_MODEL_URL, backend=BACKEND, device=DEVICE)
    print("Pose model (RTMO-L) loaded.")
    return model


def load_tracker():
    """
    Instantiates and returns the BoxMOT tracker (used for pose tracking).

    Builds a BotSORT tracker with a CLIP/Market1501 re-ID backbone. This is
    what turns per-frame RTMO detections into stable track IDs across
    frames

    Personalized config .yaml to allow better recofgnition of people when disappearing from
    view for long periods of time and allow less ID swap
    """

    from boxmot.trackers.tracker_zoo import create_tracker

    DEVICE = device_availability()
    REID = "clip_market1501"
    TRACKER_TYPE = "botsort"

    # If file not found default it to None, which usees default parameters
    tracker_config_path = Path("/home/neurolab/thesisProject/src/pose-gaze-3/files/configs/botsort_thesis.yaml")
    if not tracker_config_path.exists():
        print("Personal configuration file not found. \n" \
        "       Going back to default parameters")
        tracker_config_path = None
        
    tracker = create_tracker(
        tracker_type   = TRACKER_TYPE,
        tracker_config = tracker_config_path,
        reid_weights   = Path(f"{REID}.pt"),
        device         = DEVICE,
        half           = False,
    )

    print(f"Tracker ({TRACKER_TYPE} + {REID}) loaded.")
    return tracker


def load_gaze_models() -> dict:
    """
    Loads Gazelle and returns the variables dict consumed by gaze.py.
    """
    from gazelle.model import get_gazelle_model

    DEVICE = device_availability()
    GAZELLE_CKPT = "/home/neurolab/repositories/gazelle/checkpoints/gazelle_dinov2_vitl14_inout_childplay.pt"

    gazelle, gazelle_transform = get_gazelle_model("gazelle_dinov2_vitl14_inout")
    gazelle.load_gazelle_state_dict(torch.load(GAZELLE_CKPT, weights_only=True))
    gazelle.eval()
    gazelle.to(DEVICE)
    print("Gazelle loaded.")

    result = {
        "gazelle":           gazelle,
        "gazelle_transform": gazelle_transform,
        "DEVICE":            DEVICE
    }

    return result

