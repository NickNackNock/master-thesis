import cv2
import numpy as np
from tqdm import tqdm
from rtmlib import draw_skeleton

from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment
from collections import defaultdict



HEAD_KP_INDICES = [0, 1, 2, 3, 4]  # Nose, Left Eye, Right Eye, Left Ear, Right Ear

PERSON_BOX_COLOR = (0, 255, 0)
HEAD_BOX_COLOR   = (255, 0, 0)

# Default confidence gates. These are intentionally duplicated (not
# imported from configuration.py) so this module keeps working stand-alone;
# main.py overrides them by passing configuration.KPT_SCORE_THRESH values
# explicitly into the functions below, so there's still a single source of
# truth for an actual pipeline run.
DEFAULT_VISUALIZATION_THRESH = 0.5
DEFAULT_HEAD_BBOX_THRESH     = 0.5

KEYPOINT_NAMES = [
    "Nose", "Left Eye", "Right Eye", "Left Ear", "Right Ear",
    "Left Shoulder", "Right Shoulder", "Left Elbow", "Right Elbow",
    "Left Wrist", "Right Wrist", "Left Hip", "Right Hip",
    "Left Knee", "Right Knee", "Left Ankle", "Right Ankle",
]


# ----- KEYPOINT DATA SAVER -----
class KeypointDataSaver:
    """Accumulates per-frame keypoint data per tracked ID, saves to .xlsx."""

    # The data structure is a dictionary where each key is a track_id (int) 
    # and the value is a list of dictionaries. 
    # Each dictionary in the list represents a frame and contains:
    #   - frame index
    #   - keypoint coordinates
    #   - and scores 
    # for a specific track_id.
    def __init__(self):
        """Sets up the empty {track_id: [row, ...]} accumulator.

        Nothing is written to disk until .save() is called; this class only
        buffers data in memory for the duration of one video pass.
        """
        # { track_id: [ {frame, kp_x0, kp_y0, score0, ...}, ... ] }
        self._data: dict[int, list[dict]] = defaultdict(list)


    # Record the keypoints and scores for a specific frame and track ID
    def record(self, frame_idx: int, track_id: int,
               keypoints: np.ndarray, scores: np.ndarray):
        """Appends one row (all 17 keypoints for one person, one frame).

        `keypoints`/`scores` are expected to already be the per-person
        (17, 2) / (17,) arrays for this track_id at this frame_idx. Missing
        keypoints (kp_idx >= len(keypoints), which in practice shouldn't
        happen since RTMO always emits all 17) are stored as None so Excel
        shows a true empty cell rather than a stray 0.0.
        """
        row = {"frame": frame_idx}

        # For each keypoint, we store its x and y coordinates and its score in the row dictionary.
        for kp_idx, name in enumerate(KEYPOINT_NAMES):
            key = name.lower().replace(" ", "_")
            if kp_idx < len(keypoints):
                row[f"kp_{key}_x"] = float(keypoints[kp_idx, 0])
                row[f"kp_{key}_y"] = float(keypoints[kp_idx, 1])
                row[f"score_{key}"] = float(scores[kp_idx])

            else:
                row[f"kp_{key}_x"] = None
                row[f"kp_{key}_y"] = None
                row[f"score_{key}"] = None

        self._data[track_id].append(row)

    # Save the accumulated data to an Excel file. Each track ID gets its own sheet in the workbook.
    def save(self, path: str):
        """Writes one sheet per track_id ("Person_<id>") to an .xlsx file.

        Sheet/column naming here is the contract that filtering_pose.py and
        retinaFaceGazeLLE.py rely on to read this file back
        (Person_<id> sheets, kp_<name>_x/_y + score_<name> columns).
        """
        wb = Workbook()
        wb.remove(wb.active)  # remove default empty sheet

        header_fill   = PatternFill("solid", start_color="4F81BD")
        header_font   = Font(bold=True, color="FFFFFF", name="Arial", size=10)

        cell_font = Font(name="Arial", size=10)

        for track_id in sorted(self._data.keys()):
            ws = wb.create_sheet(title=f"Person_{track_id}")
            rows = self._data[track_id]
            if not rows:
                continue

            headers = list(rows[0].keys())
            # Write header row
            for col, h in enumerate(headers, start=1):
                cell = ws.cell(row=1, column=col, value=h)
                cell.font   = header_font
                cell.fill   = header_fill
                cell.alignment = Alignment(horizontal="center", wrap_text=True)
                ws.column_dimensions[cell.column_letter].width = 14

            # Write data rows
            for row_idx, row_data in enumerate(rows, start=2):
                for col, h in enumerate(headers, start=1):
                    cell = ws.cell(row=row_idx, column=col, value=row_data[h])
                    cell.font = cell_font

            ws.freeze_panes = "B2"  # freeze header + frame column

        wb.save(path)
        print(f"Keypoint data saved → {path}")


# ----- BOUNDING BOXES -----
def head_bbox_from_pose(keypoints, scores, score_thresh=DEFAULT_HEAD_BBOX_THRESH, expansion=1.6):
    """Derives a head bounding box from the 5 facial keypoints.

    Only keypoints with score > score_thresh are used; if 2 or fewer of the
    5 face keypoints (nose, eyes, ears) are visible, returns None rather
    than guessing from an unreliable box. `expansion` pads the tight
    keypoint bbox outward so the crop comfortably contains the whole head
    (used both for the on-screen head box and as Gazelle's head-crop input).
    Returns [x1, y1, x2, y2] or None.
    """
    head_kps    = keypoints[HEAD_KP_INDICES]
    head_scores = scores[HEAD_KP_INDICES]

    visible = head_scores > score_thresh
    #print(visible.sum())

    if visible.sum() <= 2:
        return None
    
    pts = head_kps[visible]
    x1, y1 = pts.min(axis=0)
    x2, y2 = pts.max(axis=0)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    hw, hh = (x2 - x1) / 2 * expansion, (y2 - y1) / 2 * expansion
    return np.array([cx - hw, cy - hh, cx + hw, cy + hh])


def pose_to_bbox(keypoints: np.ndarray, expansion: float = 1.25) -> np.ndarray:
    """Tight axis-aligned bbox around a full set of body keypoints, expanded
    by `expansion` around its own center. Used both to feed the tracker
    (collapsing 17 keypoints into one box) and to draw the person's body box.
    """
    x, y = keypoints[:, 0], keypoints[:, 1]
    bbox   = np.array([x.min(), y.min(), x.max(), y.max()])
    center = np.array([bbox[0] + bbox[2], bbox[1] + bbox[3]]) / 2
    return np.concatenate([
        center - (center - bbox[:2]) * expansion,
        center + (bbox[2:] - center) * expansion,
    ])


# ----- INPUT VIDEO SETUP -----
def setup_input_video(input_video):
    """Opens `input_video` and returns (cap, total_frames, fps, width, height)."""
    cap    = cv2.VideoCapture(input_video)
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return cap, total, fps, width, height


def detection_confidence(p_scores: np.ndarray) -> float:
    """More robust than a flat mean: partial occlusion of legs/feet
    shouldn't tank the score used to decide if this is a 'real' detection
    for tracking purposes."""
    top_scores = np.sort(p_scores)[-8:]
    conf = float(top_scores.mean())
    return conf


# ----- POSE ONLY -----
def run_pose_estimation(input_video, output_video, pose_model,
                        bbox_body=None,
                        kpt_thr=DEFAULT_VISUALIZATION_THRESH):
    """Runs RTMO frame-by-frame with no tracking (fastest option).

    Draws the full skeleton (kpt_thr gates which joints are visible) plus
    optional body/head bounding boxes, and writes the annotated video to
    `output_video`. No xlsx is produced here — this is the quick visual
    sanity-check pass; use run_pose_estimation_with_tracking() for the
    tracked + exported version.
    """

    # Getting video properties and setting up the writer
    cap, total, fps, width, height = setup_input_video(input_video)
    writer = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*'mp4v'),
                             fps, (width, height))
    
    # Loop through each frame of the video, with a progress bar
    with tqdm(total=total, unit='frame', desc='Processing') as pbar:

        while True:

            ret, frame = cap.read()
            if not ret:
                break
            
            # Perform pose estimation on the current frame
            keypoints, scores = pose_model(frame)
            # kpt_thr is now a parameter (single source of truth lives in
            # configuration.KPT_SCORE_THRESH["visualization"], passed in by main.py)
            frame = draw_skeleton(frame, keypoints, scores, kpt_thr=kpt_thr)

            # Draw bounding boxes for each detected person based on the pose keypoints
            for person_kps, person_scores in zip(keypoints, scores):

                if bbox_body:
                    bbox = pose_to_bbox(person_kps[:, :2])
                    x1, y1, x2, y2 = bbox.astype(int)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), PERSON_BOX_COLOR, 2)

            writer.write(frame)
            pbar.update(1)

    cap.release()
    writer.release()
    print(f'Saved to {output_video}')


# ----- POSE + TRACKING (with data export) -----
def run_pose_estimation_with_tracking(input_video, output_video, pose_model,
                                      tracker, save_xlsx: str | None = None,
                                      kpt_thr=DEFAULT_VISUALIZATION_THRESH):
    """Runs RTMO + BoxMOT tracking, drawing IDs/boxes and (optionally)
    recording every person's keypoints to `save_xlsx` via KeypointDataSaver.

    Per frame: RTMO gives per-person keypoints -> each person's 17 keypoints
    are collapsed into one bbox (pose_to_bbox) so the tracker has something
    box-shaped to match across frames -> tracker.update() assigns stable
    track_ids -> each track_id is mapped back to its original keypoints via
    `pose_mapping` (built from detection order) so the recorded xlsx rows
    carry a persistent person identity across the whole video.
    """
    cap, total, fps, width, height = setup_input_video(input_video)
    writer = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*'mp4v'),
                             fps, (width, height))

    saver = KeypointDataSaver() if save_xlsx else None
    frame_idx = 0

    with tqdm(total=total, unit="frame", desc="Tracking") as pbar:

        while True:

            ret, frame = cap.read()
            if not ret:
                break
            
            keypoints, scores = pose_model(frame)
            frame = draw_skeleton(frame, keypoints, scores, kpt_thr=kpt_thr)

            # Collapses of 17 keypoints into a single bounding box for the tracker
            dets_list, pose_mapping = [], {}


            # For each detected person, create a bounding box and map it to the corresponding keypoints and scores
            for p_kps, p_scores in zip(keypoints, scores):

                bbox = pose_to_bbox(p_kps[:, :2])
                dets_list.append([*bbox, detection_confidence(p_scores), 0.0])
                pose_mapping[len(dets_list) - 1] = (p_kps, p_scores)

            # Feeds the current frame's detections to the tracker,
            # which assigns them IDs based on its internal logic (e.g. motion, appearance)
            dets   = np.array(dets_list) if dets_list else np.empty((0, 6))
            tracks = tracker.update(dets, frame)

            if tracks is not None and len(tracks):
                for track in tracks:
                    x1, y1, x2, y2, track_id, conf, cls, ind = track
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    track_id, ind   = int(track_id), int(ind)

                    cv2.rectangle(frame, (x1, y1), (x2, y2), PERSON_BOX_COLOR, 1)
                    cv2.putText(frame, f"ID {track_id}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, PERSON_BOX_COLOR, 2)

                    # If the track ID corresponds to a detected person, retrieve their keypoints and scores
                    if ind in pose_mapping:
                        p_kps, p_scores = pose_mapping[ind]

                        # Record data
                        if saver:
                            saver.record(frame_idx, track_id, p_kps[:, :2], p_scores)

            writer.write(frame)
            pbar.update(1)
            frame_idx += 1

    cap.release()
    writer.release()
    print(f'Video saved to {output_video}')

    if saver and save_xlsx:
        saver.save(save_xlsx)
