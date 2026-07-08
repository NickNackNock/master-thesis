"""
Keypoint filtering pipeline for multi-person pose-estimation data.

Input:  an .xlsx file with one sheet per detected person. Each sheet has a
        "frame" column plus kp_<name>_x, kp_<name>_y, score_<name> columns.
Output: a filtered .xlsx (same structure) ready to feed back into your
        video-drawing step.
"""

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

KEYPOINT_NAMES = [
    "Nose", "Left Eye", "Right Eye", "Left Ear", "Right Ear",
    "Left Shoulder", "Right Shoulder", "Left Elbow", "Right Elbow",
    "Left Wrist", "Right Wrist", "Left Hip", "Right Hip",
    "Left Knee", "Right Knee", "Left Ankle", "Right Ankle",
]


def _kp_col(name: str) -> str:
    """Normalize a keypoint display name to a column-key string.
    e.g.  'Left Eye'  →  'left_eye'
    Must match the convention used by KeypointDataSaver in pose.py.
    """
    return name.lower().replace(" ", "_")


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def load_people_sheets(xlsx_path):
    """Returns {sheet_name: DataFrame} for every person in the file."""
    return pd.read_excel(xlsx_path, sheet_name=None)


def save_people_sheets(sheets: dict, out_path: str):
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        for name, df in sheets.items():
            df.to_excel(writer, sheet_name=name, index=False)


# ---------------------------------------------------------------------------
# DataFrame <-> array conversion (frame-index aware)
# ---------------------------------------------------------------------------

def df_to_arrays(df: pd.DataFrame, kp_names=KEYPOINT_NAMES):
    """
    Reindexes onto the full contiguous frame range so array position ==
    actual frame offset. Frames the person was never detected in become
    NaN rows (instead of being silently skipped).
    """
    df = df.sort_values("frame")
    frames = df["frame"].to_numpy()
    full_frames = np.arange(frames.min(), frames.max() + 1)

    df_full = df.set_index("frame").reindex(full_frames)

    n_frames = len(full_frames)
    n_kp = len(kp_names)
    kp_seq = np.full((n_frames, n_kp, 2), np.nan)
    scores = np.full((n_frames, n_kp), np.nan)

    for i, name in enumerate(kp_names):
        key = _kp_col(name)
        kp_seq[:, i, 0] = df_full[f"kp_{key}_x"].to_numpy()
        kp_seq[:, i, 1] = df_full[f"kp_{key}_y"].to_numpy()
        scores[:, i]    = df_full[f"score_{key}"].to_numpy()

    return full_frames, kp_seq, scores


def arrays_to_df(frames, kp_seq, scores, kp_names=KEYPOINT_NAMES):
    """
    Converts filtered arrays back to a DataFrame using the same lowercase
    column convention as KeypointDataSaver in pose.py (kp_left_eye_x, etc.)
    so that the filtered xlsx can be loaded directly by gaze.py.
    """
    data = {"frame": frames}
    for i, name in enumerate(kp_names):
        key = _kp_col(name)
        data[f"kp_{key}_x"]  = kp_seq[:, i, 0]
        data[f"kp_{key}_y"]  = kp_seq[:, i, 1]
        data[f"score_{key}"] = scores[:, i]
    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------

def interpolate_missing(kp_seq: np.ndarray, scores: np.ndarray, score_thresh: float = 0.5):
    """
    Linearly interpolates over missing/low-confidence frames for each
    keypoint independently. Returns the interpolated sequence plus a
    (n_frames, n_kp) boolean mask of which frames were ever "valid" for
    each keypoint (i.e. detected with score above threshold).

    Keypoints with fewer than 2 valid points can't be interpolated and are
    left as NaN -- the caller must not filter or draw these.
    """
    kp_seq = kp_seq.copy()
    n_frames, n_kp, _ = kp_seq.shape
    valid_mask = np.zeros((n_frames, n_kp), dtype=bool)
    idx_all = np.arange(n_frames)

    for k in range(n_kp):
        valid = (scores[:, k] > score_thresh) & ~np.isnan(kp_seq[:, k, 0])
        valid_mask[:, k] = valid

        if valid.sum() < 2:
            kp_seq[:, k, :] = np.nan
            continue

        for axis in range(2):
            kp_seq[:, k, axis] = np.interp(idx_all, idx_all[valid], kp_seq[valid, k, axis])

    return kp_seq, valid_mask


"""def safe_filtfilt(b, a, signal: np.ndarray) -> np.ndarray:
    #filtfilt raises if the signal is too short for its default padding;
    #fall back to returning the (already interpolated) signal unfiltered.
    padlen = 3 * max(len(a), len(b))
    if len(signal) <= padlen:
        return signal
    return filtfilt(b, a, signal)"""


def lowpass_filter_keypoints(kp_seq: np.ndarray, fps: float, cutoff: float = 6.0,
                              order: int = 2, valid_mask: np.ndarray = None) -> np.ndarray:
    """
    kp_seq: (n_frames, n_keypoints, 2), already gap-free (interpolated).
    valid_mask: (n_frames, n_keypoints) bool -- keypoints with < 2 valid
        samples are skipped instead of filtering NaN/garbage.
    """
    nyquist = fps / 2.0
    if cutoff >= nyquist:
        cutoff = 0.99 * nyquist  # clamp instead of letting butter() error out

    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype="low", analog=False)

    filtered = kp_seq.copy()
    n_kp = kp_seq.shape[1]

    
    for k in range(n_kp):
        if valid_mask is not None and valid_mask[:, k].sum() < 2:
            continue
        for axis in range(2):
            filtered[:, k, axis] = filtfilt(b, a, kp_seq[:, k, axis])
            print(f"Filtered keypoint {k} axis {axis}")

    return filtered 


def filter_person(df: pd.DataFrame, fps: float, cutoff: float = 6.0, order: int = 2,
                   score_thresh: float = 0.5, kp_names=KEYPOINT_NAMES) -> pd.DataFrame:
    """Full per-person pipeline: reindex -> interpolate -> lowpass filter -> back to df."""
    frames, kp_seq, scores = df_to_arrays(df, kp_names)
    kp_interp, valid_mask = interpolate_missing(kp_seq, scores, score_thresh)
    kp_filtered = lowpass_filter_keypoints(kp_interp, fps, cutoff, order, valid_mask)

    # Keypoints that were never reliably detected: zero them out and force
    # score to 0 so downstream drawing code skips them (kpt_thr will hide them).
    never_valid = ~valid_mask.any(axis=0)  # (n_kp,)
    out_scores = np.where(np.isnan(scores), 0.0, scores)
    out_scores[:, never_valid] = 0.0
    kp_filtered[:, never_valid, :] = 0.0

    return arrays_to_df(frames, kp_filtered, out_scores, kp_names)


def filter_all_people(xlsx_path: str, out_path: str, fps: float, cutoff: float = 6.0,
                       order: int = 2, score_thresh: float = 0.5, kp_names=KEYPOINT_NAMES):
    """Loads every sheet, filters each person, saves a new .xlsx with the same sheet names."""
    sheets = load_people_sheets(xlsx_path)
    filtered_sheets = {
        name: filter_person(df, fps, cutoff, order, score_thresh, kp_names)
        for name, df in sheets.items()
    }
    save_people_sheets(filtered_sheets, out_path)
    print(f"Filtered keypoint data saved → {out_path}")
    return filtered_sheets


# ---------------------------------------------------------------------------
# Video drawing (multi-person aware)
# ---------------------------------------------------------------------------

def draw_filtered_video(input_video_path: str, output_video_path: str, person_dfs: dict,
                         kp_names=KEYPOINT_NAMES, kpt_thr: float = 0.5):
    """
    person_dfs: {person_name: filtered_df}, as returned by filter_all_people.
    Draws all detected people for each frame in one pass (rather than one
    person at a time, since draw_skeleton expects all instances together).
    """
    import cv2
    from rtmlib import draw_skeleton

    cap = cv2.VideoCapture(input_video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    n_total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for frame_idx in range(n_total_frames):
        ret, frame = cap.read()
        if not ret:
            break

        all_kps, all_scores = [], []
        for df in person_dfs.values():
            row = df[df["frame"] == frame_idx]
            if row.empty:
                continue
            row = row.iloc[0]
            # Use _kp_col() so lookup matches the lowercase column names written
            # by arrays_to_df (e.g. "Left Eye" → "kp_left_eye_x")
            kps    = np.array([[row[f"kp_{_kp_col(n)}_x"],
                                row[f"kp_{_kp_col(n)}_y"]] for n in kp_names])
            scores = np.array([row[f"score_{_kp_col(n)}"]  for n in kp_names])
            all_kps.append(kps)
            all_scores.append(scores)

        if all_kps:
            frame = draw_skeleton(frame, np.array(all_kps), np.array(all_scores), kpt_thr=kpt_thr)
        writer.write(frame)

    cap.release()
    writer.release()
    print(f"Filtered video saved → {output_video_path}")


# ---------------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    FPS = 25.0  # use your video's actual fps, e.g. via cv2.VideoCapture(...).get(cv2.CAP_PROP_FPS)

    filtered_sheets = filter_all_people(
        xlsx_path="/home/neurolab/thesisProject/output/output_ultra_cut/pose/pose_RTMO-L.xlsx",
        out_path="keypoints_output_filtered.xlsx",
        fps=FPS,
        cutoff=6.0,     # Hz -- lower = smoother but more lag. 4-8 Hz is typical for human movement.
        order=2,        # filtfilt makes this effectively 4th-order, zero-phase
        score_thresh=0.8,
    )

    draw_filtered_video(
        input_video_path="/home/neurolab/thesisProject/data/videos/output_ultra_cut.mp4",
        output_video_path="output_filtered.mp4",
        person_dfs=filtered_sheets,
    )
