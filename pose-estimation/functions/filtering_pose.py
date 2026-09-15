import numpy as np
import pandas as pd
from scipy.signal import savgol_filter, medfilt
from openpyxl import Workbook
from openpyxl.styles import PatternFill, Font, Alignment

# Keypoint column naming convention
KEYPOINT_NAMES = [
    "Nose", "Left Eye", "Right Eye", "Left Ear", "Right Ear",
    "Left Shoulder", "Right Shoulder", "Left Elbow", "Right Elbow",
    "Left Wrist", "Right Wrist", "Left Hip", "Right Hip",
    "Left Knee", "Right Knee", "Left Ankle", "Right Ankle",
]

DEFAULT_SCORE_THRESH = 0.7


def _kp_col(name: str) -> str:
    return name.lower().replace(" ", "_")


def load_people_sheets(xlsx_path):
    return pd.read_excel(xlsx_path, sheet_name=None)

# A function to save the filtered keypoints into an xlsx file
# A sheet for each person, stylized to see better the data
def save_people_sheets(sheets: dict, out_path: str):
    wb = Workbook()
    wb.remove(wb.active)

    # Styling for the header and cells
    header_fill = PatternFill("solid", start_color="4F81BD")
    header_font = Font(bold=True, color="FFFFFF", name="Arial", size=10)
    cell_font = Font(name="Arial", size=10)

    for name, df in sheets.items():
        ws = wb.create_sheet(title=str(name))
        
        if df.empty:
            continue

        headers = list(df.columns)
        
        for col, h in enumerate(headers, start=1):
            cell = ws.cell(row=1, column=col, value=h)
            cell.font = header_font
            cell.fill = header_fill
            cell.alignment = Alignment(horizontal="center", wrap_text=True)
            ws.column_dimensions[cell.column_letter].width = 14

        for row_idx, row_data in enumerate(df.itertuples(index=False), start=2):
            for col, val in enumerate(row_data, start=1):
                clean_val = None if pd.isna(val) else val
                cell = ws.cell(row=row_idx, column=col, value=clean_val)
                cell.font = cell_font

        ws.freeze_panes = "B2"

    wb.save(out_path)


# DataFrame <-> array conversion
def df_to_arrays(df: pd.DataFrame, kp_names=KEYPOINT_NAMES):
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
    data = {"frame": frames}
    for i, name in enumerate(kp_names):
        key = _kp_col(name)
        data[f"kp_{key}_x"]  = kp_seq[:, i, 0]
        data[f"kp_{key}_y"]  = kp_seq[:, i, 1]
        data[f"score_{key}"] = scores[:, i]
    return pd.DataFrame(data)



# --------  FILTERING --------
def process_keypoint_track(kp_x, kp_y, scores, window_len, polyorder, min_seg_len, median_kernel=3):
    """
    Isolates chunks of valid data (above threshold), smoothes short 
    noise, and applies Savitzky-Golay filtering only to the valid segments.

    Filter all points regardless of score, but then when it will come to visualization
    they won't matter
    At least you can choose how much to visualize
    """
    n_frames = len(scores)
    # Valid means actually contains coordinate data
    valid =  ~np.isnan(kp_x)

    # Pad with False to easily find start/end boundaries of valid segments
    padded = np.pad(valid, (1, 1), mode='constant', constant_values=False)
    diff = np.diff(padded.astype(int))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]

    # Initialize blank output arrays (0.0 coords / 0.0 score is standard for dropped/missing points)
    out_x = np.zeros(n_frames)
    out_y = np.zeros(n_frames)
    out_scores = np.zeros(n_frames)

    #TODO: Check if this is the best way to do it, maybe there is a better way to do it
    #      Maybe a simple Butterworth
    for s, e in zip(starts, ends):
        seg_len = e - s

        # Short-burst Rejection (Despeckling)
        # If the detection only lasted a few frames between gaps, it's likely noise. Drop it.
        if seg_len < min_seg_len:
            continue

        seg_x = kp_x[s:e].copy()
        seg_y = kp_y[s:e].copy()

        # Local Median Filter (Optional but highly recommended)
        # Kills 1-frame tracking spikes *inside* the valid segment before Savgol smooths them.
        if median_kernel > 1 and seg_len >= median_kernel:
            seg_x = medfilt(seg_x, kernel_size=median_kernel)
            seg_y = medfilt(seg_y, kernel_size=median_kernel)

        # Dynamic Savgol Filter
        # N.B: SciPy requires window_len > polyorder, and window_len must be odd.
        w = min(window_len, seg_len)
        if w % 2 == 0:
            w -= 1

        if w > polyorder:
            seg_x = savgol_filter(seg_x, window_length=w, polyorder=polyorder)
            seg_y = savgol_filter(seg_y, window_length=w, polyorder=polyorder)

        # Write the cleaned segment back to the output
        out_x[s:e] = seg_x
        out_y[s:e] = seg_y
        out_scores[s:e] = scores[s:e]

    return out_x, out_y, out_scores


def filter_person(df: pd.DataFrame, 
                  window_len: int = 11, polyorder: int = 3, 
                  min_seg_len: int = 3, kp_names=KEYPOINT_NAMES) -> pd.DataFrame:
    """Full per-person pipeline: reindex -> extract valid segments -> savgol filter -> back to df."""
    frames, kp_seq, scores = df_to_arrays(df, kp_names)
    
    n_frames, n_kp, _ = kp_seq.shape
    kp_filtered = np.zeros_like(kp_seq)
    out_scores = np.zeros_like(scores)

    for k in range(n_kp):
        fx, fy, fs = process_keypoint_track(
            kp_seq[:, k, 0], 
            kp_seq[:, k, 1], 
            scores[:, k], 
            window_len=window_len,
            polyorder=polyorder,
            min_seg_len=min_seg_len,
            median_kernel=3             # Set to 0 if you want to bypass median spike removal
        )
        kp_filtered[:, k, 0] = fx
        kp_filtered[:, k, 1] = fy
        out_scores[:, k] = fs

    return arrays_to_df(frames, kp_filtered, out_scores, kp_names)


def filter_all_people(xlsx_path: str, out_path: str,
                      window_len: int = 11, polyorder: int = 3, min_seg_len: int = 3, kp_names=KEYPOINT_NAMES):
    sheets = load_people_sheets(xlsx_path)
    filtered_sheets = {
        name: filter_person(df, window_len, polyorder, min_seg_len, kp_names)
        for name, df in sheets.items()
    }
    save_people_sheets(filtered_sheets, out_path)
    print(f"Filtered keypoint data saved → {out_path}")
    return filtered_sheets



# --------  VIDEO DRAWING --------
def draw_filtered_video(input_video_path: str, output_video_path: str, person_dfs: dict,
                         kp_names=KEYPOINT_NAMES, kpt_thr: float = 0.7):
    import cv2
    from rtmlib import draw_skeleton

    indexed_dfs = {name: df.set_index("frame") for name, df in person_dfs.items()}

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
        for df in indexed_dfs.values():
            if frame_idx not in df.index:
                continue
            row = df.loc[frame_idx]
            kps    = np.array([[row[f"kp_{_kp_col(n)}_x"],
                                row[f"kp_{_kp_col(n)}_y"]] for n in kp_names])
            scores = np.array([row[f"score_{_kp_col(n)}"]  for n in kp_names])
            all_kps.append(kps)
            all_scores.append(scores)

        if all_kps:
            # We use the exact same threshold here that we used to filter
            frame = draw_skeleton(frame, np.array(all_kps), np.array(all_scores), kpt_thr=kpt_thr)
        writer.write(frame)

    cap.release()
    writer.release()
    print(f"Filtered video saved → {output_video_path}")


# Example usage
"""
if __name__ == "__main__":
    SCORE_THRESH = 0.7

    filtered_sheets = filter_all_people(
        xlsx_path="/home/neurolab/thesisProject/output/output_ultra_cut/pose/pose_RTMO-L.xlsx",
        out_path="keypoints_output_filtered.xlsx",
        score_thresh=SCORE_THRESH,
        window_len=11,      # Savgol window. Must be odd. Higher = smoother. 11 = ~0.4s at 25fps.
        polyorder=3,        # Polynomial order. 2 or 3 is best for human motion.
        min_seg_len=3       # Drop any valid detection streaks shorter than this many frames.
    )

    draw_filtered_video(
        input_video_path="/home/neurolab/thesisProject/data/videos/output_ultra_cut.mp4",
        output_video_path=OUTPUT_FILT_POSE_VIDEO,
        person_dfs=filtered_sheets,
        kpt_thr=SCORE_THRESH
    )
"""
