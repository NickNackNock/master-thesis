import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont
from rtmlib import RTMO, draw_skeleton
from boxmot.trackers.tracker_zoo import create_tracker
from gazelle.model import get_gazelle_model
from gazelle.utils import visualize_heatmap
from datetime import datetime

# ── Config ────────────────────────────────────────────────────────────────────
INPUT_VIDEO  = "/home/neurolab/thesisProject/data/videos/cut_visible.mp4"
OUTPUT_DIR   = Path("./output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

GAZELLE_CKPT = "/home/neurolab/repositories/gazelle/checkpoints/gazelle_dinov2_vitl14_inout_childplay.pt"

DETECTOR     = "yolov8l"
TRACKER_TYPE = "botsort"
REID         = "clip_market1501"

OUTPUT_VIDEO   = OUTPUT_DIR / f"gaze_{DETECTOR}_{TRACKER_TYPE}_{REID}_3.mp4"
OUTPUT_VIDEO_1 = OUTPUT_DIR / f"gaze_id1_{DETECTOR}_{TRACKER_TYPE}_{REID}_3.mp4"
OUTPUT_VIDEO_2 = OUTPUT_DIR / f"gaze_id2_{DETECTOR}_{TRACKER_TYPE}_{REID}_3.mp4"
META_FILE      = OUTPUT_DIR / f"gaze_{DETECTOR}_{TRACKER_TYPE}_{REID}_3.txt"

DEVICE  = "cuda:0" if torch.cuda.is_available() else "cpu"
BACKEND = "onnxruntime"

# Drawing settings
INOUT_THRESH     = 0.5
SCORE_THRESH_KP  = 0.4
SCORE_THRESH_DET = 0.3
HEAD_EXPANSION   = 1.6
HEATMAP_ALPHA    = 90

ID_COLORS = [
    (0,   255,   0),
    (255,  50,  50),
    (0,   200, 255),
    (255,   0, 200),
    (255, 255,   0),
]

FACE_KP_INDICES = [0, 1, 2, 3, 4]

# IDs to isolate in separate videos
SOLO_IDS = {1, 2}
# ── Helpers (unchanged) ───────────────────────────────────────────────────────

def id_color(track_id: int):
    return ID_COLORS[track_id % len(ID_COLORS)]


def pose_to_bbox(keypoints: np.ndarray, expansion: float = 1.25) -> np.ndarray:
    x, y   = keypoints[:, 0], keypoints[:, 1]
    bbox   = np.array([x.min(), y.min(), x.max(), y.max()])
    center = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2])
    return np.concatenate([
        center - (center - bbox[:2]) * expansion,
        center + (bbox[2:] - center) * expansion,
    ])


def head_bbox_from_pose(
    keypoints: np.ndarray,
    scores: np.ndarray,
    score_thresh: float = SCORE_THRESH_KP,
    expansion: float    = HEAD_EXPANSION,
) -> np.ndarray | None:
    head_kps    = keypoints[FACE_KP_INDICES]
    head_scores = scores[FACE_KP_INDICES]
    visible     = head_scores > score_thresh

    if visible.sum() < 2:
        return None

    pts    = head_kps[visible]
    x1, y1 = pts.min(axis=0)
    x2, y2 = pts.max(axis=0)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    hw, hh = (x2 - x1) / 2 * expansion, (y2 - y1) / 2 * expansion
    return np.array([cx - hw, cy - hh, cx + hw, cy + hh])


def head_bbox_normalized(head_bbox_px: np.ndarray, width: int, height: int) -> np.ndarray:
    return head_bbox_px / np.array([width, height, width, height])


def draw_person_on_frame(
    frame_clean: np.ndarray,   # ← clean frame, no other skeletons
    pil_img: Image.Image,
    track_id: int,
    x1: int, y1: int, x2: int, y2: int,
    p_kps: np.ndarray,
    p_scores: np.ndarray,
    heatmap: torch.Tensor | None,
) -> np.ndarray:
    color = id_color(track_id)

    # Only this person's skeleton on a clean base
    frame_out = draw_skeleton(
        frame_clean.copy(),
        p_kps[np.newaxis],
        p_scores[np.newaxis],
        kpt_thr=SCORE_THRESH_KP,
    )

    # Bounding box + label
    cv2.rectangle(frame_out, (x1, y1), (x2, y2), color, 2)
    cv2.putText(frame_out, f"ID {track_id}", (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Face keypoints
    for kp_idx in FACE_KP_INDICES:
        if p_scores[kp_idx] > SCORE_THRESH_KP:
            cv2.circle(frame_out,
                       (int(p_kps[kp_idx][0]), int(p_kps[kp_idx][1])),
                       5, color, -1)

    # Gaze heatmap
    if heatmap is not None:
        pil_with_heatmap = visualize_heatmap(pil_img.copy(), heatmap)
        heatmap_bgr = cv2.cvtColor(
            np.array(pil_with_heatmap.convert("RGB")), cv2.COLOR_RGB2BGR
        )
        frame_out = cv2.addWeighted(heatmap_bgr, 0.6, frame_out, 0.4, 0)

    return frame_out


# ── Model init ────────────────────────────────────────────────────────────────

print(f"Device: {DEVICE}")

pose_model = RTMO(
    onnx_model=(
        "https://download.openmmlab.com/mmpose/v1/projects/rtmo/onnx_sdk/"
        "rtmo-l_16xb16-600e_body7-640x640-b37118ce_20231211.zip"
    ),
    backend=BACKEND,
    device=DEVICE,
)

tracker = create_tracker(
    tracker_type=TRACKER_TYPE,
    tracker_config=None,
    reid_weights=Path(f"{REID}.pt"),
    device=DEVICE,
    half=False,
)

gazelle, gazelle_transform = get_gazelle_model("gazelle_dinov2_vitl14_inout")
gazelle.load_gazelle_state_dict(torch.load(GAZELLE_CKPT, weights_only=True))
gazelle.eval()
gazelle.to(DEVICE)
print("All models loaded.")


# ── Video IO ──────────────────────────────────────────────────────────────────

cap = cv2.VideoCapture(INPUT_VIDEO)
if not cap.isOpened():
    raise RuntimeError(f"Cannot open: {INPUT_VIDEO}")

total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer   = cv2.VideoWriter(str(OUTPUT_VIDEO),   fourcc, fps, (width, height))
writer_1 = cv2.VideoWriter(str(OUTPUT_VIDEO_1), fourcc, fps, (width, height))
writer_2 = cv2.VideoWriter(str(OUTPUT_VIDEO_2), fourcc, fps, (width, height))

solo_writers = {1: writer_1, 2: writer_2}


# ── Main loop ─────────────────────────────────────────────────────────────────

with tqdm(total=total, unit="frame", desc="Tracking + Gaze") as pbar:
    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        pil_img = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))

        # 1. Pose estimation
        keypoints, scores = pose_model(frame_bgr)

        # 2. Draw body skeletons on the main output frame
        #frame_bgr = draw_skeleton(frame_bgr, keypoints, scores, kpt_thr=SCORE_THRESH_KP)
        frame_clean = frame_bgr.copy()

        # 3. Build detections for tracker
        dets_list    = []
        pose_mapping = {}

        for i, (p_kps, p_scores) in enumerate(zip(keypoints, scores)):
            if p_scores.mean() < SCORE_THRESH_DET:
                continue
            bbox      = pose_to_bbox(p_kps[:, :2])
            mean_conf = float(p_scores.mean())
            dets_list.append([bbox[0], bbox[1], bbox[2], bbox[3], mean_conf, 0.0])
            pose_mapping[len(dets_list) - 1] = (p_kps, p_scores)

        dets   = np.array(dets_list) if dets_list else np.empty((0, 6))
        tracks = tracker.update(dets, frame_bgr)

        # Prepare blank frames for solo writers (written even if person not visible)
        solo_frames = {sid: frame_bgr.copy() for sid in SOLO_IDS}

        if tracks is not None and len(tracks):

            gazelle_bboxes = []
            track_meta     = []

            for track in tracks:
                x1, y1, x2, y2, track_id, conf, cls, ind = track
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                track_id        = int(track_id)
                ind             = int(ind)
                color           = id_color(track_id)

                # ── Main video: draw box + label ──────────────────────────
                cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame_bgr, f"ID {track_id}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                if ind not in pose_mapping:
                    track_meta.append(None)
                    continue

                p_kps, p_scores = pose_mapping[ind]

                # Draw skeleton per-person on main video, identified by track_id
                frame_bgr = draw_skeleton(
                    frame_bgr,
                    p_kps[np.newaxis],
                    p_scores[np.newaxis],
                    kpt_thr=SCORE_THRESH_KP,
                )


                # Face keypoints on main video
                for kp_idx in FACE_KP_INDICES:
                    if p_scores[kp_idx] > SCORE_THRESH_KP:
                        cv2.circle(frame_bgr,
                                   (int(p_kps[kp_idx][0]), int(p_kps[kp_idx][1])),
                                   5, color, -1)

                # Head bbox
                head_px = head_bbox_from_pose(p_kps[:, :2], p_scores)
                if head_px is None:
                    track_meta.append(None)
                    continue

                hx1, hy1, hx2, hy2 = head_px.astype(int)
                cv2.rectangle(frame_bgr, (hx1, hy1), (hx2, hy2), color, 1)

                norm_bbox = head_bbox_normalized(head_px, width, height)
                norm_bbox = np.clip(norm_bbox, 0.0, 1.0)

                gazelle_bboxes.append(norm_bbox)
                track_meta.append((track_id, norm_bbox, x1, y1, x2, y2, p_kps, p_scores))

            # 5. Batch Gazelle inference
            valid_entries = [m for m in track_meta if m is not None]

            if valid_entries:
                img_tensor  = gazelle_transform(pil_img).unsqueeze(0).to(DEVICE)
                model_input = {
                    "images":  img_tensor,
                    "bboxes": [[m[1] for m in valid_entries]],
                }

                with torch.no_grad():
                    gaze_out = gazelle(model_input)

                heatmaps     = gaze_out["heatmap"][0]   # (N, 64, 64)
                inout_scores = gaze_out["inout"]

                # 6. Main video heatmap + solo video rendering
                pil_overlay = pil_img.copy()

                for valid_idx, meta in enumerate(valid_entries):
                    track_id, norm_bbox, x1, y1, x2, y2, p_kps, p_scores = meta
                    heatmap = heatmaps[valid_idx]

                    # Accumulate heatmap on main overlay
                    pil_overlay = visualize_heatmap(pil_overlay, heatmap)

                    # Per-person solo frame
                    if track_id in SOLO_IDS:
                        solo_frames[track_id] = draw_person_on_frame(
                        frame_clean = frame_clean,  # ← clean, no other skeletons
                        pil_img     = pil_img,
                        track_id    = track_id,
                        x1=x1, y1=y1, x2=x2, y2=y2,
                        p_kps       = p_kps,
                        p_scores    = p_scores,
                        heatmap     = heatmap,
    )

                # Blend main heatmap overlay
                overlay_bgr = cv2.cvtColor(
                    np.array(pil_overlay.convert("RGB")), cv2.COLOR_RGB2BGR
                )
                frame_bgr = cv2.addWeighted(overlay_bgr, 0.6, frame_bgr, 0.4, 0)

        # Write all outputs
        writer.write(frame_bgr)
        for sid, w in solo_writers.items():
            w.write(solo_frames[sid])

        pbar.update(1)

cap.release()
writer.release()
writer_1.release()
writer_2.release()

# ── Metadata ──────────────────────────────────────────────────────────────────
META_FILE.write_text(f"""detector  : {DETECTOR}
tracker   : {TRACKER_TYPE}
reid      : {REID}
gazelle   : gazelle_dinov2_vitl14_inout_childplay
source    : {INPUT_VIDEO}
date      : {datetime.now().isoformat()}
device    : {DEVICE}
output    : {OUTPUT_VIDEO}
output_id1: {OUTPUT_VIDEO_1}
output_id2: {OUTPUT_VIDEO_2}
""")

print(f"\nVideo     → {OUTPUT_VIDEO}")
print(f"Video ID1 → {OUTPUT_VIDEO_1}")
print(f"Video ID2 → {OUTPUT_VIDEO_2}")
print(f"Meta      → {META_FILE}")
