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
from datetime import datetime

# ── Config ────────────────────────────────────────────────────────────────────
INPUT_VIDEO  = "/home/neurolab/thesisProject/data/videos/output_ultra_cut.mp4"
OUTPUT_DIR   = Path("./output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

GAZELLE_CKPT = "/home/neurolab/repositories/gazelle/checkpoints/gazelle_dinov2_vitl14_inout_childplay.pt"

DETECTOR     = "yolov8l"
TRACKER_TYPE = "botsort"
REID         = "clip_market1501"

OUTPUT_VIDEO = OUTPUT_DIR / f"gaze_{DETECTOR}_{TRACKER_TYPE}_{REID}_NNESIMO.mp4"
META_FILE    = OUTPUT_DIR / f"gaze_{DETECTOR}_{TRACKER_TYPE}_{REID}_NNESIMO.txt"

DEVICE  = "cuda:0" if torch.cuda.is_available() else "cpu"
BACKEND = "onnxruntime"

# Drawing settings
INOUT_THRESH     = 0.5
SCORE_THRESH_KP  = 0.4
SCORE_THRESH_DET = 0.3
HEAD_EXPANSION   = 1.6
HEATMAP_ALPHA    = 90       # 0-255, transparency of gaze overlay

# One colour per tracked ID (cycles if more than 5 people)
ID_COLORS = [
    (0,   255,   0),    # green
    (255,  50,  50),    # red
    (0,   200, 255),    # cyan
    (255,   0, 200),    # magenta
    (255, 255,   0),    # yellow
]

# RTMO face keypoint indices (MMPose format)
FACE_KP_INDICES = [0, 1, 2, 3, 4]  # nose, left_eye, right_eye, left_ear, right_ear


# ── Helpers ───────────────────────────────────────────────────────────────────

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
    """Returns [x1,y1,x2,y2] pixel head box, or None if not enough visible kps."""
    head_kps    = keypoints[FACE_KP_INDICES]
    head_scores = scores[FACE_KP_INDICES]
    visible     = head_scores > score_thresh

    if visible.sum() < 2:
        return None

    pts        = head_kps[visible]
    x1, y1     = pts.min(axis=0)
    x2, y2     = pts.max(axis=0)
    cx, cy     = (x1 + x2) / 2, (y1 + y2) / 2
    hw, hh     = (x2 - x1) / 2 * expansion, (y2 - y1) / 2 * expansion
    return np.array([cx - hw, cy - hh, cx + hw, cy + hh])


def head_bbox_normalized(head_bbox_px: np.ndarray, width: int, height: int) -> np.ndarray:
    """Convert pixel head bbox to normalized [0,1] for Gazelle."""
    return head_bbox_px / np.array([width, height, width, height])


def visualize_heatmap(
    pil_image:    Image.Image,
    heatmap:      torch.Tensor,
    norm_bbox:    np.ndarray | None = None,
    inout_score:  float | None      = None,
    color:        str               = "lime",
    alpha:        int               = HEATMAP_ALPHA,
) -> Image.Image:
    """Overlay a Gazelle heatmap on a PIL image. Returns RGBA composite."""
    if isinstance(heatmap, torch.Tensor):
        heatmap = heatmap.detach().cpu().numpy()

    heatmap_img = Image.fromarray((heatmap * 255).astype(np.uint8)).resize(
        pil_image.size, Image.Resampling.BILINEAR
    )
    heatmap_colored = plt.cm.jet(np.array(heatmap_img) / 255.0)
    heatmap_colored = (heatmap_colored[:, :, :3] * 255).astype(np.uint8)
    heatmap_rgba    = Image.fromarray(heatmap_colored).convert("RGBA")
    heatmap_rgba.putalpha(alpha)
    overlay = Image.alpha_composite(pil_image.convert("RGBA"), heatmap_rgba)

    if norm_bbox is not None:
        w, h   = pil_image.size
        xmin, ymin, xmax, ymax = norm_bbox
        draw   = ImageDraw.Draw(overlay)
        bw     = int(min(w, h) * 0.008)
        draw.rectangle(
            [xmin * w, ymin * h, xmax * w, ymax * h],
            outline=color, width=bw,
        )

        # Draw gaze target dot + line if person is looking in-frame
        if inout_score is not None and inout_score > INOUT_THRESH:
            hm_np        = heatmap if isinstance(heatmap, np.ndarray) else heatmap.cpu().numpy()
            max_idx      = np.unravel_index(np.argmax(hm_np), hm_np.shape)
            gaze_x       = max_idx[1] / hm_np.shape[1] * w
            gaze_y       = max_idx[0] / hm_np.shape[0] * h
            face_cx      = ((xmin + xmax) / 2) * w
            face_cy      = ((ymin + ymax) / 2) * h
            r            = int(min(w, h) * 0.008)
            draw.ellipse([(gaze_x - r, gaze_y - r), (gaze_x + r, gaze_y + r)], fill=color)
            draw.line([(face_cx, face_cy), (gaze_x, gaze_y)],
                      fill=color, width=int(min(w, h) * 0.004))

        if inout_score is not None:
            font_size = int(min(w, h) * 0.04)
            draw.text(
                (xmin * w, ymax * h + int(h * 0.01)),
                f"gaze: {inout_score:.2f}",
                fill=color,
                font=ImageFont.load_default(size=font_size),
            )

    return overlay


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

writer = cv2.VideoWriter(
    str(OUTPUT_VIDEO), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
)


# ── Main loop ─────────────────────────────────────────────────────────────────

with tqdm(total=total, unit="frame", desc="Tracking + Gaze") as pbar:
    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        pil_img = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))

        # 1. Pose estimation
        keypoints, scores = pose_model(frame_bgr)

        # 2. Draw body skeletons
        frame_bgr = draw_skeleton(frame_bgr, keypoints, scores, kpt_thr=SCORE_THRESH_KP)

        # 3. Build detections for tracker [x1,y1,x2,y2,conf,cls]
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

        # 4. Per-track: draw boxes, run Gazelle, overlay heatmap
        if tracks is not None and len(tracks):

            # Collect Gazelle inputs for all tracks in one batch
            gazelle_bboxes = []   # normalized bboxes for Gazelle
            track_meta     = []   # (track_id, head_bbox_px) per entry

            for track in tracks:
                x1, y1, x2, y2, track_id, conf, cls, ind = track
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                track_id        = int(track_id)
                ind             = int(ind)
                color           = id_color(track_id)

                # Person box
                cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame_bgr, f"ID {track_id}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                if ind not in pose_mapping:
                    track_meta.append(None)
                    continue

                p_kps, p_scores = pose_mapping[ind]

                # Face keypoints
                for kp_idx in FACE_KP_INDICES:
                    if p_scores[kp_idx] > SCORE_THRESH_KP:
                        cx_kp = int(p_kps[kp_idx][0])
                        cy_kp = int(p_kps[kp_idx][1])
                        cv2.circle(frame_bgr, (cx_kp, cy_kp), 5, color, -1)

                # Head bbox for Gazelle
                head_px = head_bbox_from_pose(p_kps[:, :2], p_scores)
                if head_px is None:
                    track_meta.append(None)
                    continue

                # Draw head box
                hx1, hy1, hx2, hy2 = head_px.astype(int)
                cv2.rectangle(frame_bgr, (hx1, hy1), (hx2, hy2), color, 1)

                norm_bbox = head_bbox_normalized(head_px, width, height)
                norm_bbox = np.clip(norm_bbox, 0.0, 1.0)

                gazelle_bboxes.append(norm_bbox)
                track_meta.append((track_id, norm_bbox))

            # 5. Run Gazelle in one batch over all valid faces
            valid_bboxes = [m[1] for m in track_meta if m is not None]

            if valid_bboxes:
                img_tensor  = gazelle_transform(pil_img).unsqueeze(0).to(DEVICE)
                model_input = {
                    "images":  img_tensor,
                    "bboxes": [valid_bboxes],   # one image, N faces
                }

                with torch.no_grad():
                    gaze_out = gazelle(model_input)

                heatmaps    = gaze_out["heatmap"][0]   # (N, 64, 64)
                inout_scores = gaze_out["inout"]        # (1, N) or None

                # 6. Overlay heatmaps onto pil_img, one per person
                pil_overlay = pil_img.copy()
                valid_idx   = 0

                for meta in track_meta:
                    if meta is None:
                        continue

                    track_id, norm_bbox = meta
                    color_name = ["lime", "tomato", "cyan", "fuchsia", "yellow"][track_id % 5]
                    inout_score = (
                        inout_scores[0][valid_idx].item()
                        if inout_scores is not None else None
                    )

                    """
                    pil_overlay = visualize_heatmap(
                        pil_overlay,
                        heatmaps[valid_idx],
                        norm_bbox=norm_bbox,
                        inout_score=inout_score,
                        color=color_name,
                        alpha=HEATMAP_ALPHA,
                    )
                    """
                    valid_idx += 1

                # Blend heatmap overlay back onto the BGR frame
                overlay_bgr = cv2.cvtColor(
                    np.array(pil_overlay.convert("RGB")), cv2.COLOR_RGB2BGR
                )
                # Alpha-blend: keep skeleton/boxes from frame_bgr, add heatmap softly
                frame_bgr = cv2.addWeighted(overlay_bgr, 0.6, frame_bgr, 0.4, 0)

        writer.write(frame_bgr)
        pbar.update(1)

cap.release()
writer.release()

# ── Metadata ──────────────────────────────────────────────────────────────────
META_FILE.write_text(f"""detector  : {DETECTOR}
tracker   : {TRACKER_TYPE}
reid      : {REID}
gazelle   : gazelle_dinov2_vitl14_inout_childplay
source    : {INPUT_VIDEO}
date      : {datetime.now().isoformat()}
device    : {DEVICE}
output    : {OUTPUT_VIDEO}
""")

print(f"\nVideo → {OUTPUT_VIDEO}")
print(f"Meta  → {META_FILE}")
