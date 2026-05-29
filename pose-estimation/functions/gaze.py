import cv2
import numpy as np
from PIL import Image
import torch
from tqdm import tqdm
from pathlib import Path
from gazelle.utils import visualize_heatmap

from pose import setup_input_video, pose_to_bbox, head_bbox_from_pose, draw_skeleton
# Expecting a maximum of 3 people in the frame
SOLO_IDS = [1, 2, 3]
ID_COLORS = [
    (0,   255,   0),
    (255,  50,  50),
    (0,   200, 255),
    (255,   0, 200),
    (255, 255,   0),
]

FACE_KP_INDICES = [0, 1, 2, 3, 4]

# Drawing settings
INOUT_THRESH     = 0.5
SCORE_THRESH_KP  = 0.4
SCORE_THRESH_DET = 0.3
HEAD_EXPANSION   = 1.6
HEATMAP_ALPHA    = 90

solo_writers = {}

def draw_person_on_frame(
    frame_clean: np.ndarray,   # ← clean frame, no other skeletons
    pil_img: Image.Image,
    track_id: int,
    x1: int, y1: int, x2: int, y2: int,
    p_kps: np.ndarray,
    p_scores: np.ndarray,
    heatmap: torch.Tensor | None,
) -> np.ndarray:
    color = ID_COLORS[track_id % len(ID_COLORS)]

    # Only this person's skeleton on a clean base
    frame_out = draw_skeleton(
        frame_clean.copy(),
        p_kps[np.newaxis],
        p_scores[np.newaxis],
        kpt_thr=0.5,
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

def run_gaze_estimation(input_video, output_video, pose_model, gazelle_variables):
     
    gazelle = gazelle_variables["gazelle"]
    gazelle_transform = gazelle_variables["gazelle_transform"]
    tracker = gazelle_variables["tracker"]
    DEVICE = gazelle_variables["DEVICE"]

    cap, total, fps , width, height = setup_input_video(input_video)

    #writer =  cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    out      = Path(output_video)
    stem     = out.stem     # e.g. "gaze_yolov8l_botsort_clip_market1501"
    out_dir  = out.parent   # e.g. Path("./output/gaze")

    writer_1 = cv2.VideoWriter(str(out_dir / f"{stem}_ID1.mp4"), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    writer_2 = cv2.VideoWriter(str(out_dir / f"{stem}_ID2.mp4"), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    writer_3 = cv2.VideoWriter(str(out_dir / f"{stem}_ID3.mp4"), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    for name, w in [("ID1", writer_1), ("ID2", writer_2), ("ID3", writer_3)]:
        if not w.isOpened():
            raise RuntimeError(f"VideoWriter failed for {name} — check path and codec")

    solo_writers = {
        1: writer_1,
        2: writer_2,
        3: writer_3,
    }

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
                    color           = ID_COLORS[track_id % len(ID_COLORS)]

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
                        kpt_thr = 0.5,
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


                    norm_bbox = head_px / np.array([width, height, width, height])
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
            #writer.write(frame_bgr)
            for sid, w in solo_writers.items():
                w.write(solo_frames[sid])

            pbar.update(1)

    cap.release()
    #writer.release()
    writer_1.release()
    writer_2.release()
    writer_3.release()
