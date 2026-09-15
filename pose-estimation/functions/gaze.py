
import cv2
import numpy as np
from PIL import Image # Python Imaging Library: PIL

import torch

from tqdm import tqdm
from pathlib import Path
from collections import defaultdict, deque
from gazelle.utils import visualize_heatmap
from retinaface import RetinaFace

from pose import setup_input_video, head_bbox_from_pose, draw_skeleton, KEYPOINT_NAMES

# Maximum 3 peoople (even though there should be only two in the scene)
# Maybe I can just ignore multiple people and assign it the ID's I want at priori when I 
# modify the xlsx
SOLO_IDS = [1, 2, 3]
ID_COLORS = [
    (0,   255,   0),
    (255,  50,  50),
    (0,   200, 255),
    (255,   0, 200),
    (255, 255,   0),
]

# Face keypoints from RTMlib
FACE_KP_INDICES = [0, 1, 2, 3, 4]

INOUT_THRESH     = 0.5   
SCORE_THRESH_DET = 0.3   

# Function that calculates the Intersection Over Union between
# - the area found with the Pose Keypoints
# - the face found by RetinaFace
# This is done since the pose reliably distinguish between Parent and Child
# And adds a layer of check when it comes to face detections and tracking
def calculate_iou(box1, box2):
    """Calculates Intersection over Union (IoU) between two bounding boxes [x1, y1, x2, y2]."""
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    iou = intersection_area / float(box1_area + box2_area - intersection_area + 1e-6)
    return iou



# Matches RetinaFace detections to tracked RTMO bodies
# based on Bounding Box IoU (previous function)
# This prevents a body from "hijacking" a nearby face
# that doesn't actually overlap its head.
def assign_face_ids_via_pose(retina_detections, tracked_bodies, min_iou=0.05, det_score_thresh=SCORE_THRESH_DET):
    
    if not isinstance(retina_detections, dict):
        return []

    candidates = [] 

    # Detection of keypoints with RetinaFace
    # TODO: look up RetinaFace.get("score")
    #       It's just the cofidance of how much it thisnks it is a face or
    #       Is it the sum of the confidence of keypoints?
    #       Also maybe increase/change Threshold: 0.3 seems too low
    for face_key, face_data in retina_detections.items():
        det_score = face_data.get("score")
        if det_score is not None and det_score < det_score_thresh:
            continue  

        # IoU between fac_ebox i.e RetinaFace and head_box i.e from Pose 
        face_box = face_data["facial_area"]

        for body in tracked_bodies:
            body_id = body['id']
            head_box = body['head_box']

            iou = calculate_iou(face_box, head_box)
            if iou > min_iou:
                candidates.append((iou, face_key, face_data, body_id))

    # Greedily assign highest IoU overlap first 
    # Since there are more than one person in the scene, we assume
    # it is the highest confidence one
    candidates.sort(key=lambda c: c[0], reverse=True)
    used_faces, used_bodies = set(), set()
    matched_faces = []

    for iou, face_key, face_data, body_id in candidates:
        if face_key in used_faces or body_id in used_bodies:
            continue
        face_data['true_id'] = body_id
        matched_faces.append(face_data)
        used_faces.add(face_key)
        used_bodies.add(body_id)

    return matched_faces

# This function is taken and a bit adapted from the Demo Colab from the
# Github of GazeLLE (https://github.com/fkryan/gazelle)
# The only difference is made by a slight "filter" on the "jittering" 
# of the vector
def draw_gaze_vector(frame, source_xy, heatmap, color, track_id, gaze_history, show_focus=True):
    h, w = frame.shape[:2]
    heatmap_np = heatmap.detach().cpu().numpy()
    max_index = np.unravel_index(np.argmax(heatmap_np), heatmap_np.shape)
    focus_value = float(heatmap_np[max_index])

    raw_gaze_x = max_index[1] / heatmap_np.shape[1] * w
    raw_gaze_y = max_index[0] / heatmap_np.shape[0] * h

    # Doing a "Median Filter" with confidence values above an
    # arbitrary threshold
    # TODO: Maybe check if this threshold makes sense or maybe do it on a 
    #       predefined window 
    if focus_value > 0.30:
        gaze_history[track_id].append((raw_gaze_x, raw_gaze_y))
    elif len(gaze_history[track_id]) == 0:
        gaze_history[track_id].append((raw_gaze_x, raw_gaze_y))

    recent_points = list(gaze_history[track_id])
    med_gaze_x = np.median([p[0] for p in recent_points])
    med_gaze_y = np.median([p[1] for p in recent_points])

    sx, sy = source_xy
    thickness = max(1, int(0.003 * min(w, h)))
    radius    = max(3, int(0.006 * min(w, h)))

    cv2.line(frame, (int(sx), int(sy)), (int(med_gaze_x), int(med_gaze_y)), color, thickness)
    cv2.circle(frame, (int(med_gaze_x), int(med_gaze_y)), radius, color, -1)

    if show_focus:
        cv2.putText(frame, f"focus: {focus_value:.2f}",
                    (int(med_gaze_x) + 8, int(med_gaze_y) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    return frame, focus_value


def load_pose_xlsx(xlsx_path: str, kp_names=KEYPOINT_NAMES) -> dict:
    import pandas as pd
    sheets = pd.read_excel(xlsx_path, sheet_name=None)
    frame_data: dict[int, list] = defaultdict(list)

    for sheet_name, df in sheets.items():
        track_id = int(sheet_name.split("_")[1])
        for _, row in df.iterrows():
            frame_idx = int(row["frame"])
            kps = np.array(
                [[row[f"kp_{n.lower().replace(' ', '_')}_x"],
                  row[f"kp_{n.lower().replace(' ', '_')}_y"]]
                 for n in kp_names],
                dtype=float,
            )
            scores = np.array(
                [row[f"score_{n.lower().replace(' ', '_')}"] for n in kp_names],
                dtype=float,
            )
            frame_data[frame_idx].append((track_id, kps, scores))

    print(f"Loaded pose data for {len(sheets)} person(s) from {xlsx_path}")
    return dict(frame_data)

# TODO: put in configuration.py and maybe pass it throgh main
DEFAULT_THRESHOLDS = {
    "visualization":    0.5,  
    "head_bbox":         0.5,   
    "gaze_body_valid":   0.5,   
    "gaze_face_kp_draw": 0.4,   
}


def run_gaze_from_pose_xlsx(input_video, output_video, pose_xlsx, gazelle_variables,
                             thresholds: dict = None):
    
    thr = thresholds
    
    gazelle           = gazelle_variables["gazelle"]
    gazelle_transform = gazelle_variables["gazelle_transform"]
    DEVICE            = gazelle_variables["DEVICE"]

    frame_data = load_pose_xlsx(pose_xlsx)

    # -- Video parameters --
    cap, total, fps, width, height = setup_input_video(input_video)

    out     = Path(output_video)
    stem    = out.stem    
    out_dir = out.parent  

    writer_1 = cv2.VideoWriter(str(out_dir / f"{stem}_ID1.mp4"), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    writer_2 = cv2.VideoWriter(str(out_dir / f"{stem}_ID2.mp4"), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    writer_3 = cv2.VideoWriter(str(out_dir / f"{stem}_ID3.mp4"), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    solo_writers = {1: writer_1, 2: writer_2, 3: writer_3}
    gaze_history = defaultdict(lambda: deque(maxlen=7))

    # -- Main Loop --
    with tqdm(total=total, unit="frame", desc="Gaze (pre-computed poses)") as pbar:
        frame_idx = 0

        while True:
            ret, frame_bgr = cap.read() 
            if not ret:
                break

            # PIL: Python Imaging Library
            pil_img     = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
            solo_frames = {sid: frame_bgr.copy() for sid in SOLO_IDS}
            people = frame_data.get(frame_idx, [])

            if people:
                gaze_candidates = []


                # TODO: rn it ALWAYS draws skeleton and builds head boxes for gaze 
                #       Do I need to draw the pose?   
                #       I don't think I need it 
                #       Maybe without it I can gain a few frames when it elaborates the video
                for track_id, kps, scores in people:

                    # Draw skeleton based on body validity threshold
                    valid_kp = scores > thr["gaze_body_valid"]
                    draw_scores = np.where(valid_kp, scores, 0.0)

                    # TODO: What am I doing here?
                    #       Is it possible that I'm drawing double the skeletons?
                    frame_bgr = draw_skeleton(
                        frame_bgr, kps[np.newaxis], draw_scores[np.newaxis], kpt_thr=thr["visualization"]
                    )
                    if track_id in SOLO_IDS:
                        solo_frames[track_id] = draw_skeleton(
                            solo_frames[track_id], kps[np.newaxis], draw_scores[np.newaxis], kpt_thr=thr["visualization"]
                        )

                    # Now check if we can build a head bounding box to use for face matching
                    head_px = head_bbox_from_pose(kps, scores, score_thresh=thr["head_bbox"])
                    
                    if head_px is None:
                        # Clear history if the head is lost so we don't draw a stale vector
                        gaze_history[track_id].clear()
                        continue

                    gaze_candidates.append({
                        "id": track_id, 
                        "head_box": head_px.astype(int),
                        "kps": kps,
                        "scores": scores
                    })


                # --- DETECT FACES & MATCH USING IoU ---
                raw_face_dets = RetinaFace.detect_faces(frame_bgr)
                matched_faces = assign_face_ids_via_pose(raw_face_dets, gaze_candidates, min_iou=0.05)
                matched_face_dict = {f['true_id']: f for f in matched_faces}

                # Draw RetinaFace bounding boxes for matched faces
                valid_entries = []
                for face in matched_faces:
                    f_id = face['true_id']
                    f_color = ID_COLORS[f_id % len(ID_COLORS)]
                    fx1, fy1, fx2, fy2 = face['facial_area']

                    cv2.rectangle(frame_bgr, (fx1, fy1), (fx2, fy2), f_color, 2)
                    cv2.putText(frame_bgr, f"Face ID {f_id}", (fx1, fy1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, f_color, 1)

                    for lx, ly in face['landmarks'].values():
                        cv2.circle(frame_bgr, (int(lx), int(ly)), 3, (0, 0, 255), -1)

                    if f_id in solo_frames:
                        cv2.rectangle(solo_frames[f_id], (fx1, fy1), (fx2, fy2), f_color, 2)
                        for lx, ly in face['landmarks'].values():
                            cv2.circle(solo_frames[f_id], (int(lx), int(ly)), 3, (0, 0, 255), -1)

                    # Build the expanded Gazelle bounding box using the RetinaFace coordinates
                    box_w, box_h = fx2 - fx1, fy2 - fy1
                    hx1 = max(0, fx1 - int(box_w * 0.2))
                    hy1 = max(0, fy1 - int(box_h * 0.2))
                    hx2 = min(width, fx2 + int(box_w * 0.2))
                    hy2 = min(height, fy2 + int(box_h * 0.2))

                    norm_bbox = np.array([hx1, hy1, hx2, hy2]) / np.array([width, height, width, height])
                    norm_bbox = np.clip(norm_bbox, 0.0, 1.0)

                    valid_entries.append((f_id, norm_bbox))

                # Clear gaze history for candidates that didn't get a face match this frame
                for cand in gaze_candidates:
                    if cand["id"] not in matched_face_dict:
                        gaze_history[cand["id"]].clear()


                # --- RUN GAZELLE FOR VALID FACES ---
                if valid_entries:
                    img_tensor  = gazelle_transform(pil_img).unsqueeze(0).to(DEVICE)
                    model_input = {
                        "images":  img_tensor,
                        "bboxes": [[m[1] for m in valid_entries]],
                    }

                    with torch.no_grad():
                        gaze_out = gazelle(model_input)

                    heatmaps = gaze_out["heatmap"][0]   
                    inout_scores = gaze_out["inout"][0] if "inout" in gaze_out else None

                    pil_overlay = pil_img.copy()

                    for valid_idx, (track_id, norm_bbox) in enumerate(valid_entries):
                        heatmap = heatmaps[valid_idx]
                        color   = ID_COLORS[track_id % len(ID_COLORS)]

                        is_in_frame = True
                        if inout_scores is not None:
                            inout_prob = float(inout_scores[valid_idx])
                            if inout_prob < thr.get("gaze_inout", INOUT_THRESH):
                                is_in_frame = False

                        hx_c = (norm_bbox[0] + norm_bbox[2]) / 2 * width
                        hy_c = (norm_bbox[1] + norm_bbox[3]) / 2 * height

                        # TODO: Save "focus_value" and make draw_gaze_vector also 
                        #       return the coordinates of source - target
                        #       to save them as well

                        if is_in_frame:
                            pil_overlay = visualize_heatmap(pil_overlay, heatmap)
                            frame_bgr, focus_value = draw_gaze_vector(
                                frame_bgr, (hx_c, hy_c), heatmap, color, track_id, gaze_history
                            )
                        else:
                            gaze_history[track_id].clear()
                            cv2.putText(frame_bgr, "OUT OF FRAME", (int(hx_c) - 40, int(hy_c) - 15),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

                        if track_id in SOLO_IDS:
                            if is_in_frame:
                                pil_with_heatmap = visualize_heatmap(pil_img.copy(), heatmap)
                                heatmap_bgr = cv2.cvtColor(np.array(pil_with_heatmap.convert("RGB")), cv2.COLOR_RGB2BGR)
                                solo_frames[track_id] = cv2.addWeighted(heatmap_bgr, 0.6, solo_frames[track_id], 0.4, 0)

                                solo_frames[track_id], _ = draw_gaze_vector(
                                    solo_frames[track_id], (hx_c, hy_c), heatmap, color, track_id, gaze_history
                                )
                            else:
                                cv2.putText(solo_frames[track_id], "OUT OF FRAME", (int(hx_c) - 40, int(hy_c) - 15),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

                    overlay_bgr = cv2.cvtColor(np.array(pil_overlay.convert("RGB")), cv2.COLOR_RGB2BGR)
                    frame_bgr = cv2.addWeighted(overlay_bgr, 0.6, frame_bgr, 0.4, 0)

            for sid, w in solo_writers.items():
                w.write(solo_frames[sid])

            pbar.update(1)
            frame_idx += 1

    cap.release()
    writer_1.release()
    writer_2.release()
    writer_3.release()
    print(f"Gaze output saved → {out_dir}")
