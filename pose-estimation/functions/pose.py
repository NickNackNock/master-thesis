import cv2
import numpy as np
from tqdm import tqdm
from rtmlib import draw_skeleton

# REMINDER: maybe pass this in as an argument from main.py
HEAD_KP_INDICES = [0, 1, 2, 3, 4]  # nose, eyes, ears

PERSON_BOX_COLOR = (0,  255,   0)   # green  — full body
HEAD_BOX_COLOR = (255, 0, 0)     # blue — head

# ----- BOUNDING BOXES FUNCTIONS -----
def head_bbox_from_pose(keypoints, scores, score_thresh=0.1, expansion=1.6):
    """Returns [x1, y1, x2, y2] derived from head keypoints."""
    head_kps    = keypoints[HEAD_KP_INDICES]   
    head_scores = scores[HEAD_KP_INDICES]       

    visible = head_scores > score_thresh
    if visible.sum() < 2:
        return None

    pts = head_kps[visible]
    x1, y1 = pts.min(axis=0)
    x2, y2 = pts.max(axis=0)

    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    hw, hh = (x2 - x1) / 2 * expansion, (y2 - y1) / 2 * expansion
    return np.array([cx - hw, cy - hh, cx + hw, cy + hh])


def pose_to_bbox(keypoints: np.ndarray, expansion: float = 1.25) -> np.ndarray:
    """Get bounding box from keypoints."""
    x = keypoints[:, 0]
    y = keypoints[:, 1]
    bbox = np.array([x.min(), y.min(), x.max(), y.max()])
    center = np.array([bbox[0] + bbox[2], bbox[1] + bbox[3]]) / 2
    bbox = np.concatenate([
        center - (center - bbox[:2]) * expansion,
        center + (bbox[2:] - center) * expansion
    ])
    return bbox


# ----- INFO INPUT VIDEO -----
def setup_input_video(input_video):
    
    cap = cv2.VideoCapture(input_video)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    return cap, total, fps , width, height


# This function runs pose estimation on the input video and saves the output with drawn skeletons
def run_pose_estimation(input_video, output_video, pose_model, bbox_body = None, bbox_head = None):
    
    cap, total, fps , width, height = setup_input_video(input_video)
    writer = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    with tqdm(total=total, unit='frame', desc='Processing') as pbar:
        while True:
            ret, img = cap.read()
            if not ret:
                break

            # 1. Run detections
            keypoints, scores = pose_model(img)

            # 2. Draw the standard body skeleton from RTMO
            img = draw_skeleton(img, keypoints, scores, kpt_thr=0.5)

            # 3. Draw bounding boxes
            for person_kps, person_scores in zip(keypoints, scores):
                
                if  bbox_body == True:
                    bbox = pose_to_bbox(person_kps[:, :2])
                    x1, y1, x2, y2 = bbox.astype(int)
                    cv2.rectangle(img, (x1, y1), (x2, y2), PERSON_BOX_COLOR, 2)
                
                if bbox_head == True:
                    head_bbox = head_bbox_from_pose(person_kps[:, :2], person_scores)
                    if head_bbox is not None:
                        hx1, hy1, hx2, hy2 = head_bbox.astype(int)
                        cv2.rectangle(img, (hx1, hy1), (hx2, hy2), HEAD_BOX_COLOR, 2)

            # 4. Write the final frame once
            writer.write(img)  
            pbar.update(1)

    cap.release()
    writer.release()
    print(f'Saved to {output_video}')


def run_pose_estimation_with_tracking(input_video, output_video, pose_model, tracker):
    
    cap, total, fps , width, height = setup_input_video(input_video)
    writer = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    with tqdm(total=total, unit="frame", desc="Tracking") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 1. Pose estimation
            keypoints, scores = pose_model(frame)

            # 2. Draw skeletons
            frame = draw_skeleton(frame, keypoints, scores, kpt_thr=0.5)

            # 3. Build detections for tracker  [x1,y1,x2,y2,conf,cls]
            dets_list    = []
            pose_mapping = {}   # det_index → (keypoints, scores)

            for i, (p_kps, p_scores) in enumerate(zip(keypoints, scores)):

                bbox     = pose_to_bbox(p_kps[:, :2])
                mean_conf = float(p_scores.mean())
                dets_list.append([bbox[0], bbox[1], bbox[2], bbox[3], mean_conf, 0.0])
                pose_mapping[len(dets_list) - 1] = (p_kps, p_scores)

            dets = np.array(dets_list) if dets_list else np.empty((0, 6))

            # 4. Update tracker  → [x1,y1,x2,y2, id, conf, cls, det_ind]
            tracks = tracker.update(dets, frame)

            # 5. Draw results
            if tracks is not None and len(tracks):
                for track in tracks:
                    x1, y1, x2, y2, track_id, conf, cls, ind = track
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    track_id        = int(track_id)
                    ind             = int(ind)

                    # Person box + ID
                    cv2.rectangle(frame, (x1, y1), (x2, y2), PERSON_BOX_COLOR, 1)
                    cv2.putText(
                        frame, f"ID {track_id}",
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, PERSON_BOX_COLOR, 2,
                    )

                    if ind in pose_mapping:
                        p_kps, p_scores = pose_mapping[ind]

                        # Head bounding box
                        head_bbox = head_bbox_from_pose(p_kps[:, :2], p_scores)
                        if head_bbox is not None:
                            hx1, hy1, hx2, hy2 = head_bbox.astype(int)
                            cv2.rectangle(frame, (hx1, hy1), (hx2, hy2), HEAD_BOX_COLOR, 2)
                            cv2.putText(
                                frame, f"Head {track_id}",
                                (hx1, hy1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, HEAD_BOX_COLOR, 1,
    )

                        # Face keypoints (nose, eyes, ears) with track ID label
                        

            writer.write(frame)
            pbar.update(1)

    cap.release()
    writer.release()
