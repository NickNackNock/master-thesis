import cv2
import numpy as np
from tqdm import tqdm
from rtmlib import Wholebody, draw_skeleton, RTMO

HEAD_KP_INDICES = [0, 1, 2, 3, 4]  # nose, eyes, ears

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

INPUT_VIDEO = '/home/neurolab/thesisProject/data/videos/output_ultra_cut.mp4'
OUTPUT_VIDEO = "./output/output_pose_RTMO-L_NEESIMO.mp4"

device = 'cuda'              
backend = 'onnxruntime'     
openpose_skeleton = False   

# Initialize models
pose_model = RTMO(
    onnx_model='https://download.openmmlab.com/mmpose/v1/projects/rtmo/onnx_sdk/rtmo-l_16xb16-600e_body7-640x640-b37118ce_20231211.zip',  
    backend=backend,
    device=device
)

cap    = cv2.VideoCapture(INPUT_VIDEO)
total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps    = cap.get(cv2.CAP_PROP_FPS)
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

writer = cv2.VideoWriter(OUTPUT_VIDEO, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

with tqdm(total=total, unit='frame', desc='Processing') as pbar:
    while True:
        ret, img = cap.read()
        if not ret:
            break

        # 1. Run detections
        keypoints, scores = pose_model(img)

        # 2. Draw the standard body skeleton from RTMO
        img = draw_skeleton(img, keypoints, scores, kpt_thr=0.5)
        

        # 4. Draw bounding boxes
        for person_kps, person_scores in zip(keypoints, scores):
            # Full body box (Green)
            bbox = pose_to_bbox(person_kps[:, :2])
            x1, y1, x2, y2 = bbox.astype(int)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            """
            # Separate Head Box (Blue)
            head_bbox = head_bbox_from_pose(person_kps[:, :2], person_scores)
            if head_bbox is not None:
                hx1, hy1, hx2, hy2 = head_bbox.astype(int)
                cv2.rectangle(img, (hx1, hy1), (hx2, hy2), (255, 0, 0), 2)
            """

        # 5. Write the final frame once
        writer.write(img)  
        pbar.update(1)

cap.release()
writer.release()
print(f'Saved to {OUTPUT_VIDEO}')
