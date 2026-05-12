import cv2
import numpy as np
from tqdm import tqdm
from rtmlib import Wholebody, draw_skeleton, RTMPose, ViTPose

def pose_to_bbox(keypoints: np.ndarray, expansion: float = 1.25) -> np.ndarray:
    """Get bounding box from keypoints.

    Args:
        keypoints (np.ndarray): Keypoints of person.
        expansion (float): Expansion ratio of bounding box.

    Returns:
        np.ndarray: Bounding box of person.
    """
    x = keypoints[:, 0]
    y = keypoints[:, 1]
    bbox = np.array([x.min(), y.min(), x.max(), y.max()])
    center = np.array([bbox[0] + bbox[2], bbox[1] + bbox[3]]) / 2
    bbox = np.concatenate([
        center - (center - bbox[:2]) * expansion,
        center + (bbox[2:] - center) * expansion
    ])
    return bbox

INPUT_VIDEO  = "C:/Lorem/Ipsum/"
OUTPUT_VIDEO = "C:/Lorem/Ipsum/Output.mp4"

device = 'cpu'              # cpu, cuda, mps
backend = 'onnxruntime'     # opencv, onnxruntime, openvino
openpose_skeleton = False   # True for openpose-style, False for mmpose-style

'''
pose_model = Wholebody(to_openpose=openpose_skeleton,
                      mode='lightweight',  # 'performance', 'lightweight', 'balanced'
                      backend=backend, device=device)

'''
from rtmlib import PoseTracker, Custom, RTMO
from functools import partial

# Best model for my activity
pose_model = RTMO(
    onnx_model='https://download.openmmlab.com/mmpose/v1/projects/rtmo/onnx_sdk/rtmo-m_16xb16-600e_body7-640x640-39e78cc4_20231211.zip',  # or let it auto-download
    backend='onnxruntime',
    device='cpu'
)
'''
custom = partial(
    Custom,
    det_class='YOLOX',
    det='https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_darknet.onnx',
    det_input_size=(640, 640),
    pose_class='RTMO',
    pose='https://download.openmmlab.com/mmpose/v1/projects/rtmo/onnx_sdk/rtmo-m_16xb16-600e_body7-640x640-39e78cc4_20231211.zip',
    pose_input_size=(640, 640),   # ← width, height — must match model name
    backend='onnxruntime',
    device='cpu'
)
pose_model = PoseTracker(custom, det_frequency=5, mode='balanced', tracking = False)
'''

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

        keypoints, scores = pose_model(img)

        # if you want to use black background instead of original image,
        # img = np.zeros(img.shape, dtype=np.uint8)
        img = draw_skeleton(img, keypoints, scores, kpt_thr=0.75)

        #SCORE_THRESHOLD = 0.75
        for person_kps, person_scores in zip(keypoints, scores):
            #if person_scores.mean() < SCORE_THRESHOLD:
            #    continue  # skip low-confidence detections
            
            bbox = pose_to_bbox(person_kps[:, :2])
            x1, y1, x2, y2 = bbox.astype(int)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        writer.write(img)  
        pbar.update(1)

cap.release()
writer.release()
print(f'Saved to {OUTPUT_VIDEO}')