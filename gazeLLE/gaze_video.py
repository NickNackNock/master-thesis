"""
Gaze Estimation Video Processor — VS Code version
Based on GazeLLE (gazelle_dinov2_vitl14_inout) + RetinaFace
"""
import torch
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from retinaface import RetinaFace
from gazelle.model import get_gazelle_model
import cv2

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Device: " + device)
# Load Gaze-LLE model
model, transform = get_gazelle_model("gazelle_dinov2_vitl14_inout")
model.load_gazelle_state_dict(torch.load(
    "/lorem/ipsum/gazelle/checkpoints/gazelle_dinov2_vitl14_inout.pt",
    weights_only=True
))
model.eval()
model.to(device)


def detect_faces(image):
    width, height = image.size
    resp = RetinaFace.detect_faces(np.array(image))

    # Guard: no faces detected (RetinaFace returns a list or empty dict)
    if not isinstance(resp, dict) or len(resp) == 0:
        print("[WARN] No faces detected.")
        return None, []

    bboxes = [resp[key]['facial_area'] for key in resp.keys()]

    img_tensor = transform(image).unsqueeze(0).to(device)
    norm_bboxes = [[np.array(bbox) / np.array([width, height, width, height]) for bbox in bboxes]]

    model_input = {
        "images": img_tensor,
        "bboxes": norm_bboxes
    }
    return model_input, bboxes


def visualize_heatmap(pil_image, heatmap, norm_bbox=None, inout_score=None):
    if isinstance(heatmap, torch.Tensor):
        heatmap = heatmap.detach().cpu().numpy()

    heatmap_img = Image.fromarray((heatmap * 255).astype(np.uint8)).resize(
        pil_image.size, Image.Resampling.BILINEAR
    )
    heatmap_colored = plt.cm.jet(np.array(heatmap_img) / 255.)
    heatmap_colored = (heatmap_colored[:, :, :3] * 255).astype(np.uint8)
    heatmap_rgba = Image.fromarray(heatmap_colored).convert("RGBA")
    heatmap_rgba.putalpha(90)
    overlay_image = Image.alpha_composite(pil_image.convert("RGBA"), heatmap_rgba)

    if norm_bbox is not None:
        width, height = pil_image.size
        xmin, ymin, xmax, ymax = norm_bbox
        draw = ImageDraw.Draw(overlay_image)
        draw.rectangle(
            [xmin * width, ymin * height, xmax * width, ymax * height],
            outline="lime",
            width=int(min(width, height) * 0.01)
        )
        if inout_score is not None:
            score_val = inout_score.item() if hasattr(inout_score, 'item') else float(inout_score)
            text = f"in-frame: {score_val:.2f}"
            text_x = xmin * width
            text_y = ymax * height + int(height * 0.01)
            draw.text(
                (text_x, text_y), text, fill="lime",
                font=ImageFont.load_default(size=int(min(width, height) * 0.05))
            )
    return overlay_image


def visualize_all(pil_image, heatmaps, bboxes, inout_scores, inout_thresh=0.5):
    colors = ['lime', 'tomato', 'cyan', 'fuchsia', 'yellow']
    overlay_image = pil_image.convert("RGBA")
    draw = ImageDraw.Draw(overlay_image)
    width, height = pil_image.size

    for i in range(len(bboxes)):
        bbox = bboxes[i]
        xmin, ymin, xmax, ymax = bbox
        color = colors[i % len(colors)]
        draw.rectangle([xmin * width, ymin * height, xmax * width, ymax * height], outline=color, width=int(min(width, height) * 0.01))

        if inout_scores is not None:
            inout_score = inout_scores[i]
            text = f"in-frame: {inout_score:.2f}"
            text_width = draw.textlength(text)
            text_height = int(height * 0.01)
            text_x = xmin * width
            text_y = ymax * height + text_height
            draw.text((text_x, text_y), text, fill=color, font=ImageFont.load_default(size=int(min(width, height) * 0.05)))

        if inout_scores is not None and inout_score > inout_thresh:
            heatmap = heatmaps[i]
            heatmap_np = heatmap.detach().cpu().numpy()
            max_index = np.unravel_index(np.argmax(heatmap_np), heatmap_np.shape)
            gaze_target_x = max_index[1] / heatmap_np.shape[1] * width
            gaze_target_y = max_index[0] / heatmap_np.shape[0] * height
            bbox_center_x = ((xmin + xmax) / 2) * width
            bbox_center_y = ((ymin + ymax) / 2) * height

            draw.ellipse([(gaze_target_x-5, gaze_target_y-5), (gaze_target_x+5, gaze_target_y+5)], fill=color, width=int(0.005*min(width, height)))
            draw.line([(bbox_center_x, bbox_center_y), (gaze_target_x, gaze_target_y)], fill=color, width=int(0.005*min(width, height)))

    return overlay_image

def run_video():

    video = "/path/to/video/output_ultra_cut.mp4"
    #video = "cut.mp4"
    output_video = "gaze_output_LvideoATT.mp4"
    
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video}")
    
    fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    last_viz = None          # reuse previous heatmap on skipped frames
    frame_idx = 0

    print(f"Processing {total} frames …")

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        pil_img = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        
        #---------------------------------------------------------
        model_input, bboxes = detect_faces(pil_img)
        
        if model_input is None:
            print("No faces found, nothing to visualize.")

        else:
            with torch.no_grad():
                output = model(model_input)


            heatmap = output["heatmap"][0][0]    # [64, 64]
            inout   = output["inout"]

            
            if inout is not None:
                score = inout[0][0].item()
                in_frame = score > 0.5
            else:
                score, in_frame = None, True

            num_faces = len(bboxes)
            # build per-face inout scores list
            inout_scores = [inout[0][i].item() for i in range(num_faces)] if inout is not None else None

            viz_pil = visualize_all(pil_img, output["heatmap"][0], model_input["bboxes"][0], inout_scores)
            last_viz = cv2.cvtColor(np.array(viz_pil.convert("RGB")), cv2.COLOR_RGB2BGR)

            # Overlay in/out score
            if score is not None:
                label = f"In-frame: {score:.2f}"
                color = (0, 200, 0) if in_frame else (0, 0, 220)
                cv2.putText(last_viz, label, (12, 36),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)
                

            writer.write(last_viz)

            frame_idx += 1
            if frame_idx % 50 == 0:
                print(f"  {frame_idx}/{total}")

    cap.release()
    writer.release()
    print(f"Done - {output_video}")



if __name__ == '__main__':
    run_video()

        
