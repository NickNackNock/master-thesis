import cv2
import pose
import gaze
from filtering_pose import filter_all_people, draw_filtered_video
from stereo_reconstruction import reconstruct_3d

from configuration import (
    DEVICE,
    INPUT_VIDEO,
    OUTPUT_RAW_POSE_VIDEO,
    OUTPUT_RAW_POSE_TRACKED,
    OUTPUT_RAW_POSE_XLSX,
    OUTPUT_FILT_POSE_VIDEO,
    OUTPUT_FILT_POSE_XLSX,
    OUTPUT_RAW_GAZE_STEM,
    STEREO_CALIB_FILE,
    load_pose_model,
    load_tracker,
    load_gaze_models,
)


def main():
    print(f"\nUsing device: {DEVICE}\n")

    print("What do you want to run?")
    print("1. Pose Estimation  (raw RTMO skeleton  +  tracking with xlsx export)")
    print("2. Filter Pose Data (xlsx → filtered xlsx  +  filtered video)")
    print("3. Gaze Estimation  (uses pre-computed filtered pose xlsx — skips pose inference)")
    print("4. 3D Stereo Reconstruction")

    choice = input("Enter the number of your choice: ")
    match choice:

        case "1":
            print("Loading pose model and tracker...")
            pose_model = load_pose_model()
            tracker    = load_tracker()

            # Basic RTMO, no tracking (skeleton only, fastest)
            print("\nRunning raw pose estimation (no tracking)...")
            pose.run_pose_estimation(
                INPUT_VIDEO, str(OUTPUT_RAW_POSE_VIDEO),
                pose_model, bbox_body=True, bbox_head=False,
            )

            # Pose + tracking IDs + xlsx keypoint export
            print("\nRunning pose estimation with tracking + xlsx export...")
            pose.run_pose_estimation_with_tracking(
                INPUT_VIDEO, str(OUTPUT_RAW_POSE_TRACKED),
                pose_model, tracker,
                save_xlsx=str(OUTPUT_RAW_POSE_XLSX),
            )

        case "2":
            # Read fps from the source video so the Butterworth cutoff is correct
            fps = cv2.VideoCapture(INPUT_VIDEO).get(cv2.CAP_PROP_FPS)
            print(f"Source video fps: {fps:.2f}")
            print(f"Filtering pose data from: {OUTPUT_RAW_POSE_XLSX}")

            filtered_sheets = filter_all_people(
                xlsx_path    = str(OUTPUT_RAW_POSE_XLSX),
                out_path     = str(OUTPUT_FILT_POSE_XLSX),
                fps          = fps,
                cutoff       = 6.0,     # Hz — lower = smoother but more lag; 4-8 Hz typical for human movement
                order        = 2,       # filtfilt makes this effectively 4th-order, zero-phase
                score_thresh = 0.8,
            )

            print("\nDrawing filtered pose video...")
            draw_filtered_video(
                INPUT_VIDEO, str(OUTPUT_FILT_POSE_VIDEO), filtered_sheets,
            )

        case "3":
            # Uses the filtered xlsx so that noisy raw keypoints don't confuse Gazelle's
            # head-bbox input — no pose model or tracker needed at runtime
            print("Loading Gazelle...")
            gazelle_vars = load_gaze_models(with_tracker=False)

            print(f"Running gaze estimation from: {OUTPUT_FILT_POSE_XLSX}")
            gaze.run_gaze_from_pose_xlsx(
                input_video       = INPUT_VIDEO,
                output_video      = str(OUTPUT_RAW_GAZE_STEM),
                pose_xlsx         = str(OUTPUT_FILT_POSE_XLSX),
                gazelle_variables = gazelle_vars,
            )

        
        case "4":
            # 3D stereo reconstruction from pre-computed filtered keypoints.
            # No GPU models needed, purely geometry.
            
            print(f"Reconstructing 3D poses from:")
            print(f"  cam1: {OUTPUT_FILT_POSE_XLSX}")
            print(f"  cam2: {OUTPUT_FILT_POSE_XLSX_CAM2}")
            print(f"  calib: {STEREO_CALIB_FILE}")
            if STEREO_FRAME_OFFSET != 0:
                print(f"  frame offset (cam2 lag): {STEREO_FRAME_OFFSET:+d} frames")

            reconstruct_3d(
                xlsx_cam1    = str(OUTPUT_FILT_POSE_XLSX),
                xlsx_cam2    = str(OUTPUT_FILT_POSE_XLSX_CAM2),
                calib_path   = STEREO_CALIB_FILE,
                out_path     = str(OUTPUT_3D_XLSX),
                frame_offset = STEREO_FRAME_OFFSET,
            )

        case _:
            print("Invalid choice. Please enter 1, 2, or 3.")


if __name__ == "__main__":
    main()
