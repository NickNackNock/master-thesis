import cv2
import pose
from filtering_pose import filter_all_people, draw_filtered_video
from retinaFaceGazeLLE import run_gaze_from_pose_xlsx
import os
import sys
from pathlib import Path

import configuration as config

def run_pose_menu_option(input_video, output_folder_raw, output_folder_filt):
    """
    Menu option 1: 
    Raw RTMO skeleton (no tracking) + tracked run with xlsx export
    """
    print("Loading pose model and tracker...")
    pose_model = config.load_pose_model()
    tracker    = config.load_tracker()

    output_files = config.file_names(output_folder_raw, output_folder_filt)
    output_video_pose_raw = output_files["output_video_pose_raw"]
    output_video_poseTracking_raw = output_files["output_video_poseTracking_raw"]
    output_xlsx_pose_raw = output_files["output_xlsx_pose_raw"]

    thesholds = config.defined_thresholds()
    VISUAL_TH = thesholds["visualization"]
    
    # Basic RTMO, no tracking (skeleton only, fastest)
    print("\nRunning raw pose estimation (no tracking)...")

    pose.run_pose_estimation(
        input_video, 
        str(output_video_pose_raw),
        pose_model, 
        bbox_body = True,
        kpt_thr = VISUAL_TH,
    )

    # Pose + tracking IDs + xlsx keypoint export
    print("\nRunning pose estimation with tracking + xlsx export...")

    pose.run_pose_estimation_with_tracking(
        input_video, 
        str(output_video_poseTracking_raw),
        pose_model, tracker,
        save_xlsx = str(output_xlsx_pose_raw),
        kpt_thr = VISUAL_TH,
    )


def run_filter_menu_option(input_video,output_folder_raw, output_folder_filt):
    """
    Menu option 2: 
    Filter the raw pose xlsx and re-draw the video from filtered keypoints.
    """
    # Read fps from the source video so the Butterworth cutoff is correct
    fps = cv2.VideoCapture(input_video).get(cv2.CAP_PROP_FPS)
    print(f"Source video fps: {fps:.2f}")

    output_files = config.file_names(output_folder_raw, output_folder_filt)
    output_xlsx_pose_raw = output_files["output_xlsx_pose_raw"]
    output_xlsx_pose_filt = output_files["output_xlsx_pose_filt"]
    output_video_pose_filt = output_files["output_video_pose_filt"]

    thesholds = config.defined_thresholds()
    VISUAL_TH = thesholds["visualization"]

    print(f"Filtering pose data from: {output_xlsx_pose_raw}")

    filtered_sheets = filter_all_people(
        xlsx_path = output_xlsx_pose_raw,
        out_path = output_xlsx_pose_filt,
        window_len = 11,      # Savgol window. Must be odd. Higher = smoother. 11 = 0.4s at 25fps.
        polyorder = 3,        # Polynomial order. 2 or 3 is best for human motion.
        min_seg_len = 3       # Drop any valid detection streaks shorter than this many frames.
    )

    print("\nDrawing filtered pose video...")
    draw_filtered_video(
        input_video_path =   input_video,
        output_video_path = output_video_pose_filt,
        person_dfs = filtered_sheets,
        kpt_thr = VISUAL_TH
    )


def run_gaze_menu_option(input_video, output_folder_raw, output_folder_filt):
    """
    Menu option 3: 
    Gaze estimation from the pre-computed filtered pose xlsx.
    so that noisy raw keypoints don't confuse Gazelle's
    head-bbox input
    """
    output_files = config.file_names(output_folder_raw, output_folder_filt)
    output_xlsx_pose_filt = output_files["output_xlsx_pose_filt"]
    output_video_gaze_filt = output_files["output_video_gaze_filt"]

    # Uses the filtered xlsx  — no pose model or tracker needed at runtime
    print("Loading Gazelle...")
    gazelle_vars = config.load_gaze_models()
    thresholds = config.defined_thresholds()

    print(f"Running gaze estimation from: {output_xlsx_pose_filt}")
    run_gaze_from_pose_xlsx(
        input_video       = input_video,
        output_video      = str(output_video_gaze_filt),
        pose_xlsx         = str(output_xlsx_pose_filt),
        gazelle_variables = gazelle_vars,
        thresholds        = thresholds,
    )

def change_file(data_folder, current_dir, output_folder):

    while True:
        print(f"\nCurrent target directory: {current_dir}")
        choice = input("Do you want to change folder? [Y/n] ('q' to quit): ").strip().lower()

        if choice == "q":
            print("Program Aborted.")
            sys.exit(0)

        # If they want to change the folder, update `current_dir`
        elif choice == "y" or choice == "":
            if data_folder.exists():
                # Show only directories
                dirs = os.listdir(data_folder)
                print(f"\nAvailable directories: {dirs}")
            else:
                print(f"Data folder '{data_folder}' does not exist.")
                continue

            dir_name = input("Insert the directory name you want to use: ").strip()
            target_dir = data_folder / dir_name
            
            if not target_dir.is_dir():
                print(f"Directory '{dir_name}' not found. Try again.\n")
                continue
            
            # Successfully chose a new directory
            current_dir = target_dir
            print(f"Directory changed to: {current_dir}")

        elif choice != "n":
            print("Invalid choice. Please enter 'y', 'n', or 'q'.\n")
            continue
        
        
        # Change the actual OS working directory to the chosen folder
        os.chdir(current_dir)

        while True:
            # List only files (ignore sub-folders) for a cleaner prompt
            files = [f.name for f in current_dir.iterdir() if f.is_file()]
            print(f"\nFiles in {current_dir.name}: {files}")

            file_name = input("\nInsert the file name you want to run ('b' to go back to folder selection): ").strip()
            
            if file_name.lower() == 'b':
                break # Breaks out of the file loop, goes back to the folder prompt
                
            target_file = current_dir / file_name

            if not target_file.is_file():
                print(f"File '{file_name}' not found")
                continue # Stays in the file selection loop
            
            print(f"\nFile successfully selected: {target_file}")

            # After everything is selected, create the output folder and subfolders for raw and filtered outputs
            output_folder.mkdir(parents = True, exist_ok = True)
            output_folder_raw, output_folder_filt = config.check_output_folder(output_folder, str(current_dir)[-5:]) 

            return target_file, output_folder_raw, output_folder_filt

        

def main():
    DEVICE = config.device_availability()
    print(f"\nUsing device: {DEVICE}\n")

    """
    Directory stuctured like follows:
        project_folder  -> src (main.py + functions)
                        -> data (PCI_n -> videos)
                        -> output (raw, filtered -> videos)                   
    """
    project_folder = Path(r"/home/neurolab/thesisProject")
    src_folder = project_folder / "src" / "Pose-Gaze-4"
    data_folder = project_folder / "data"
    output_folder = project_folder / "outputs"
    
    # This variable is to track selected fodlder
    current_dir = data_folder / "PCI_1" 

    # Change to source folder
    try:
        os.chdir(src_folder)
        print(f"Moved to the following folder: {src_folder}")
    except FileNotFoundError:
        print(f"Error: Source folder '{src_folder}' does not exist.")
        sys.exit(1)

    # If source folder is found, check if data is a folder that exists
    # If it's not found it needs to be stopped and checked
    if not data_folder.exists():
        print(f"Error: data folder '{data_folder}' does not exist.")
        sys.exit(1)

    # Menù of choices if I want to change file/directory 
    target_file, output_folder_raw, output_folder_filt = change_file(data_folder,current_dir, output_folder)

   
    while True:

        print("What do you want to run?")
        print("1. Pose Estimation  (raw RTMO skeleton  +  tracking with xlsx export)")
        print("2. Filter Pose Data (xlsx filtered xlsx  +  filtered video)")
        print("3. Gaze Estimation  (uses pre-computed filtered pose xlsx skips pose inference)")
        print("4. Change File")
        print("q. Quit")

        choice = input("Enter the number of your choice: ").strip().lower()
        match choice:
            case "1":
                run_pose_menu_option(target_file, output_folder_raw, output_folder_filt)
            case "2":
                run_filter_menu_option(target_file, output_folder_raw, output_folder_filt)
            case "3":
                run_gaze_menu_option(target_file, output_folder_raw, output_folder_filt)
            case "4":
                target_file, output_folder_raw, output_folder_filt = change_file(data_folder,current_dir, output_folder)
            case "q" | "quit" | "exit":
                break
            case _:
                print("Invalid choice. Please enter 1, 2, 3, 4 or q.")


if __name__ == "__main__":
    main()
