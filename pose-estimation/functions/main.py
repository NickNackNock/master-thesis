import pose
import gaze

from configuration import (DEVICE, INPUT_VIDEO, OUTPUT_POSE, OUTPUT_GAZE, pose_model, gazelle_variables, tracker)


def main():
    print(f"\nUsing device: {DEVICE} \n")            

    print("What  do you want to run?")
    print("1. Pose Estimation")
    print("2. Pose Estimation + Tracking ID")
    print("3. Gaze Estimation (Pose + Tracking ID)")

    choice = input("Enter the number of your choice: ")
    match choice:

        case "1":
            print("Running Pose Estimation...")
            pose.run_pose_estimation(INPUT_VIDEO, OUTPUT_POSE, pose_model, bbox_body = True, bbox_head = False)

        case "2":
            print("Running Pose Estimation + Tracking ID...")
            pose.run_pose_estimation_with_tracking(INPUT_VIDEO, OUTPUT_POSE, pose_model, tracker)

        case "3":
            print("Running Gaze Estimation (Pose + Tracking ID)...")
            gaze.run_gaze_estimation(INPUT_VIDEO, str(OUTPUT_GAZE), pose_model, gazelle_variables)

        case _:
            print("Invalid choice. Please enter 1, 2, or 3.")


if __name__ == "__main__":
    main()
