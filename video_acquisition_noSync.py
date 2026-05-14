import PySpin
import msvcrt
import threading
import subprocess
import queue
import cv2
import os
from datetime import datetime

# Queue before dropping frames
frame_queue = queue.Queue(maxsize=500)

# -- PATHWAYS --
spinview_path = "Lorem/Ipsum"
save_dir = "C:/Recordings"
raw_dir = os.path.join(save_dir, "raw")
color_dir = os.path.join(save_dir, "color")
os.makedirs(raw_dir, exist_ok=True)
os.makedirs(color_dir, exist_ok=True)

# -- Global Variables --
system = None
cam = None
cam_list = None
recording = False
TARGET_FPS = 29.9 # To adjust

def start_spinnaker():
    print("Spinnaker software started...")
    subprocess.Popen([spinview_path])


def start_recording():
    global system, cam, cam_list, recording

    # Set the cameras connected (Just one)
    system = PySpin.System.GetInstance()
    cam_list = system.GetCameras()
    cam = cam_list[0]
    cam.Init()

    # Passages copied from the AcquisitionMultipleThread.py from SpinnakerExamples
    s_node_map = cam.GetTLStreamNodeMap()
    buffer_mode = PySpin.CEnumerationPtr(s_node_map.GetNode('StreamBufferCountMode'))
    buffer_count = PySpin.CIntegerPtr(s_node_map.GetNode('StreamBufferCountManual'))
    if PySpin.IsAvailable(buffer_mode) and PySpin.IsWritable(buffer_mode):
        manual = buffer_mode.GetEntryByName('Manual')
        if PySpin.IsAvailable(manual) and PySpin.IsReadable(manual):
            buffer_mode.SetIntValue(manual.GetValue())
    if PySpin.IsAvailable(buffer_count) and PySpin.IsWritable(buffer_count):
        buffer_count.SetValue(min(buffer_count.GetMax(), 40))

    # Setting Framerate
    cam.AcquisitionFrameRateEnable.SetValue(True)
    cam.AcquisitionFrameRate.SetValue(TARGET_FPS)
    cam.AcquisitionMode.SetValue(PySpin.AcquisitionMode_Continuous)

    # Acquisition
    cam.BeginAcquisition()
    recording = True
    print("Recording started...")

    # Setting color image RGB
    processor = PySpin.ImageProcessor()
    processor.SetColorProcessing(PySpin.SPINNAKER_COLOR_PROCESSING_ALGORITHM_BILINEAR)

    # Saving type of name file (COULD BE FUNCTION THAT DOES THIS)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_path = os.path.join(save_dir, f"session_{timestamp}.mp4")

    # Using a separate Thread to save it so it  
    saver_thread = threading.Thread(
        target=save_video,
        args=(video_path,),
        daemon=True
    )
    saver_thread.start()

    last_frame_id = None

    while recording:
        try:
            image = cam.GetNextImage(1000)
            frame_id = image.GetFrameID()

            # Corrupterd Frames 
            if image.IsIncomplete():
                image.Release()
                print(f"frame_{frame_id} CORRUPTED (incomplete transfer)")
                continue
            
            # Dropped, unsaved frames for the video
            if last_frame_id is not None:
                gap = frame_id - last_frame_id - 1
                if gap > 0:
                    print(f"DROPPED {gap} frame(s) between {last_frame_id} and {frame_id}")

            # If everything went right save that frame in RGB
            last_frame_id = frame_id

            image_converted = processor.Convert(image, PySpin.PixelFormat_BGR8)
            image.Release()

            # This is in the case that the Queue buffer has overflown
            try:
                frame_queue.put_nowait((frame_id, image_converted.GetNDArray().copy()))
            except queue.Full:
                print(f"Queue full, dropping frame {frame_id}")

        # This is for an unexpected error (Such as when we press "E" and end the recording)
        except PySpin.SpinnakerException as e:
            print(f"Capture error: {e}")
            break

    frame_queue.put(None)
    saver_thread.join()
    print(f"Video saved to: {video_path}")


def save_video(video_path):
    writer = None
    while True:
        item = frame_queue.get()
        if item is None:
            break
        frame_id, frame_data = item

        if writer is None:
            h, w = frame_data.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(video_path, fourcc, TARGET_FPS, (w, h))

        writer.write(frame_data)

    if writer is not None:
        writer.release()


def stop_recording():
    global cam, cam_list, system, recording
    recording = False
    try:
        if cam is not None:
            cam.EndAcquisition()
            cam.DeInit()
            del cam
            cam = None
        if cam_list is not None:
            cam_list.Clear()
            cam_list = None
        if system is not None:
            system.ReleaseInstance()
            system = None
        print("Recording stopped cleanly.")
    except PySpin.SpinnakerException as e:
        print(f"Error stopping recording: {e}")


# This is to remove the "Enter" when pressing the letter
def get_keypress():
    return msvcrt.getch().decode('utf-8')


def run():
    global recording_thread

    print("Press 'V' to open Viewer  ")
    print("Press 'R' to start Recording")
    print("Press 'S' to Stop recording ")
    print("Press 'Q' to Quit          \n")

    while True:
        key = get_keypress().upper()

        if key == 'V':
            start_spinnaker()

        elif key == 'R':
            print("Starting recording")
            recording_thread = threading.Thread(target=start_recording)
            recording_thread.daemon = True
            recording_thread.start()

        elif key == 'S':
            stop_recording()
            if recording_thread is not None:
                recording_thread.join()  # wait for capture + saver to finish cleanly
                recording_thread = None

        elif key == 'Q':
            print("\nDisconnecting...")
            break


if __name__ == '__main__':
    run()
