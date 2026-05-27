import PySpin
import subprocess
import threading
import queue
import cv2
import os
from datetime import datetime
import time

# -- PATHS (edit here if needed) --
SPINVIEW_PATH = "C:/Program Files/Teledyne/Spinnaker/bin64/vs2015/SpinView_WPF_v140.exe"
SAVE_DIR = "C:/Recordings"
os.makedirs(SAVE_DIR, exist_ok=True)

# -- Camera settings --
TARGET_FPS = 25
BUFFER_SIZE = 500

# -- Internal state --
system = None
cam = None
cam_list = None
recording = False
recording_thread = None
frame_queue = queue.Queue(maxsize=BUFFER_SIZE)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def open_viewer():
    """Launch the Spinnaker SpinView viewer application."""
    print("Starting SpinView...")
    subprocess.Popen([SPINVIEW_PATH])


def start_recording(fire_at):
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

    # Setting color image RGB
    processor = PySpin.ImageProcessor()
    processor.SetColorProcessing(PySpin.SPINNAKER_COLOR_PROCESSING_ALGORITHM_BILINEAR)

    # Saving type of name file (COULD BE FUNCTION THAT DOES THIS)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_path = os.path.join(SAVE_DIR, f"session_{timestamp}.mp4")

    # Using a separate Thread to save it so it  
    saver_thread = threading.Thread(
        target=save_video,
        args=(video_path,),
        daemon=True
    )
    saver_thread.start()

    last_frame_id = None

    # waiting time to start here
    now = time.time()
    if fire_at > now:
        # Coarse sleep, then a tight spin for sub-millisecond accuracy
        time.sleep(fire_at - now - 0.001)
        while time.time() < fire_at:
            pass

    print(f"Recording actually started at {time.time():.3f}")

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



def stop_recording(fire_at):
    """
    Signal the capture loop to stop, release the camera, and wait for the
    saver thread to flush all queued frames before returning.
    """
    global recording_thread
    
    stop_camera(fire_at)

    if recording_thread is not None:
        recording_thread.join()
        recording_thread = None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def init_camera():
    """Initialise the Spinnaker system and the first available camera."""
    global system, cam, cam_list

    system = PySpin.System.GetInstance()
    cam_list = system.GetCameras()
    cam = cam_list[0]
    cam.Init()

    """Set the stream buffer to Manual mode with a fixed frame count."""
    s_node_map = cam.GetTLStreamNodeMap()
    buffer_mode = PySpin.CEnumerationPtr(s_node_map.GetNode('StreamBufferCountMode'))
    buffer_count = PySpin.CIntegerPtr(s_node_map.GetNode('StreamBufferCountManual'))

    if PySpin.IsAvailable(buffer_mode) and PySpin.IsWritable(buffer_mode):
        manual = buffer_mode.GetEntryByName('Manual')
        if PySpin.IsAvailable(manual) and PySpin.IsReadable(manual):
            buffer_mode.SetIntValue(manual.GetValue())

    if PySpin.IsAvailable(buffer_count) and PySpin.IsWritable(buffer_count):
        buffer_count.SetValue(min(buffer_count.GetMax(), 40))

    cam.AcquisitionFrameRateEnable.SetValue(True)
    cam.AcquisitionFrameRate.SetValue(TARGET_FPS)
    cam.AcquisitionMode.SetValue(PySpin.AcquisitionMode_Continuous)


def stop_camera(fire_at):
    """Set the recording flag to False and release all camera resources."""
    global cam, cam_list, system, recording

    recording = False

    # waiting time to start here
    now = time.time()
    if fire_at > now:
        # Coarse sleep, then a tight spin for sub-millisecond accuracy
        time.sleep(fire_at - now - 0.001)
        while time.time() < fire_at:
            pass

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
        print(f"Recording actually ended at {time.time():.3f}")
        
    except PySpin.SpinnakerException as e:
        print(f"Error stopping recording: {e}")


def save_video(video_path):
    """
    Saver thread: reads frames from the queue and writes them to an MP4 file.
    Exits when it receives the sentinel value None.
    """
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
