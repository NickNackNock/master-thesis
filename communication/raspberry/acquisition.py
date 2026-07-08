import PySpin
import subprocess
import threading
import queue
import cv2
import os
import sys
import select
import tty
import termios
from datetime import datetime
import time
import numpy as np

# -- PATHS --
SAVE_DIR = "./Recordings"
os.makedirs(SAVE_DIR, exist_ok=True)

# -- Camera settings --
TARGET_FPS = 25
BUFFER_SIZE = 50

# -- Internal state --
system = None
cam = None
cam_list = None
recording = False
recording_thread = None
frame_queue = queue.Queue(maxsize=BUFFER_SIZE)


# ── Public API ────────────────────────────────────────────────────────────────

def open_viewer():
    """SpinView is a Windows application; not available on this platform."""
    print("[open_viewer] SpinView is Windows-only — skipping on this platform.")


def start_recording(fire_at, on_status=None):
    global recording, recording_thread
    if recording:
        print("\nRecording already started!")
        return

    recording_thread = threading.Thread(
        target=_capture_loop,
        args=(fire_at, on_status),
        daemon=True
    )
    recording_thread.start()


def stop_recording(fire_at, on_status=None):
    global recording_thread
    stop_camera(fire_at, on_status=on_status)
    if recording_thread is not None:
        recording_thread.join()
        recording_thread = None


# ── Internal helpers ──────────────────────────────────────────────────────────

def _capture_loop(fire_at, on_status=None):
    """Camera capture loop — runs in its own thread."""
    global system, cam, cam_list, recording

    system = PySpin.System.GetInstance()
    cam_list = system.GetCameras()
    if cam_list.GetSize() == 0:
        print("\n[Error] No camera found!")
        recording = False
        cam_list.Clear()
        system.ReleaseInstance()
        return

    cam = cam_list[0]
    cam.Init()
    cam.OffsetX.SetValue(0)
    cam.OffsetY.SetValue(0)
    cam.Width.SetValue(1920)
    cam.Height.SetValue(1080)

    s_node_map = cam.GetTLStreamNodeMap()
    buffer_mode = PySpin.CEnumerationPtr(s_node_map.GetNode('StreamBufferCountMode'))
    buffer_count = PySpin.CIntegerPtr(s_node_map.GetNode('StreamBufferCountManual'))

    if PySpin.IsAvailable(buffer_mode) and PySpin.IsWritable(buffer_mode):
        manual = buffer_mode.GetEntryByName('Manual')
        if PySpin.IsAvailable(manual) and PySpin.IsReadable(manual):
            buffer_mode.SetIntValue(manual.GetValue())

    if PySpin.IsAvailable(buffer_count) and PySpin.IsWritable(buffer_count):
        buffer_count.SetValue(min(buffer_count.GetMax(), 128))

    cam.AcquisitionFrameRateEnable.SetValue(True)
    cam.AcquisitionFrameRate.SetValue(TARGET_FPS)
    cam.AcquisitionMode.SetValue(PySpin.AcquisitionMode_Continuous)

    last_frame_id = None

    # Precision-timed start
    now = time.time()
    if fire_at > now:
        time.sleep(fire_at - now - 0.001)
        while time.time() < fire_at:
            pass

    cam.BeginAcquisition()
    recording = True
        
    timestamp = datetime.now().strftime('%Y-%m-%d %H-%M-%S.%f_cam2')[:-3]
    video_path = os.path.join(SAVE_DIR, f"session_{timestamp}.mkv")

    saver_thread = threading.Thread(
        target=save_video,
        args=(video_path,),
        daemon=True
    )
    saver_thread.start()
    
    print("\n[RECORDING STARTED] Capturing frames")
    if on_status:
        on_status(f"Recording started at {time.time():.3f}")

    while recording:
        try:
            image = cam.GetNextImage(1000)
            frame_id = image.GetFrameID()

            if image.IsIncomplete():
                image.Release()
                print(f"frame_{frame_id} CORRUPTED")
                continue

            if last_frame_id is not None:
                gap = frame_id - last_frame_id - 1
                if gap > 0:
                    print(f"DROPPED {gap} frame(s)")

            last_frame_id = frame_id
            raw_array = image.GetNDArray().copy()
            image.Release()

            try:
                frame_queue.put_nowait((frame_id, raw_array))
            except queue.Full:
                print(f"Queue full, dropping frame {frame_id}")

        except PySpin.SpinnakerException as e:
            print(f"Capture error: {e}")
            break

    frame_queue.put(None)
    saver_thread.join()
    print(f"\n[RECORDING FINISHED] Video saved at: {video_path}")
    # Notify MQTT side that the file is ready
    if on_status:
        on_status(f"Video saved to {video_path}")


def stop_camera(fire_at, on_status=None):
    """Set the recording flag to False and release all camera resources."""
    global cam, cam_list, system, recording

    if not recording:
        return

    # Precision-timed stop
    now = time.time()
    if fire_at > now:
        time.sleep(fire_at - now - 0.001)
        while time.time() < fire_at:
            pass

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
        if on_status:
            on_status(f"Recording actually ended at {time.time():.3f}")

    except PySpin.SpinnakerException as e:
        print(f"Error stopping recording: {e}")


def save_video(video_path):
    """Drain the frame queue and encode via ffmpeg (libx264 → .mkv)."""
    ffmpeg = None
    black_frame = None
    last_frame_id = None

    while True:
        item = frame_queue.get()
        if item is None:
            break

        frame_id, frame_data = item

        if ffmpeg is None:
            h, w = frame_data.shape  # Bayer → single-channel, so shape is (H, W)

            ffmpeg = subprocess.Popen(
                [
                    "ffmpeg",
                    "-hide_banner", "-loglevel", "error",
                    "-y",
                    "-f", "rawvideo",
                    "-pix_fmt", "bgr24",
                    "-s", f"{w}x{h}",
                    "-r", str(TARGET_FPS),
                    "-i", "-",
                    "-c:v", "libx264",
                    "-preset", "ultrafast",
                    "-crf", "20",
                    "-pix_fmt", "yuv420p",
                    video_path,
                ],
                stdin=subprocess.PIPE,
            )

            black_frame = np.zeros((h, w, 3), dtype=np.uint8)

        # Fill any dropped frames with black
        if last_frame_id is not None:
            gap = frame_id - last_frame_id - 1
            for _ in range(gap):
                ffmpeg.stdin.write(black_frame.tobytes())

        frame_bgr = cv2.cvtColor(frame_data, cv2.COLOR_BayerBG2BGR)
        ffmpeg.stdin.write(frame_bgr.tobytes())
        last_frame_id = frame_id

    if ffmpeg is not None:
        ffmpeg.stdin.close()
        ffmpeg.wait()


# ── Standalone keyboard mode (Linux only) ────────────────────────────────────

def _get_key_async():
    """Return a keypress without blocking, or None if none pending."""
    if select.select([sys.stdin], [], [], 0.05)[0]:
        return sys.stdin.read(1)
    return None


if __name__ == "__main__":
    old_settings = termios.tcgetattr(sys.stdin)
    try:
        tty.setcbreak(sys.stdin.fileno())

        print("\n=== MENU ===")
        print(" Press [r] to RECORD")
        print(" Press [s] to STOP recording")
        print(" Press [q] to QUIT program")
        print("=" * 40 + "\n")

        while True:
            char = _get_key_async()
            if char:
                if char.lower() == 'r':
                    if not recording:
                        start_recording(fire_at=time.time())
                    else:
                        print("\nYou are already recording!")

                elif char.lower() == 's':
                    if recording:
                        print("\nStopping...")
                        stop_recording(fire_at=time.time())
                    else:
                        print("\nNo recording in progress.")

                elif char.lower() == 'q':
                    if recording:
                        print("\nClosing program...")
                        stop_recording(fire_at=time.time())
                    break

    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
        print("\nProgram stopped successfully.")
