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

# -- Preview settings (tuned for Raspberry Pi 5 / budget CPU) --
SHOW_PREVIEW = True        # toggle the live cv2.imshow preview window on/off
PREVIEW_MAX_WIDTH = 640    # small preview frame -> cheap resize + cheap demosaic-on-view
PREVIEW_EVERY_N = 20        # only build a preview frame every 20th captured frame (a slideshow)

# -- Internal state --
system = None
cam = None
cam_list = None
recording = False
recording_thread = None
frame_queue = queue.Queue(maxsize=BUFFER_SIZE)

preview_queue = queue.Queue(maxsize=1)  # always holds only the latest preview frame
preview_thread = None
preview_stop_event = threading.Event()
preview_thread_lock = threading.Lock()


# --------- Public API ---------

def check_camera_connected():
    """Check whether a camera is currently detected on the bus."""
    try:
        sys_instance = PySpin.System.GetInstance()
        cams = sys_instance.GetCameras()
        num = cams.GetSize()

        info = []
        # Use an index-based loop. Python's 'for c in cams' can sometimes
        # create lingering hidden references under the hood in PySpin.
        for i in range(num):
            cam_obj = cams.GetByIndex(i)
            try:
                tl_nodemap = cam_obj.GetTLDeviceNodeMap()
                model_node = PySpin.CStringPtr(tl_nodemap.GetNode('DeviceModelName'))
                serial_node = PySpin.CStringPtr(tl_nodemap.GetNode('DeviceSerialNumber'))

                model = (model_node.GetValue()
                         if PySpin.IsAvailable(model_node) and PySpin.IsReadable(model_node)
                         else "Unknown model")
                serial = (serial_node.GetValue()
                          if PySpin.IsAvailable(serial_node) and PySpin.IsReadable(serial_node)
                          else "Unknown SN")
                info.append(f"{model} (SN {serial})")

                # 1. Explicitly delete node references
                del model_node
                del serial_node
                del tl_nodemap

            except PySpin.SpinnakerException:
                info.append("Unknown camera (could not read device info)")
            finally:
                # 2. Explicitly delete the individual camera reference
                del cam_obj

        # 3. Clear the list, then explicitly delete the list reference
        cams.Clear()
        del cams

        sys_instance.ReleaseInstance()
        return num > 0, info

    except PySpin.SpinnakerException as e:
        print(f"[check_camera_connected] Error: {e}")
        return False, []


def start_recording(fire_at, on_status=None, show_preview=False):
    global recording, recording_thread, SHOW_PREVIEW
    
    # Update the global flag so ALL functions (preview thread, push frame, etc.) see it
    SHOW_PREVIEW = show_preview
    
    if recording:
        print("\nRecording already started!")
        return

    recording_thread = threading.Thread(
        target=_capture_loop,
        args=(fire_at, on_status), # No need to pass show_preview here anymore
        daemon=True
    )
    recording_thread.start()


def stop_recording(fire_at, on_status=None):
    global recording_thread
    stop_camera(fire_at, on_status=on_status)
    if recording_thread is not None:
        recording_thread.join()
        recording_thread = None


# --------- Preview helpers---------

def _push_preview_frame(raw_array, frame_id):
    """Build a small preview frame and hand it to the display thread.

    Non-blocking and best-effort: only 1 in every PREVIEW_EVERY_N frames is
    even converted (demosaic is the expensive part), and the preview queue
    holds just the single latest frame so the window can never build a
    backlog and never slows down capture/encoding.
    """
    if not SHOW_PREVIEW:
        return
    if PREVIEW_EVERY_N > 1 and (frame_id % PREVIEW_EVERY_N) != 0:
        return
    try:
        preview_bgr = cv2.cvtColor(raw_array, cv2.COLOR_BayerBG2BGR)
        ph, pw = preview_bgr.shape[:2]
        if pw > PREVIEW_MAX_WIDTH:
            scale = PREVIEW_MAX_WIDTH / pw
            preview_bgr = cv2.resize(preview_bgr, (int(pw * scale), int(ph * scale)),
                                      interpolation=cv2.INTER_NEAREST)  # cheapest interpolation

        if preview_queue.full():
            try:
                preview_queue.get_nowait()
            except queue.Empty:
                pass
        preview_queue.put_nowait(preview_bgr)
    except (cv2.error, queue.Full):
        pass


def _ensure_preview_thread():
    """Start the single persistent preview thread once, on first use.

    IMPORTANT: OpenCV's HighGUI backend (GTK/Qt on Linux, which is what the
    Pi's desktop uses too) binds its window-system connection to whichever
    thread first calls a highgui function. Spawning a brand-new thread for
    every recording (start it in start_recording/_capture_loop, join it in
    stop_camera — as an earlier version of this did) means the *second*
    recording's window silently stops updating: the backend stays attached
    to the first, now-dead thread, and calls from the new thread are
    dropped without raising an error. So we create exactly one preview
    thread for the whole process lifetime and just feed it frames.
    """
    global preview_thread
    if not SHOW_PREVIEW:
        return
    with preview_thread_lock:
        if preview_thread is not None and preview_thread.is_alive():
            return
        preview_stop_event.clear()
        preview_thread = threading.Thread(target=_preview_loop, daemon=True)
        preview_thread.start()


def _preview_loop():
    """Persistent thread that owns the preview window for the whole program run.

    Just keeps pulling from preview_queue for as long as the process is
    alive. Between recordings the queue is simply empty, so the loop idles
    (window stays on the last frame shown) until the next recording starts
    pushing frames again — no window teardown/recreation involved.
    """
    window_name = "Camera Preview"
    os.environ.setdefault("QT_LOGGING_RULES", "qt.qpa.fonts=false")
    try:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    except cv2.error as e:
        print(f"[Preview] Could not open display window (no GUI/display available?): {e}")
        return

    while not preview_stop_event.is_set():
        try:
            frame = preview_queue.get(timeout=0.2)
        except queue.Empty:
            cv2.waitKey(1)  # still pump the GUI event loop even with no new frame
            continue
        try:
            cv2.imshow(window_name, frame)
        except cv2.error as e:
            print(f"[Preview] Display error: {e}")
        cv2.waitKey(1)

    try:
        cv2.destroyWindow(window_name)
        cv2.waitKey(1)
    except cv2.error:
        pass


# --------- Internal helpers ---------

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

    timestamp = datetime.now().strftime('%Y-%m-%d %H-%M-%S.%f')[:-3]
    video_path = os.path.join(SAVE_DIR, f"session_{timestamp}_cam2.mkv")

    saver_thread = threading.Thread(
        target=save_video,
        args=(video_path,),
        daemon=True
    )
    saver_thread.start()

    if SHOW_PREVIEW:
        _ensure_preview_thread()

    print("\n[RECORDING STARTED] Capturing frames")
    if on_status:
        on_status(f"Recording started at {time.time():.3f}")

    # Two independent drop counters, because they mean very different things:
    #  - camera_dropped_frames: the CAMERA itself reports a gap in frame IDs.
    #    This means the sensor/USB transfer lost the frame before your code
    #    ever saw it — nothing your Python code does can recover this.
    #  - pipeline_dropped_frames: the camera delivered the frame fine, but
    #    frame_queue was full (the saver/ffmpeg side is falling behind), so
    #    YOUR software chose to discard it. This is a software/CPU bottleneck,
    #    not a camera problem.
    camera_dropped_frames = 0
    pipeline_dropped_frames = 0
    QUEUE_DROP_REPORT_EVERY = 25  # ~1s of drops at 25 fps, avoid flooding MQTT

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
                    camera_dropped_frames += gap
                    print(f"DROPPED {gap} frame(s) [CAMERA-SIDE: sensor/transfer loss]")
                    if on_status:
                        on_status(
                            f"WARNING: camera itself dropped {gap} frame(s) around "
                            f"frame {frame_id} (sensor/USB transfer loss — not a "
                            f"software/queue issue). Total camera-side drops so far: "
                            f"{camera_dropped_frames}"
                        )

            last_frame_id = frame_id
            host_timestamp_s = time.time()  # PC wall-clock time at acquisition
            raw_array = image.GetNDArray().copy()
            image.Release()

            _push_preview_frame(raw_array, frame_id)

            try:
                frame_queue.put_nowait((frame_id, raw_array, host_timestamp_s))
            except queue.Full:
                pipeline_dropped_frames += 1
                print(f"Queue full, dropping frame {frame_id} [PIPELINE-SIDE: saver/encoder too slow]")
                if on_status and pipeline_dropped_frames % QUEUE_DROP_REPORT_EVERY == 0:
                    on_status(
                        f"WARNING: pipeline (queue/encoder) has dropped "
                        f"{pipeline_dropped_frames} frame(s) so far — the camera "
                        f"delivered these fine, but the Pi couldn't save/encode "
                        f"them fast enough. Not a camera fault."
                    )

        except PySpin.SpinnakerException as e:
            print(f"Capture error: {e}")
            break

    frame_queue.put(None)
    saver_thread.join()

    print(f"\n[RECORDING FINISHED] Video saved at: {video_path}")

    summary = (
        f"Recording finished. Camera-side drops (sensor/transfer loss): "
        f"{camera_dropped_frames}. Pipeline-side drops (queue full / encoder "
        f"too slow): {pipeline_dropped_frames}."
    )
    print(summary)

    # Notify MQTT side that the file is ready, plus the drop breakdown
    if on_status:
        on_status(f"Video saved to {video_path}")
        on_status(summary)


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
    """Drain the frame queue and encode via ffmpeg (libx264 → .mkv).

    Also writes a companion "<video_name>_timestamps.txt" file with one
    line per output frame:
      - host_time_s: time.time() on the PC/Pi when the frame was received
        from the camera.

    Frames that were dropped and filled in with a black frame are marked
    DROPPED with an empty timestamp field.
    """
    ffmpeg = None
    black_frame = None
    last_frame_id = None

    timestamps_path = os.path.splitext(video_path)[0] + "_timestamps.txt"
    ts_file = open(timestamps_path, "w")
    ts_file.write("frame_id,host_time_s,status\n")

    while True:
        item = frame_queue.get()
        if item is None:
            break

        frame_id, frame_data, host_timestamp_s = item

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
            for missing_id in range(last_frame_id + 1, frame_id):
                ffmpeg.stdin.write(black_frame.tobytes())
                ts_file.write(f"{missing_id},,DROPPED\n")

        frame_bgr = cv2.cvtColor(frame_data, cv2.COLOR_BayerBG2BGR)
        ffmpeg.stdin.write(frame_bgr.tobytes())

        ts_file.write(f"{frame_id},{host_timestamp_s:.6f},OK\n")
        last_frame_id = frame_id

    ts_file.close()

    if ffmpeg is not None:
        ffmpeg.stdin.close()
        ffmpeg.wait()
    
    #re-naming the file with the true start time acquisition
    print('Converting the file to the true starting time-stamp')
    
    # Open timestamps_path
    with open(timestamps_path, "r") as fd:
        for n, line in enumerate(fd):
            if n == 1: # Row 0 is the header, row 1 is the first actual frame
                # Convert the string to a float
                true_ts = float(line.split(',')[1])
                break # We have what we need, stop reading the file
    
    # Use fromtimestamp to match local time
    true_start = datetime.fromtimestamp(true_ts).strftime('%Y-%m-%d %H-%M-%S.%f')[:-3]
    true_file_video = os.path.join(SAVE_DIR, f"session_{true_start}_cam2.mkv")
    true_file_ts = os.path.join(SAVE_DIR, f"session_{true_start}_cam2.txt")
    
    # Renaming both files: video and timestamp
    os.rename(video_path, true_file_video) 
    os.rename(timestamps_path, true_file_ts)
    print('File renamed successfully')
    print(f"[save_video] Frame timestamps saved at: {timestamps_path}")


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
