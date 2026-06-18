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
BUFFER_SIZE = 500

# -- Internal state --
system = None
cam = None
cam_list = None
recording = False
recording_thread = None
frame_queue = queue.Queue(maxsize=BUFFER_SIZE)


# Public API

def _capture_loop(fire_at, on_status=None):
    """Loop interno di cattura eseguito in un thread separato."""
    global system, cam, cam_list, recording

    system = PySpin.System.GetInstance()
    cam_list = system.GetCameras()
    if cam_list.GetSize() == 0:
        print("\n[ERRORE] Nessuna telecamera trovata!")
        recording = False
        cam_list.Clear()
        system.ReleaseInstance()
        return

    cam = cam_list[0]
    cam.Init()

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

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_path = os.path.join(SAVE_DIR, f"session_{timestamp}.mp4")

    saver_thread = threading.Thread(
        target=save_video,
        args=(video_path,),
        daemon=True
    )
    saver_thread.start()

    last_frame_id = None

    now = time.time()
    if fire_at > now:
        time.sleep(fire_at - now - 0.001)
        while time.time() < fire_at:
            pass

    if on_status:
        on_status(f"Recording started at {time.time():.3f}")

    cam.BeginAcquisition()
    recording = True
    print(f"\n[REGISTRAZIONE AVVIATA] Salvataggio in corso...")

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
    print(f"\n[REGISTRAZIONE TERMINATA] Video salvato in: {video_path}")


def start_recording(fire_at, on_status=None):
    global recording, recording_thread
    if recording:
        print("\nRegistrazione già in corso!")
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


def stop_camera(fire_at, on_status=None):
    global cam, cam_list, system, recording

    if not recording:
        return

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
        
    except PySpin.SpinnakerException as e:
        print(f"Error stopping recording: {e}")


def save_video(video_path):
    writer = None
    black_frame = None
    last_frame_id = None

    while True:
        item = frame_queue.get()
        if item is None:
            break

        frame_id, frame_data = item

        if writer is None:
            h, w = frame_data.shape[:2]
            # Usiamo 'mp4v' o 'XVID' che sono ben supportati su Linux
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(video_path, fourcc, TARGET_FPS, (w, h))
            black_frame = np.zeros((h, w, 3), dtype=np.uint8)

        if last_frame_id is not None:
            gap = frame_id - last_frame_id - 1
            for _ in range(gap):
                writer.write(black_frame)

        frame_bgr = cv2.cvtColor(frame_data, cv2.COLOR_BayerBG2BGR)
        writer.write(frame_bgr)
        last_frame_id = frame_id

    if writer is not None:
        writer.release()


# --- GESTIONE TASTIERA NON BLOCCANTE (LINUX) ---
def get_key_async():
    """Controlla se è stato premuto un tasto nel terminale senza bloccare il ciclo."""
    if select.select([sys.stdin], [], [], 0.05)[0]:
        return sys.stdin.read(1)
    return None


if __name__ == "__main__":
    # Salva le impostazioni originali del terminale
    old_settings = termios.tcgetattr(sys.stdin)
    try:
        # Imposta il terminale in modalità "cbreak" (legge i tasti istantaneamente senza 'Invio')
        tty.setcbreak(sys.stdin.fileno())

        print("\n=== CONTROLLO ACQUISIZIONE (LINUX) ===")
        print(" Premi [r] per INIZIARE la registrazione")
        print(" Premi [s] per INTERROMPRE la registrazione")
        print(" Premi [q] per USCIRE dal programma")
        print("=======================================\n")

        while True:
            char = get_key_async()
            
            if char:
                if char.lower() == 'r':
                    if not recording:
                        start_recording(fire_at=time.time())
                    else:
                        print("\nStai già registrando!")

                elif char.lower() == 's':
                    if recording:
                        print("\nInterruzione richiesta...")
                        stop_recording(fire_at=time.time())
                    else:
                        print("\nNessuna registrazione attiva.")

                elif char.lower() == 'q':
                    if recording:
                        print("\nChiusura in corso...")
                        stop_recording(fire_at=time.time())
                    break

    finally:
        # Ripristina lo stato originale del terminale (fondamentale, altrimenti il terminale si rompe)
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
        print("\nProgramma terminato.")
