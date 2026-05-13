import PySpin
import subprocess
from paho.mqtt import client as mqtt_client
from paho.mqtt.client import CallbackAPIVersion
import random
import threading
import os
import queue
import cv2
import json
import time

broker = 'broker.emqx.io'
port = 1883
topic_subscribe = "python/mqtt/commands"
topic_publish = "python/mqtt/status"
client_id = f'python-mqtt-subscriber-{random.randint(0, 10000)}'  # Fix 2: distinct prefix + wider range
username = 'emqx'
password = 'public'

system = None
cam = None
cam_list = None
recording = False
spinview_path = "C:/Program Files/Teledyne/Spinnaker/bin64/vs2015/SpinView_WPF_v140.exe"
save_dir = "C:/Users/mensi/OneDrive/Desktop/Thesis/Recordings"
os.makedirs(save_dir, exist_ok=True)


def start_spinnaker():
    print("Spinnaker software started...")
    subprocess.Popen([spinview_path])


import queue

frame_queue = queue.Queue(maxsize=500)

def start_recording():
    global system, cam, cam_list, recording
    system = PySpin.System.GetInstance()
    cam_list = system.GetCameras()
    cam = cam_list[0]
    cam.Init()
    cam.AcquisitionMode.SetValue(PySpin.AcquisitionMode_Continuous)
    cam.BeginAcquisition()
    recording = True
    print("Recording started...")

    processor = PySpin.ImageProcessor()
    processor.SetColorProcessing(PySpin.SPINNAKER_COLOR_PROCESSING_ALGORITHM_HQ_LINEAR)

    # Start a separate thread just for saving
    saver_thread = threading.Thread(target=save_frames, daemon=True)
    saver_thread.start()

    while recording:
        try:
            image = cam.GetNextImage()
            if image.IsIncomplete():
                image.Release()
                continue

            # Convert immediately and put raw data in queue — fast
            image_converted = processor.Convert(image, PySpin.PixelFormat_BGR8)
            frame_id = image.GetFrameID()
            image.Release()  # release camera buffer ASAP

            frame_queue.put((frame_id, image_converted.GetNDArray().copy()))

        except PySpin.SpinnakerException as e:
            print(f"Capture error: {e}")
            break

    frame_queue.put(None)  # signal saver to stop
    saver_thread.join()

def save_frames():
    """Runs in its own thread, saves frames from the queue."""
    while True:
        item = frame_queue.get()
        if item is None:
            break
        frame_id, frame_data = item
        filepath = os.path.join(save_dir, f"frame_{frame_id}.jpg")
        cv2.imwrite(filepath, frame_data)

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


def connect_mqtt():
    def on_connect(client, userdata, flags, rc, properties=None):
        if rc == 0:
            print("Connected to MQTT Broker!")
        else:
            print(f"Failed to connect, return code {rc}")

    client = mqtt_client.Client(
        callback_api_version=CallbackAPIVersion.VERSION2,
        client_id=client_id
    )
    client.username_pw_set(username, password)
    client.on_connect = on_connect
    client.connect(broker, port)
    return client


def subscribe(client: mqtt_client):
    def on_message(client, userdata, msg):
        payload = json.loads(msg.payload.decode())
        command = payload["command"]
        fire_at = payload["scheduled_at"]
        print(f"Received `{payload}` from `{msg.topic}` topic")  # Fix 1: msg.topic not msg.topic_subscribe

        now = time.time()

        match command:
            case "START_PLAYER":
                start_spinnaker()
                publish(client, "Application Started")

            case "START_RECORDING":
                if fire_at > now:
                    time.sleep(fire_at - now)
                    while time.time() < fire_at:
                        pass
                publish(client, "Recording Started at " + str(time.time()))
                print("Starting recording")
                t = threading.Thread(target=start_recording)
                t.daemon = True
                t.start()
                

            case "STOP_RECORDING":
                if fire_at > now:
                    time.sleep(fire_at - now)
                    while time.time() < fire_at:
                        pass
                publish(client, "Recording Ended at " + str(time.time()))
                print("Recording ended")
                stop_recording()
                

    client.subscribe(topic_subscribe)
    client.on_message = on_message


def publish(client, message):
    result = client.publish(topic_publish, message, qos=1)
    if result[0] == 0:
        print(f"Sent '{message}' to topic {topic_publish}")
    else:
        print(f"Failed to send message to topic {topic_publish}")


def run():
    client = connect_mqtt()
    subscribe(client)
    client.loop_forever()


if __name__ == '__main__':
    run()
