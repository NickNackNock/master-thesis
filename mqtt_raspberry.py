import PySpin
import subprocess
from paho.mqtt import client as mqtt_client
from paho.mqtt.client import CallbackAPIVersion
import random
import threading
import os

broker = 'broker.emqx.io'
port = 1883
topic = "python/mqtt"
client_id = f'python-mqtt-{random.randint(0, 100)}'
username = 'emqx'
password = 'public'

# --- Camera recording logic ---
system = None
cam = None
recording = False
spinview_path = "Lorem/Ipsum"  #Linux: "/opt/spinnaker/bin/spinview"
save_dir = "path/to/save/directory"

os.makedirs(save_dir, exist_ok=True)

def  start_spinnaker():
    subprocess.Popen([spinview_path])
    print("Spinnaker software started...")

def start_recording():
    global system, cam, cam_list, recording
    system = PySpin.System.GetInstance()
    cam_list = system.GetCameras()
    cam = cam_list[0]
    cam.Init()

    # Set acquisition mode to continuous
    cam.AcquisitionMode.SetValue(PySpin.AcquisitionMode_Continuous)
    cam.BeginAcquisition()
    recording = True
    print("Recording started...")

    while recording:
        image = cam.GetNextImage()
        if not image.IsIncomplete():
            # Convert from Bayer RAW to RGB
            image_converted = image.Convert(PySpin.PixelFormat_RGB8, PySpin.HQ_LINEAR)
            
            filepath = os.path.join(save_dir, f"frame_{image.GetFrameID()}.jpg")
            image_converted.Save(filepath)
        image.Release()

def stop_recording():
    global cam, cam_list, system, recording
    recording = False

    try:
        if cam is not None:
            cam.EndAcquisition()
            cam.DeInit()
            del cam          # explicitly delete the reference
            cam = None

        if cam_list is not None:
            cam_list.Clear()  # must clear BEFORE releasing system
            cam_list = None

        if system is not None:
            system.ReleaseInstance()
            system = None

        print("Recording stopped cleanly.")

    except PySpin.SpinnakerException as e:
        print(f"Error stopping recording: {e}")

#  MQTT setup 
def connect_mqtt() -> mqtt_client:
    def on_connect(client, userdata, flags, rc, properties=None):
        if rc == 0:
            print("Connected to MQTT Broker!")
        else:
            print("Failed to connect, return code %d\n", rc)

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
        payload = msg.payload.decode().strip()
        print(f"Received `{payload}` from `{msg.topic}` topic")

        #if payload == "5":
        #    start_spinnaker()

        if payload == "15":
            print("Starting recording...")
            t = threading.Thread(target=start_recording)
            t.daemon = True
            t.start()

        elif payload == "25":
            print("Stopping recording...")
            stop_recording()

    client.subscribe(topic)
    client.on_message = on_message


def run():
    client = connect_mqtt()
    subscribe(client)
    client.loop_forever()


if __name__ == '__main__':
    run()