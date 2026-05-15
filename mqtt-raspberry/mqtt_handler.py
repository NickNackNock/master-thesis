import json
import random
import time
import threading

from paho.mqtt import client as mqtt_client
from paho.mqtt.client import CallbackAPIVersion

import acquisition

# -- Broker settings (edit here if needed) --
BROKER   = 'broker.emqx.io'
PORT     = 1883
USERNAME = 'emqx'
PASSWORD = 'public'

TOPIC_SUB = "python/mqtt/commands"
TOPIC_PUB = "python/mqtt/status"

client_id = f'python-mqtt-{random.randint(0, 100_000)}'


def connect_mqtt():
    def on_connect(client, userdata, flags, rc, properties=None):
        if rc == 0:
            print("Connected to MQTT Broker!")
        else:
            print(f"Failed to connect, return code {rc}")

    client = mqtt_client.Client(
        callback_api_version=CallbackAPIVersion.VERSION2,
        client_id = client_id
    )
    client.username_pw_set(USERNAME, PASSWORD)
    client.on_connect = on_connect
    client.connect(BROKER, PORT)
    return client


def subscribe(client: mqtt_client.Client):
    """Attach the message handler and subscribe to the command topic."""
    def on_message(client, userdata, msg):
        try:
            payload  = json.loads(msg.payload.decode())
            command  = payload["command"]
            fire_at  = payload.get("scheduled_at", 0)
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Malformed MQTT payload: {e}")
            return

        print(f"Received command '{command}' (scheduled_at={fire_at}) on {msg.topic}")

        now = time.time()
        if fire_at > now:
            # Coarse sleep, then a tight spin for sub-millisecond accuracy
            time.sleep(fire_at - now - 0.001)
            while time.time() < fire_at:
                pass

        match command:
            case "START_PLAYER":
                acquisition.open_viewer()
                publish(client, "SpinView launched")

            case "START_RECORDING":
                publish(client, f"Recording started at {time.time():.3f}")
                # Run in a thread so on_message returns immediately
                threading.Thread(target=acquisition.start_recording, daemon=True).start()

            case "STOP_RECORDING":
                publish(client, f"Recording stopped at {time.time():.3f}")
                # stop_recording blocks until everything is flushed — keep it off the MQTT thread
                threading.Thread(target=acquisition.stop_recording, daemon=True).start()

            case _:
                print(f"Unknown command: '{command}'")

    client.subscribe(TOPIC_SUB)
    client.on_message = on_message


def publish(client: mqtt_client.Client, message: str):
    """Publish a status string to the status topic (QoS 1)."""
    result = client.publish(TOPIC_PUB, message, qos=1)
    if result[0] == 0:
        print(f"Published: '{message}' → {TOPIC_PUB}")
    else:
        print(f"Failed to publish to {TOPIC_PUB}")

