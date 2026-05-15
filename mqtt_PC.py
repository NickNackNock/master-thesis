import random
import sys
import tty
import termios
import time
from paho.mqtt import client as mqtt_client
from paho.mqtt.client import CallbackAPIVersion
import json

broker = 'broker.emqx.io'
port = 1883
topic_publish = "python/mqtt/commands"
topic_subscribe = "python/mqtt/status"
client_id = f'python-mqtt-{random.randint(0, 1000)}'
username = 'emqx'
password = 'public'

# How far ahead (seconds) to schedule the sync command.
# Must be larger than worst-case network latency to all devices.
SYNC_LEAD_TIME = 2.0


def connect_mqtt():
    def on_connect(client, userdata, flags, rc, properties=None):
        if rc == 0:
            print("Connected to MQTT Broker!")
            client.subscribe(topic_subscribe)
            print(f"Subscribed to `{topic_subscribe}` for status updates")
        else:
            print("Failed to connect, return code %d\n", rc)

    def on_message(client, userdata, msg):
        print(f"[STATUS] Topic: `{msg.topic}` | Message: `{msg.payload.decode()}`")

    client = mqtt_client.Client(
        callback_api_version=CallbackAPIVersion.VERSION2,
        client_id=client_id
    )
    client.username_pw_set(username, password)
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(broker, port)
    return client


def publish(client, command, scheduled_at: float | None = None):
    """
    Publish a command, optionally with a Unix timestamp telling
    subscribers exactly when to execute it.
    """
    payload = {
        "command": command,
        "scheduled_at": scheduled_at or time.time(),  # epoch seconds (float)
    }
    message = json.dumps(payload)
    result = client.publish(topic_publish, message, qos=1)
    status = result[0]
    if status == 0:
        if scheduled_at:
            delay = scheduled_at - time.time()
            print(f"Sent `{command}` scheduled in {delay:.2f}s  →  topic `{topic_publish}`")
        else:
            print(f"Sent `{command}` to topic `{topic_publish}`")
    else:
        print(f"Failed to send message to topic `{topic_publish}`")


def publish_synced(client, command):
    """Schedule a command SYNC_LEAD_TIME seconds from now."""
    fire_at = time.time() + SYNC_LEAD_TIME
    print("Should start/end at: " + str(fire_at))
    publish(client, command, scheduled_at=fire_at)


def get_keypress():
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch


def run():
    client = connect_mqtt()
    time.sleep(1)
    client.loop_start()
    print("MQTT Publisher is running\n")
    print("Press 'V' to open the Viewer  (immediate)")
    print("Press 'R' to start Recording  (SYNCED)")
    print("Press 'S' to Stop recording    (SYNCED)")
    print("Press 'Q' to Quit\n")

    while True:
        key = get_keypress().upper()

        if key == 'V':
            publish(client, "START_PLAYER")           # no sync needed
        elif key == 'R':
            publish_synced(client, "START_RECORDING") # all devices fire together
        elif key == 'S':
            publish_synced(client, "STOP_RECORDING")  # all devices stop together
        elif key == 'Q':
            print("\nDisconnecting...")
            client.loop_stop()
            client.disconnect()
            break


if __name__ == '__main__':
    run()
