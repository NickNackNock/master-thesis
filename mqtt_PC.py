import random
import time
from paho.mqtt import client as mqtt_client
from paho.mqtt.client import CallbackAPIVersion  # Add this import

broker = 'broker.emqx.io'
port = 1883
topic = "python/mqtt"
client_id = f'python-mqtt-{random.randint(0, 1000)}'
username = 'emqx'
password = 'public'

def connect_mqtt():
    def on_connect(client, userdata, flags, rc, properties=None):  # Add properties param
        if rc == 0:
            print("Connected to MQTT Broker!")
        else:
            print("Failed to connect, return code %d\n", rc)

    client = mqtt_client.Client(
        callback_api_version=CallbackAPIVersion.VERSION2,  # Add this
        client_id=client_id
    )
    client.username_pw_set(username, password)
    client.on_connect = on_connect
    client.connect(broker, port)
    return client

def publish(client):
    msg_count = 0
    msg = f"{msg_count}"
    print(f"Send `{msg}` to topic `{topic}`")
    

def run():
    client = connect_mqtt()
    client.loop_start()
    val = input("press 's' to start publishing messages: ")
    if val.lower() == 's':
        publish(client)

if __name__ == '__main__':
    run()