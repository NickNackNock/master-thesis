import mqtt_handler


def main():
    print("Connecting to MQTT broker…")
    client = mqtt_handler.connect_mqtt()
    mqtt_handler.subscribe(client)

    print(f"Subscribed to '{mqtt_handler.TOPIC_SUB}'. Waiting for commands…")
    print("  START_PLAYER    — launch SpinView")
    print("  START_RECORDING — begin capture")
    print("  STOP_RECORDING  — end capture and save video")
    print("(Ctrl-C to quit)\n")

    try:
        client.loop_forever()
    except KeyboardInterrupt:
        print("\nShutting down…")
        import acquisition
        acquisition.stop_recording()


if __name__ == '__main__':
    main()
