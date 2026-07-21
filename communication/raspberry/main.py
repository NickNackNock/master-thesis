import time
import subprocess
import mqtt_handler

def ensure_time_synced():
    print("[Init] Enabling NTP and waiting for time synchronization...")
    
    # 1. Turn on NTP
    subprocess.run(["sudo", "timedatectl", "set-ntp", "true"], check=False)
    
    # 2. Wait until the system actually confirms the clock is synced
    while True:
        result = subprocess.run(["timedatectl"], capture_output=True, text=True)
        if "System clock synchronized: yes" in result.stdout:
            print("[Init] System clock is synchronized!")
            break
        
        print("[Init] Waiting for NTP sync...")
        time.sleep(1.0)
        
        
def lower_background_priority():
    print("[Init] Lowering priority of background GUI/Network processes...")
    
    for process_name in ["rustdesk", "Xorg"]:
        try:
            # 1. Find the PIDs for the process
            result = subprocess.run(["pgrep", process_name], capture_output=True, text=True)
            pids = result.stdout.strip().split('\n')
            
            # Remove any empty strings if no PIDs were found
            pids = [pid for pid in pids if pid]
            
            if pids:
                # 2. Run sudo renice on all found PIDs
                cmd = ["sudo", "renice", "-n", "19", "-p"] + pids
                subprocess.run(cmd, check=False, stdout=subprocess.DEVNULL)
                print(f" Set {process_name} (PIDs: {', '.join(pids)}) to lowest priority (+19).")
            else:
                print(f"{process_name} is not running, skipping.")
                
        except Exception as e:
            print(f"Could not renice {process_name}: {e}")
            

def main():
    ensure_time_synced()
    lower_background_priority()
    
    print("Connecting to MQTT broker...")
    client = mqtt_handler.connect_mqtt()
    mqtt_handler.subscribe(client)

    print(f"Subscribed to '{mqtt_handler.TOPIC_SUB}'. Waiting for commands…")
    print("  CALIB_START_RECORDING - begin capture with video feedback")
    print("  START_RECORDING - begin capture")
    print("  STOP_RECORDING  - end capture and save video")
    print("  CHECK_CAMERA    - check whether the camera is physically connected")
    print("(Ctrl-C to quit)\n")

    try:
        client.loop_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        import acquisition
        acquisition.stop_recording(time.time())


if __name__ == '__main__':
    main()
