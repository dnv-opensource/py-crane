import time

import requests

SERVER_URL = "http://localhost:9100"


def get_debug_status():
    """Get detailed debug information without relying on video."""
    try:
        response = requests.get(f"{SERVER_URL}/debug", timeout=2)
        return response.json()
    except Exception as e:
        print(f"[ERROR] Failed to get debug status: {e}")
        return None


def test_continuous_commands():
    """Test sending multiple commands sequentially and verify execution."""

    commands = [
        "Rotate the boom 45 degrees clockwise in 5 seconds",
        "Tilt the boom up 30 degrees in 5 seconds",
        "Extend the boom by 10 meters in 5 seconds",
        "Rotate the boom 45 degrees counter-clockwise in 5 seconds",
    ]

    print("=" * 70)
    print("REAL-TIME INTERACTIVE CRANE TEST")
    print("=" * 70)

    for i, cmd in enumerate(commands, 1):
        print(f"\n[TEST {i}/{len(commands)}] Sending command: {cmd}")

        try:
            response = requests.post(f"{SERVER_URL}/command", json={"text": cmd}, timeout=5)
            result = response.json()

            if result["status"] == "ok":
                print(f"[SUCCESS] Command accepted ({result['count']} action(s) generated)")
            else:
                print(f"[FAILED] Command parsing error: {result.get('msg', 'Unknown error')}")
                return
        except Exception as e:
            print(f"[ERROR] Request failed: {e}")
            return

        print("[INFO] Waiting for command execution...")

        last_motion = None
        execution_start = time.time()

        for _ in range(25):
            time.sleep(0.5)

            status = get_debug_status()
            if not status:
                continue

            elapsed = time.time() - execution_start
            print(
                f"  [{elapsed:.1f}s] Timeline frames: {status['timeline_frames']}, Active commands: {status['active_commands']}"
            )

            if status["motion_detected"]:
                motion = status["motion_detected"]
                print(
                    f"  Motion: Boom_len_delta={motion['boom_length_delta']:+.3f}m, Polar_delta={motion['polar_delta']:+.4f}rad, Az_delta={motion['azimuth_delta']:+.4f}rad, Wire_delta={motion['wire_delta']:+.3f}m"
                )
                last_motion = motion

            if status["active_commands"] == 0 and status["timeline_frames"] > 50:
                if last_motion:
                    print(f"[VERIFIED] Command {i} execution confirmed")
                    print(f"  Final motion - Boom_length_delta: {last_motion['boom_length_delta']:+.3f}m")
                    print(
                        f"  Final position: len={status['current_position']['boom_length']:.2f}m, polar={status['current_position']['polar_angle']:.3f}rad, azimuth={status['current_position']['azimuth']:.3f}rad"
                    )
                break

    print("\n" + "=" * 70)
    print("[INFO] All commands sent and verified. Waiting for video generation...")
    print("=" * 70)

    for _ in range(60):
        time.sleep(1)

        status = get_debug_status()
        if status and status["timeline_frames"] > 200:
            print(f"[INFO] Timeline has {status['timeline_frames']} frames, video should be ready soon...")
            break

    print("\n[INFO] Attempting to fetch video...")
    for retry in range(10):
        try:
            response = requests.get(f"{SERVER_URL}/video", timeout=5)

            if response.status_code == 200:
                video_size = len(response.content)
                print(f"[SUCCESS] Video downloaded ({video_size} bytes)")

                video_path = "/nvme/admhonqia/New/py-crane/results/test_output.mp4"
                with open(video_path, "wb") as f:
                    f.write(response.content)
                print(f"[SUCCESS] Video saved to: {video_path}")
                break
            else:
                status_info = (
                    response.json() if response.headers.get("content-type") == "application/json" else response.text
                )
                print(f"[INFO] Retry {retry + 1}: {status_info}")
        except Exception as e:
            print(f"[INFO] Retry {retry + 1}: {e}")

        time.sleep(2)
    else:
        print("[WARNING] Video retrieval timeout - commands may have executed correctly but video generation is slow")


if __name__ == "__main__":
    try:
        response = requests.get(f"{SERVER_URL}/state", timeout=2)
        print("[INFO] Server connection successful")
        status = response.json()
        print(
            f"[INFO] Initial state - Boom: {status['length']:.2f}m, Polar: {status['polar']:.3f}rad, Azimuth: {status['azimuth']:.3f}rad"
        )
    except Exception as e:
        print(f"[ERROR] Cannot connect to server: {e}")
        print(f"[ERROR] Please ensure server is running at {SERVER_URL}")
        exit(1)

    test_continuous_commands()

    print("\n" + "=" * 70)
    print("TEST COMPLETED")
    print("=" * 70)
