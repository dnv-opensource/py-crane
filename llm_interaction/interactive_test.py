import time

import requests

BASE_URL = "http://localhost:9100"

print("=" * 70)
print(" Testing Crane Commands (Sequential Execution)")
print("=" * 70)

commands = [
    ("Lift boom up 30 degrees within 5 seconds", 2),
    ("Rotate boom 120 degrees clockwise in 10 seconds", 5),
    # ("Extend boom length to 35 meters over 5 seconds", 2),
]

for i, (cmd_text, wait_time) in enumerate(commands, 1):
    print(f"\n[{i}/3] Sending: {cmd_text}")
    print("-" * 70)

    response = requests.post(f"{BASE_URL}/command", json={"text": cmd_text})

    print(f" Response: {response.json()}")
    print(f" Waiting {wait_time}s for execution...\n")

    # Monitor status while waiting
    for _remaining in range(wait_time, 0, -1):
        status = requests.get(f"{BASE_URL}/status").json()
        print(
            f"  t={status['simulation_time']:.2f}s | Frames={status['timeline_frames']} | Active={status['active_commands']}"
        )

        if status["motion_delta"]:
            delta = status["motion_delta"]
            print(
                f"    Motion: L={delta['len']:+.2f}m, θ={delta['polar']:+.4f}rad, ψ={delta['az']:+.4f}rad, w={delta['wire']:+.2f}m"
            )

        for detail in status["command_details"]:
            state = "DONE" if detail["finished"] else ("RUN" if detail["started"] else "WAIT")
            print(f"[{detail['type']}] {state}")

        time.sleep(1)

print("\n" + "=" * 70)
print(" All commands sent!")
print("=" * 70)

# Final status
status = requests.get(f"{BASE_URL}/status").json()
print(f"\nFinal Simulation Time: {status['simulation_time']:.2f}s")
print(f"Total Frames: {status['timeline_frames']}")

if status["motion_delta"]:
    delta = status["motion_delta"]
    print("\nTotal Motion:")
    print(f"  • Boom length change: {delta['len']:+.2f}m")
    print(f"  • Polar angle change: {delta['polar']:+.4f}rad")
    print(f"  • Azimuth change: {delta['az']:+.4f}rad")
    print(f"  • Wire length change: {delta['wire']:+.2f}m")

print("\n🎥 Matplotlib window shows live animation")
print("📊 Close window to stop server")
