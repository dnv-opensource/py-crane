import os
import threading
import time

import matplotlib.pyplot as plt
import numpy as np
from flask import Flask, jsonify, request
from llm_interaction.llm_parser import LLMCommandParser
from llm_interaction.physics_executor import PhysicsExecutor
from src.py_crane.mobile_crane import MobileCrane

# ===============================
# CONFIG
# ===============================

DT = 0.02

# ===============================
# GLOBAL STATE
# ===============================

crane = MobileCrane()
executor = PhysicsExecutor(crane)

# Initialize crane
booms = list(crane.booms())
booms[1].boom_setter([3.0, 0, 0])
booms[2].boom_setter([30.0, np.radians(60), 0])
booms[3].boom_setter([20.0, None, None])
crane.calc_statics_dynamics(None)
booms[3].pendulum_relax()

current_commands: list = []
command_lock = threading.Lock()

sim_time = 0.0
sim_time_lock = threading.Lock()

timeline = []
timeline_lock = threading.Lock()

# ===============================
# LLM PARSER
# ===============================

parser = LLMCommandParser(
    api_key=os.getenv("AZURE_KEY", ""),
    backend="azure",
    azure_endpoint=os.getenv("AZURE_ENDPOINT", ""),
    azure_deployment="gpt-4o",
    azure_api_version="2024-12-01-preview",
)

# ===============================
# FLASK APP
# ===============================

app = Flask(__name__)


@app.route("/command", methods=["POST"])
def queue_command():
    """Queue a command for execution."""
    data = request.get_json()
    text = data.get("text", "")

    print(f"\n[CMD] Received: {text}")

    try:
        cmds = parser.parse_command(text, current_time=0.0)
        print(f"[CMD] Parsed {len(cmds)} command(s)")

        executor.controls["pedestal_azimuth"].setgoal(1, None)
        executor.controls["boom_polar"].setgoal(1, None)
        executor.controls["boom_length"].setgoal(1, None)
        executor.controls["wire_length"].setgoal(1, None)

        with command_lock:
            with sim_time_lock:
                current_sim_time = sim_time

            current_commands.clear()

            for cmd in cmds:
                cmd.start_time = current_sim_time
                current_commands.append(cmd)
                print(f"[CMD] Queued: {cmd.type.value} (start_time={cmd.start_time:.2f}s, duration={cmd.duration}s)")

        executor.command_state.clear()
        executor.rotation_tracker.clear()

        return jsonify({"status": "ok", "commands": len(cmds)})

    # try:
    #     cmds = parser.parse_command(text, current_time=0.0)
    #     print(f"[CMD] Parsed {len(cmds)} command(s)")

    #     with command_lock:
    #         with sim_time_lock:
    #             current_sim_time = sim_time

    #         for cmd in cmds:
    #             cmd.start_time = current_sim_time
    #             current_commands.append(cmd)
    #             print(f"[CMD] Queued: {cmd.type.value} (start_time={cmd.start_time:.2f}s, duration={cmd.duration}s)")

    #     executor.command_state.clear()
    #     executor.rotation_tracker.clear()

    #     return jsonify({"status": "ok", "commands": len(cmds)})

    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback

        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 400


@app.route("/status", methods=["GET"])
@app.route("/debug", methods=["GET"])
def get_status():
    """Get current simulation status."""
    with sim_time_lock:
        t = sim_time
    with timeline_lock:
        frames = len(timeline)
        if frames > 0:
            first = timeline[0]
            last = timeline[-1]
            delta = {
                "len": last["len"] - first["len"],
                "polar": last["polar"] - first["polar"],
                "az": last["az"] - first["az"],
                "wire": last["wire"] - first["wire"],
            }
        else:
            delta = None

    with command_lock:
        cmds = len(current_commands)
        details = []
        for cmd in current_commands:
            cid = id(cmd)
            state = executor.command_state.get(cid, {})
            details.append(
                {
                    "type": cmd.type.value,
                    "start_time": cmd.start_time,
                    "duration": cmd.duration,
                    "started": state.get("started", False),
                    "finished": state.get("finished", False),
                }
            )

    return jsonify(
        {
            "simulation_time": float(t),
            "timeline_frames": frames,
            "motion_delta": delta,
            "active_commands": cmds,
            "command_details": details,
        }
    )


# ===============================
# SIMULATION LOOP
# ===============================
def simulation_loop():
    """Run main simulation loop."""
    global sim_time

    print("[SIM] Started")
    step_counter = 0
    completed_cmds = set()

    while True:
        with command_lock:
            cmds = list(current_commands)

        # Debug output every 50 steps
        if step_counter % 50 == 0 and len(cmds) > 0:
            print(f"\n[SIM] t={sim_time:.2f}s | {len(cmds)} active")
            for cmd in cmds:
                cid = id(cmd)
                state = executor.command_state.get(cid, {})
                started = state.get("started", False)
                finished = state.get("finished", False)
                s = "DONE" if finished else ("RUN" if started else "WAIT")
                progress = ((sim_time - cmd.start_time) / cmd.duration * 100) if cmd.duration else 0
                print(f"  {cmd.type.value}: {s} (t_start={cmd.start_time:.2f}s, progress={progress:.0f}%)")

        # Execute commands
        for cmd in cmds:
            cid = id(cmd)
            try:
                executor.execute_command(cmd, sim_time, DT)

                # Check if command just finished
                cmd_state = executor.command_state.get(cid, {})
                if cmd_state.get("finished", False) and cid not in completed_cmds:
                    completed_cmds.add(cid)
                    print(f"[CMD] COMPLETED: {cmd.type.value}")

            except Exception as e:
                print(f"[ERROR] {e}")
                import traceback

                traceback.print_exc()

        # Step simulation
        executor.step(sim_time, DT)
        crane.do_step(sim_time, DT)

        # Record state
        with timeline_lock:
            timeline.append(
                {
                    "t": sim_time,
                    "len": executor.boom.boom[0],
                    "polar": executor.boom.boom[1],
                    "az": executor.pedestal.boom[2],
                    "wire": executor.rope.boom[0],
                }
            )
            if len(timeline) > 5000:
                timeline.pop(0)

        # Remove finished commands from queue (FIXED: modify in-place)
        with command_lock:
            # Find which commands are finished
            to_remove = []
            for cmd in current_commands:
                cid = id(cmd)
                state = executor.command_state.get(cid, {})
                if state.get("finished", False):
                    to_remove.append(cmd)
                    print(f"[QUEUE] Removing completed: {cmd.type.value}")

            # Remove them
            for cmd in to_remove:
                current_commands.remove(cmd)

        step_counter += 1
        with sim_time_lock:
            sim_time += DT
        time.sleep(DT)


# ===============================
# ANIMATION LOOP
# ===============================


def animation_loop():
    """Show real-time matplotlib window."""

    print("[ANIM] Opening window...")

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")
    fig.tight_layout()

    ax.set_xlim(-60, 60)
    ax.set_ylim(-60, 60)
    ax.set_zlim(0, 70)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")

    last_idx = 0
    frame_count = 0

    try:
        while plt.fignum_exists(fig.number):
            with timeline_lock:
                if len(timeline) > last_idx:
                    latest = timeline[-1]
                    last_idx = len(timeline)
                else:
                    latest = None

            if latest:
                try:
                    # Create temp crane
                    playback = MobileCrane()
                    b_list = list(playback.booms())
                    b_list[1].boom_setter([3.0, 0, 0])
                    b_list[2].boom_setter([30.0, np.radians(60), 0])
                    b_list[3].boom_setter([20.0, None, None])
                    playback.calc_statics_dynamics(None)
                    b_list[3].pendulum_relax()

                    # Update to current state
                    b = list(playback.booms())
                    b[2].boom_setter([latest["len"], latest["polar"], None])
                    b[1].boom_setter([None, None, latest["az"]])
                    b[3].boom_setter([latest["wire"], None, None])
                    playback.calc_statics_dynamics(None)

                    # Draw
                    ax.clear()
                    for boom in playback.booms():
                        color = {"pedestal": "brown", "rope": "red"}.get(boom.name, "blue")
                        lw = {"pedestal": 10, "rope": 2}.get(boom.name, 3)
                        ax.plot(
                            [boom.origin[0], boom.end[0]],
                            [boom.origin[1], boom.end[1]],
                            [boom.origin[2], boom.end[2]],
                            color=color,
                            linewidth=lw,
                            alpha=0.85,
                        )

                    ax.set_xlim(-35, 35)
                    ax.set_ylim(-35, 35)
                    ax.set_zlim(0, 25)
                    ax.set_xlabel("X (m)")
                    ax.set_ylabel("Y (m)")
                    ax.set_zlabel("Z (m)")
                    ax.set_title(
                        f"t={latest['t']:.2f}s | L={latest['len']:.1f}m | θ={latest['polar']:.2f}rad | ψ={latest['az']:.2f}rad"
                    )

                    plt.draw()
                    plt.pause(0.01)

                    frame_count += 1

                except Exception as e:
                    print(f"[ANIM ERROR] {e}")
            else:
                plt.pause(0.01)

    except Exception:  # noqa: BLE001
        pass

    print("[ANIM] Window closed")
    plt.close("all")


# ===============================
# INPUT LOOP (TERMINAL INPUT)
# ===============================


def input_loop():
    """Read and execute commands from terminal input."""
    global sim_time

    print("[INPUT] Ready for terminal input")
    print("[INPUT] Type commands and press Enter (Ctrl+C to stop)\n")

    try:
        text = input("> ")
        cmds = parser.parse_command(text, current_time=0.0)
        print(f"[INPUT] Parsed {len(cmds)} command(s)")

        # ★ 立即停止所有旧运动 ★
        executor.controls["pedestal_azimuth"].setgoal(1, None)
        executor.controls["boom_polar"].setgoal(1, None)
        executor.controls["boom_length"].setgoal(1, None)
        executor.controls["wire_length"].setgoal(1, None)

        with command_lock:
            with sim_time_lock:
                current_sim_time = sim_time

            current_commands.clear()

            for cmd in cmds:
                cmd.start_time = current_sim_time
                current_commands.append(cmd)
                print(f"[INPUT] Queued: {cmd.type.value} (start_time={cmd.start_time:.2f}s, duration={cmd.duration}s)")

            executor.command_state.clear()
            executor.rotation_tracker.clear()
            print("[INPUT] Ready for next command\n")

    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback

        traceback.print_exc()
        print()


# ===============================
# MAIN
# ===============================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("Crane Simulator with Real-time Animation")
    print("=" * 70)
    print(f"API Key: {bool(os.getenv('AZURE_KEY'))}")
    print(f"Endpoint: {os.getenv('AZURE_ENDPOINT', 'NOT SET')}")
    print("\nHTTP API:")
    print("  POST /command    {'text': '...'}")
    print("  GET  /status or /debug")
    print("=" * 70 + "\n")

    # Start simulation thread
    sim_thread = threading.Thread(target=simulation_loop, daemon=True)
    sim_thread.start()
    print("[SIM] Simulation thread started")
    time.sleep(0.5)

    # Start Flask in background thread
    def run_flask():
        app.run(host="0.0.0.0", port=9100, threaded=True, use_reloader=False, debug=False)

    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()
    print("[FLASK] Server running on http://127.0.0.1:9100")
    time.sleep(1)

    # Start terminal input thread
    input_thread = threading.Thread(target=input_loop, daemon=False)
    input_thread.start()
    print("[INPUT] Terminal input thread started\n")

    # Run animation in main thread
    print("[ANIM] Starting matplotlib window...\n")
    animation_loop()

    print("\n[MAIN] Shutdown")
