import threading
import time
import os
import uvicorn
from typing import List
import numpy as np
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
from src.py_crane.mobile_crane import MobileCrane
from py_crane.animation import AnimateCrane
from llm_interaction.physics_executor import PhysicsExecutor
from llm_interaction.llm_parser import LLMCommandParser
import tempfile
import shutil

# ===============================
# CONFIG
# ===============================

DT = 0.02
VIDEO_PATH = "results/live_crane.mp4"
VIDEO_TEMP_PATH = "results/live_crane_temp.mp4"

# ===============================
# GLOBAL STATE
# ===============================

app = FastAPI()

crane = MobileCrane()
executor = PhysicsExecutor(crane)

# 初始化 crane
booms = list(crane.booms())
booms[1].boom_setter([3.0, 0, 0])
booms[2].boom_setter([30.0, np.radians(60), 0])
booms[3].boom_setter([20.0, None, None])
crane.calc_statics_dynamics(None)
booms[3].pendulum_relax()

current_commands: List = []
command_lock = threading.Lock()

sim_time = 0.0

timeline = []
timeline_lock = threading.Lock()

last_video_frame_count = 0
video_gen_lock = threading.Lock()

# ===============================
# LLM PARSER
# ===============================

parser = LLMCommandParser(
    api_key="",
    backend="azure",
    azure_endpoint="",
    azure_deployment="gpt-4o",
    azure_api_version="",
)

# ===============================
# REQUEST
# ===============================


class CommandRequest(BaseModel):
    text: str


# ===============================
# API
# ===============================


@app.post("/command")
def set_command(req: CommandRequest):
    global current_commands

    print(f"\n[COMMAND RECEIVED] {req.text}")

    try:
        cmds = parser.parse_command(req.text, current_time=0.0)
    except Exception as e:
        print(f"[ERROR] LLM parsing failed: {e}")
        return {"status": "error", "msg": str(e)}

    with command_lock:
        for cmd in cmds:
            cmd.start_time = sim_time
            print(f"[COMMAND PARSED] Type={cmd.type.value}, start_time={cmd.start_time:.2f}s, duration={cmd.duration}s")
        
        current_commands = cmds
        executor.command_state.clear()
        executor.rotation_tracker.clear()

    return {"status": "ok", "count": len(cmds)}


@app.get("/video")
def get_video():
    if os.path.exists(VIDEO_PATH):
        return FileResponse(VIDEO_PATH, media_type="video/mp4")
    else:
        return {"status": "video_generating", "message": "Video is being generated. Please try again in a few seconds."}


@app.get("/state")
def state():
    return {
        "length": float(executor.boom.boom[0]),
        "polar": float(executor.boom.boom[1]),
        "azimuth": float(executor.pedestal.boom[2]),
    }


@app.get("/debug")
def debug_status():
    """Debug endpoint to verify command execution and timeline data"""
    with timeline_lock:
        timeline_len = len(timeline)
        if timeline_len > 0:
            first_frame = timeline[0]
            last_frame = timeline[-1]
            motion_summary = {
                "boom_length_delta": last_frame["len"] - first_frame["len"],
                "polar_delta": last_frame["polar"] - first_frame["polar"],
                "azimuth_delta": last_frame["az"] - first_frame["az"],
                "wire_delta": last_frame["wire"] - first_frame["wire"],
            }
        else:
            motion_summary = None
    
    with command_lock:
        active_cmds = len(current_commands)
        cmd_details = []
        for cmd in current_commands:
            cid = id(cmd)
            cmd_state = executor.command_state.get(cid, {})
            cmd_details.append({
                "type": cmd.type.value,
                "start_time": cmd.start_time,
                "duration": cmd.duration,
                "started": cmd_state.get("started", False),
                "finished": cmd_state.get("finished", False),
            })
    
    return {
        "simulation_time": float(sim_time),
        "timeline_frames": timeline_len,
        "motion_detected": motion_summary,
        "active_commands": active_cmds,
        "command_details": cmd_details,
        "current_position": {
            "boom_length": float(executor.boom.boom[0]),
            "polar_angle": float(executor.boom.boom[1]),
            "azimuth": float(executor.pedestal.boom[2]),
            "wire_length": float(executor.rope.boom[0]),
        }
    }


# ===============================
# SIMULATION LOOP
# ===============================


def simulation_loop():
    global sim_time

    previous_state = None
    step_counter = 0
    command_completion_logged = {}

    while True:
        with command_lock:
            cmds = list(current_commands)

        for cmd in cmds:
            executor.execute_command(cmd, sim_time, DT)

        executor.step(sim_time, DT)
        crane.do_step(sim_time, DT)

        current_state = {
            "len": executor.boom.boom[0],
            "polar": executor.boom.boom[1],
            "az": executor.pedestal.boom[2],
            "wire": executor.rope.boom[0],
        }

        with timeline_lock:
            timeline.append({
                "t": sim_time,
                **current_state
            })

            if len(timeline) > 5000:
                timeline.pop(0)

        step_counter += 1
        
        if step_counter % 50 == 0:
            print(f"\n[STATUS] SIM_TIME={sim_time:.2f}s | Active_Commands={len(cmds)} | Timeline_Frames={len(timeline)}")
            print(f"[POSITION] Boom_Length={current_state['len']:.2f}m | Polar_Angle={current_state['polar']:.3f}rad | Azimuth={current_state['az']:.3f}rad | Wire_Length={current_state['wire']:.2f}m")
            
            if previous_state:
                delta_len = current_state['len'] - previous_state['len']
                delta_polar = current_state['polar'] - previous_state['polar']
                delta_az = current_state['az'] - previous_state['az']
                delta_wire = current_state['wire'] - previous_state['wire']
                
                if abs(delta_len) > 0.01 or abs(delta_polar) > 0.001 or abs(delta_az) > 0.001 or abs(delta_wire) > 0.01:
                    print(f"[MOTION] Boom_Delta={delta_len:+.3f}m | Polar_Delta={delta_polar:+.4f}rad | Azimuth_Delta={delta_az:+.4f}rad | Wire_Delta={delta_wire:+.3f}m")
            
            for cmd in cmds:
                cid = id(cmd)
                if cid in executor.command_state:
                    state = executor.command_state[cid]
                    cmd_name = cmd.type.value
                    
                    if not state.get("started", False):
                        print(f"[COMMAND] {cmd_name} - WAITING (start_time={cmd.start_time:.2f}s, current_time={sim_time:.2f}s)")
                    elif not state.get("finished", False):
                        print(f"[COMMAND] {cmd_name} - EXECUTING (progress={((sim_time - cmd.start_time) / cmd.duration * 100) if cmd.duration else 0:.1f}%)")
                    else:
                        if cid not in command_completion_logged:
                            command_completion_logged[cid] = True
                            print(f"[COMMAND] {cmd_name} - COMPLETED at t={sim_time:.2f}s")

            previous_state = current_state.copy()

        sim_time += DT
        time.sleep(DT)


# ===============================
# VIDEO GENERATION
# ===============================


def video_loop():
    global last_video_frame_count
    
    while True:
        time.sleep(1)

        with timeline_lock:
            current_frame_count = len(timeline)
            if current_frame_count < 50:
                continue
            
            if current_frame_count <= last_video_frame_count:
                continue
                
            frames = list(timeline)

        with video_gen_lock:
            print(f"\n[VIDEO] Generating video from frame {last_video_frame_count} to {current_frame_count}")
            
            check_motion = len(frames) > 1
            if check_motion:
                first_frame = frames[0]
                last_frame = frames[-1]
                motion_detected = (
                    abs(first_frame["len"] - last_frame["len"]) > 0.1 or
                    abs(first_frame["polar"] - last_frame["polar"]) > 0.01 or
                    abs(first_frame["az"] - last_frame["az"]) > 0.01 or
                    abs(first_frame["wire"] - last_frame["wire"]) > 0.1
                )
                
                print(f"[VIDEO] Frame range: {first_frame['t']:.2f}s to {last_frame['t']:.2f}s")
                print(f"[VIDEO] Motion check - Boom_len: {first_frame['len']:.2f}m->{last_frame['len']:.2f}m, Polar: {first_frame['polar']:.3f}->{last_frame['polar']:.3f}, Azimuth: {first_frame['az']:.3f}->{last_frame['az']:.3f}")
                print(f"[VIDEO] Motion detected: {motion_detected}")

            playback = MobileCrane()
            booms = list(playback.booms())

            booms[1].boom_setter([3.0, 0, 0])
            booms[2].boom_setter([30.0, np.radians(60), 0])
            booms[3].boom_setter([20.0, None, None])
            playback.calc_statics_dynamics(None)
            booms[3].pendulum_relax()

            def movement(c, dt, t_end):
                b = list(c.booms())
                
                sample_interval = max(1, len(frames) // 10)
                frame_index = 0

                for frame_idx, f in enumerate(frames):
                    b[2].boom_setter([f["len"], f["polar"], None])
                    b[1].boom_setter([None, None, f["az"]])
                    b[3].boom_setter([f["wire"], None, None])

                    c.calc_statics_dynamics(None)

                    if frame_idx % sample_interval == 0 or frame_idx == len(frames) - 1:
                        print(f"[VIDEO] Frame {frame_idx}/{len(frames)}: t={f['t']:.2f}s, len={f['len']:.2f}m, polar={f['polar']:.3f}, az={f['az']:.3f}, wire={f['wire']:.2f}m")
                    
                    yield (f["t"], c)

            animator = AnimateCrane(
                crane=playback,
                movement=movement,
                dt=DT,
                t_end=frames[-1]["t"] if frames else 1.0,
                figsize=(10, 8),
                axes_lim=((-60, 60), (-60, 60), (0, 70)),
                interval=10,
                title="Crane Replay",
            )

            os.makedirs("results", exist_ok=True)

            try:
                print("[VIDEO] Starting animation rendering and saving...")
                animator.save_animation(VIDEO_TEMP_PATH)
                
                if os.path.exists(VIDEO_TEMP_PATH):
                    shutil.move(VIDEO_TEMP_PATH, VIDEO_PATH)
                    last_video_frame_count = current_frame_count
                    print(f"[VIDEO] SUCCESS - Video saved with {current_frame_count} frames")
                else:
                    print("[VIDEO] ERROR - Temp video file not created")
            except Exception as e:
                print(f"[VIDEO] ERROR during save: {e}")
                import traceback
                traceback.print_exc()
            finally:
                animator.close()


# ===============================
# START THREADS
# ===============================

threading.Thread(target=simulation_loop, daemon=True).start()
threading.Thread(target=video_loop, daemon=True).start()

# ===============================
# MAIN
# ===============================

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9100)
