import os
import time

import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv
from llm_interaction.llm_parser import LLMCommandParser
from llm_interaction.physics_executor import PhysicsExecutor
from matplotlib.animation import FuncAnimation
from src.py_crane.animation import AnimateCrane
from src.py_crane.mobile_crane import MobileCrane

load_dotenv()

api_key = os.getenv("AZURE_KEY")
azure_endpoint = os.getenv("AZURE_ENDPOINT")


def llm_driven_simulation(crane: MobileCrane, commands: list, dt: float = 0.01, t_end: float = 30.0):
    """Integrate [Controls] system."""
    executor = PhysicsExecutor(crane)
    booms = list(crane.booms())

    booms[1].boom_setter([3.0, 0, 0])
    booms[2].boom_setter([30.0, np.radians(60), 0])
    booms[3].boom_setter([20.0, None, None])
    crane.calc_statics_dynamics(None)
    booms[3].pendulum_relax()

    for t in np.linspace(0, t_end, int(t_end / dt) + 1):
        for cmd in commands:
            executor.execute_command(cmd, t, dt)

        executor.step(t, dt)
        crane.do_step(t, dt)

        yield (t, crane)


if __name__ == "__main__":
    ## Choose backend: "qwen" or "azure"
    backend = "azure"
    if backend == "azure":
        parser = LLMCommandParser(
            api_key=api_key,
            backend="azure",
            azure_endpoint=azure_endpoint,
            azure_deployment="gpt-4o",  # deployment name on Azure
            azure_api_version="2024-12-01-preview",
        )
    else:
        # qwen backend (local server)
        parser = LLMCommandParser(
            api_key=api_key,
            backend="qwen",
            api_url="",
        )

    current_simulation_time = 0.0

    # User input natural language command
    print("\n Natural Language Command:\n")
    # user_command = "Lower the boom by 45 degrees in 5 seconds"
    # user_command = "Extend the boom by 20 meters in 5 seconds"
    user_command = "Rotate the boom 40 degrees clockwise within 5 seconds"
    print("User: {user_command}\n")

    # LLM parsing
    commands = parser.parse_command(user_command, current_time=current_simulation_time)
    print(commands)

    if not commands:
        print(" Failed to parse command, using default action")
        commands = []

    # Create crane
    crane = MobileCrane()

    # Create animator
    animator = AnimateCrane(
        crane=crane,
        movement=lambda c, dt, t_end: llm_driven_simulation(c, commands, dt, t_end),
        dt=0.01,
        t_end=5.0,
        figsize=(6, 5),
        axes_lim=((-60, 60), (-60, 60), (0, 70)),
        interval=10,
        title="LLM-Driven Crane Simulation",
    )

    # print("\n Generating animation...\n")
    # animator.save_animation("/nvme/admhonqia/New/py-crane/results/crane_llm_driven_2000.mp4")
    # print("Animation saved!!")

    # out_path = "C:\Users\HONQIA\Documents\python_workspace\AI_devs\py-crane\videos\crane_llm_driven_2000.mp4"
    from pathlib import Path

    out_dir = Path(r"C:\Users\HONQIA\Documents\python_workspace\AI_devs\py-crane\videos")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "crane_llm_driven_2000.mp4"

    # 构建 FuncAnimation（与 save_animation 中逻辑一致）
    ani = FuncAnimation(
        animator.fig,
        animator.update,
        frames=animator.movement(animator.crane, dt=animator.dt, t_end=animator.t_end, **animator.kwargs),
        init_func=animator.init_fig,
        interval=animator.interval,
        repeat=animator.repeat,
        blit=False,
        cache_frame_data=False,
    )

    plt.show(block=False)

    time.sleep(animator.t_end + animator.dt)

    # writer = "ffmpeg"
    # ani.save(str(out_path), writer=writer, fps=max(1, 1000 // animator.interval))
    # print("Animation saved to", out_path)
    plt.show()
