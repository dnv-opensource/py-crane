import numpy as np
import matplotlib
from src.py_crane.mobile_crane import MobileCrane
from src.py_crane.animation import AnimateCrane
from llm_interaction.llm_parser import LLMCommandParser
from llm_interaction.physics_executor import PhysicsExecutor

matplotlib.use("Agg")


def llm_driven_simulation(crane: MobileCrane, commands: list, dt: float = 0.01, t_end: float = 30.0):
    """Intergrating [Controls] system"""
    executor = PhysicsExecutor(crane)
    booms = list(crane.booms())

    booms[1].boom_setter([3.0, 0, 0])
    booms[2].boom_setter([30.0, np.radians(60), 0])
    booms[3].boom_setter([20.0, None, None])
    crane.calc_statics_dynamics(None)
    booms[3].pendulum_relax()

    for time in np.linspace(0, t_end, int(t_end / dt) + 1):
        for cmd in commands:
            executor.execute_command(cmd, time, dt)

        executor.step(time, dt)
        crane.do_step(time, dt)

        yield (time, crane)


if __name__ == "__main__":
    ## Choose backend: "qwen" or "azure"
    backend = "azure"

    ## Initialize LLM parser
    api_key = ""  

    if backend == "azure":
        parser = LLMCommandParser(
            api_key=api_key,
            backend="azure",
            azure_endpoint="",
            azure_deployment="gpt-4o",  # deployment name on Azure
            azure_api_version="",
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
    user_command = "Rotate the boom 90 degrees clockwise within 10 seconds"
    print(f"User: {user_command}\n")

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
        t_end=10.0,
        figsize=(12, 10),
        axes_lim=((-60, 60), (-60, 60), (0, 70)),
        interval=10,
        title="LLM-Driven Crane Simulation",
    )

    print("\n Generating animation...\n")
    animator.save_animation("/nvme/admhonqia/New/py-crane/results/crane_llm_driven_2000.mp4")
    print("Animation saved!!")
