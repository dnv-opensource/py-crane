import json
import math
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import requests
from dotenv import load_dotenv
from matplotlib.animation import FuncAnimation

from py_crane.crane import Crane

load_dotenv()
# 改为支持窗口显示的后端
matplotlib.use("TkAgg")

api_key = os.getenv("AZURE_KEY")
azure_endpoint = os.getenv("AZURE_ENDPOINT")


# =====================
# LLM Config
# =====================

BACKEND = "azure"  #  "azure" or "qwen"

# --- Azure ---
AZURE_ENDPOINT = azure_endpoint
AZURE_API_KEY = api_key
AZURE_API_VERSION = "2024-12-01-preview"
AZURE_DEPLOYMENT = "gpt-4o"

# --- Qwen ---
QWEN_API_URL = ""
QWEN_API_KEY = ""


# =====================
# Test Queries
# =====================

TEST_QUERIES = [
    # "A small crane with a pedestal, one boom 5 meters long pointing forward and a 10 meter wire pointing down, no movement.",
    "A small crane with a pedestal which is 2 meters long, with a single boom 10 m long, luffed up to 60 degrees, and a 7 m wire pointing down, no movement.",
    # "Two-segment crane: pedestal, boom1 4 m at 30 degrees, boom2 6 m at 170 degrees, and a 1.5 m wire.",
    # "Knuckle-style crane: pedestal, boom1 3 m straight up, boom2 5 m at 45 degrees backward, boom3 7 m at 45 degrees forward, wire 2 m.",
    # "Heavy-duty base: pedestal mass 2000 kg with a 15 m horizontal boom and a 5 m wire.",
]


# =====================
# Prompt
# =====================

CREATION_PROMPT = """
You are given a natural language description of a crane.

Output ONLY a JSON array.

Each item describes one crane element:

- name: string
- type: ["pedestal","boom","wire"]
- length: meters
- polar_deg: elevation angle (degrees)
- azimuth_deg: horizontal direction (degrees)
- mass: optional
- mass_center: optional
- q_factor: optional

For wire:
- polar_deg and azimuth_deg are not physically meaningful

Example:

[
  {"name":"pedestal","type":"pedestal","length":1.0,"polar_deg":0,"azimuth_deg":0,"mass":2000},
  {"name":"boom1","type":"boom","length":10.0,"polar_deg":90,"azimuth_deg":0},
  {"name":"wire","type":"wire","length":1.0,"polar_deg":90,"azimuth_deg":0}
]

ONLY output JSON.
"""


# =====================
# Unified LLM Call (REST API)
# =====================


def call_llm(prompt: str, backend: str = "azure") -> str:
    """Unified LLM interface (Azure / Qwen) using REST API."""

    if backend == "azure":
        url = f"{AZURE_ENDPOINT}/openai/deployments/{AZURE_DEPLOYMENT}/chat/completions?api-version={AZURE_API_VERSION}"

        headers = {
            "Content-Type": "application/json",
            "api-key": AZURE_API_KEY,
        }

        payload = {
            "messages": [
                {
                    "role": "system",
                    "content": "You convert text into structured JSON describing a crane. Output only valid JSON.",
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            "temperature": 0.0,
        }

        response = requests.post(url, headers=headers, json=payload)
        return response.json()["choices"][0]["message"]["content"]

    elif backend == "qwen":
        headers = {
            "Authorization": QWEN_API_KEY,
            "Content-Type": "application/json",
        }

        payload = {
            "model": "Qwen3-235B-A22B-Instruct",
            "messages": [
                {
                    "role": "system",
                    "content": "You convert text into structured JSON describing a crane. Output only valid JSON.",
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            "temperature": 0.0,
        }

        response = requests.post(QWEN_API_URL, headers=headers, data=json.dumps(payload))
        return response.json()["choices"][0]["message"]["content"]

    else:
        raise ValueError(f"Unsupported backend: {backend}")


# =====================
# Utils
# =====================


def to_radians(deg):
    return 0.0 if deg is None else float(deg) * math.pi / 180.0


def build_crane_from_spec(spec: list) -> Crane:
    """Build a Crane object from LLM-generated spec."""

    crane = Crane()

    for item in spec:
        # -----------------------------
        # Basic parameters
        # -----------------------------
        name = item.get("name", "boom")
        typ = item.get("type", "boom")

        length = float(item.get("length", 1.0))

        polar_deg = item.get("polar_deg")
        azimuth_deg = item.get("azimuth_deg")

        mass = float(item.get("mass", 100.0))
        mass_center = item.get("mass_center", 0.5)
        q_factor = item.get("q_factor", None)

        # -----------------------------
        # Angle configuration
        # -----------------------------

        if typ == "wire":
            polar = np.pi / 2
            azimuth = 0.0
        else:
            polar = to_radians(polar_deg)
            azimuth = to_radians(azimuth_deg)

        # -----------------------------
        # Construction parameters
        # -----------------------------
        kwargs = {
            "mass": mass,
            "mass_center": mass_center,
            "boom": (length, polar, azimuth),
        }

        if typ == "wire":
            kwargs["q_factor"] = q_factor if q_factor is not None else 10.0

        # -----------------------------
        # Add to crane
        # -----------------------------
        crane.add_boom(name, **kwargs)

    crane.calc_statics_dynamics(None)

    return crane


# =====================
# Visualization
# =====================


def rotating_camera_simulation(crane: Crane, dt=0.01, t_end=3.0):
    for time in np.linspace(0, t_end, int(t_end / dt) + 1):
        crane.do_step(time, dt)
        yield (time, crane)


def show_crane_static(crane: Crane, title: str = "Crane Static View"):
    """Display a static image of the crane using matplotlib."""

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Set axis limits based on crane geometry
    ax.set_xlim(-20, 20)
    ax.set_ylim(-20, 20)
    ax.set_zlim(0, 10)

    # Set labels
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")

    # Draw booms
    for b in crane.booms():
        lw = {"pedestal": 10, "rope": 1}.get(b.name, 4)
        ax.plot(
            [b.origin[0], b.end[0]],
            [b.origin[1], b.end[1]],
            [b.origin[2], b.end[2]],
            linewidth=lw,
            label=b.name,
        )

    # Set view angle
    ax.view_init(elev=20, azim=45)
    ax.legend()
    ax.set_title(title)

    plt.show()


def show_crane_animation(crane: Crane, title: str = "Crane Animation"):
    """Display an animated crane using matplotlib (with rotating camera)."""

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Set axis limits
    ax.set_xlim(-20, 20)
    ax.set_ylim(-20, 20)
    ax.set_zlim(0, 30)

    # Set labels
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")

    # Store line objects
    lines = []
    for b in crane.booms():
        lw = {"pedestal": 10, "rope": 1}.get(b.name, 4)
        line = ax.plot(
            [b.origin[0], b.end[0]],
            [b.origin[1], b.end[1]],
            [b.origin[2], b.end[2]],
            linewidth=lw,
            label=b.name,
        )[0]
        lines.append(line)

    def init():
        return lines

    def update_frame(frame):
        time, crane_updated = frame
        for i, b in enumerate(crane_updated.booms()):
            lines[i].set_data_3d(
                [b.origin[0], b.end[0]],
                [b.origin[1], b.end[1]],
                [b.origin[2], b.end[2]],
            )

        # Rotate camera
        azim = (time / 3.0) * 360.0
        ax.view_init(elev=20, azim=azim)
        ax.set_title(f"{title} ({time:.1f}s)")

        return lines

    # Keep reference to animation object to prevent garbage collection
    ani = FuncAnimation(  # noqa: F841
        fig,
        update_frame,
        frames=rotating_camera_simulation(crane, dt=0.01, t_end=3.0),
        init_func=init,
        interval=10,
        blit=False,
        cache_frame_data=False,
    )

    ax.legend()
    plt.show()


# =====================
# Main
# =====================


def main():
    user_query = TEST_QUERIES[0]

    prompt = CREATION_PROMPT + "\n\n" + user_query

    print("Query:", user_query)
    print("\nCalling LLM...")

    result = call_llm(prompt, backend=BACKEND)

    print("LLM Output:")
    print(result)

    try:
        spec = json.loads(result)
    except Exception as e:
        print(f"JSON parse failed: {e}")
        return

    crane = build_crane_from_spec(spec)
    print("Crane built successfully!")

    # Choose visualization mode
    print("\nVisualization options:")
    print("1. Static image")
    print("2. Animated view (rotating camera)")

    choice = input("Select option (1 or 2, default=1): ").strip() or "1"

    if choice == "2":
        print("Showing animated view...")
        show_crane_animation(crane, "LLM-Generated Crane Animation")
    else:
        print("Showing static view...")
        show_crane_static(crane, "LLM-Generated Crane Static View")


if __name__ == "__main__":
    main()
