#!/usr/bin/env python3
"""
configuration_llm.py

Create an arbitrary crane from a natural-language description via LLM,
instantiate a Crane, compute statics, and save a 3D plot image.
"""

import json
import math
import requests
import numpy as np
import matplotlib

matplotlib.use("Agg")
from py_crane.animation import AnimateCrane
import os
from py_crane.crane import Crane
from openai import AzureOpenAI


# =====================
# LLM Config
# =====================

BACKEND = "azure"  # ✅ "azure" or "qwen"

# --- Azure ---
AZURE_ENDPOINT = ""
AZURE_API_KEY = ""
AZURE_API_VERSION = ""
AZURE_DEPLOYMENT = "gpt-4o"

# --- Qwen ---
QWEN_API_URL = ""
QWEN_API_KEY = ""


# =====================
# Test Queries
# =====================

TEST_QUERIES = [
    "A small crane with a pedestal, one boom 5 meters long pointing forward and a 10 meter wire pointing down, no movement.",
    "Pedestal with a single boom 10 m long, luffed up to 45 degrees and a 10 m wire.",
    "Two-segment crane: pedestal, boom1 4 m at 30 degrees, boom2 6 m at 170 degrees, and a 1.5 m wire.",
    "Knuckle-style crane: pedestal, boom1 3 m straight up, boom2 5 m at 45 degrees backward, boom3 7 m at 45 degrees forward, wire 2 m.",
    "Heavy-duty base: pedestal mass 2000 kg with a 15 m horizontal boom and a 5 m wire.",
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


class AnimateCraneWithCamera(AnimateCrane):
    def _update(self, frame):
        time, crane = frame
        artists = super()._update(frame)
        azim = (time / self.t_end) * 360.0
        elev = 25
        self.ax.view_init(elev=elev, azim=azim)
        return artists


# =====================
# Unified LLM Call
# =====================


def call_llm(prompt: str, backend: str = "azure") -> str:
    """
    Unified LLM interface (Azure / Qwen)
    """

    if backend == "azure":
        client = AzureOpenAI(
            api_version=AZURE_API_VERSION,
            azure_endpoint=AZURE_ENDPOINT,
            api_key=AZURE_API_KEY,
        )

        response = client.chat.completions.create(
            model=AZURE_DEPLOYMENT,
            messages=[
                {
                    "role": "system",
                    "content": "You convert text into structured JSON describing a crane. Output only valid JSON.",
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            temperature=0.0,
        )

        return response.choices[0].message.content

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
    """
    Build a Crane object from LLM-generated spec.
    """

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


def render_crane_animation(crane, output="results/llm_crane_02.mp4"):
    os.makedirs("results", exist_ok=True)

    animator = AnimateCraneWithCamera(
        crane=crane,
        movement=lambda c, dt, t_end: rotating_camera_simulation(c, dt, 3.0),
        dt=0.01,
        t_end=3.0,
        figsize=(10, 8),
        axes_lim=((-20, 20), (-20, 20), (0, 30)),
        interval=10,
        title="Rotating View Crane",
    )

    print("Rendering animation...")

    animator.save_animation(output)

    print(f" Saved to {output}")


# =====================
# Main
# =====================


def main():
    user_query = TEST_QUERIES[1]

    prompt = CREATION_PROMPT + "\n\n" + user_query

    print("Query:", user_query)

    result = call_llm(prompt, backend=BACKEND)

    print("LLM Output:")
    print(result)

    try:
        spec = json.loads(result)
    except Exception:
        print("JSON parse failed")
        return

    crane = build_crane_from_spec(spec)

    render_crane_animation(crane)

    print("Crane image saved.")


if __name__ == "__main__":
    main()
