import requests
import json
from typing import List
from llm_interaction.llm_command import Command, CommandType


class LLMCommandParser:
    """
    Parse natural language to physics commands using LLM
    Supports:
        - Azure OpenAI
        - Qwen (custom endpoint)
        - OpenAI
    """

    def __init__(
        self,
        api_key: str,
        backend: str = "azure",
        azure_endpoint: str = None,
        azure_deployment: str = None,
        azure_api_version: str = None,
        api_url: str = None,
        model: str = "gpt-4o-mini",
    ):
        self.api_key = api_key
        self.backend = backend

        # Azure
        self.azure_endpoint = azure_endpoint
        self.azure_deployment = azure_deployment
        self.azure_api_version = azure_api_version

        #  Qwen / custom
        self.api_url = api_url

        #  OpenAI
        self.model = model


    def _build_prompt(self, user_input: str) -> str:
        return f"""
    You are a physics command parser for a crane simulation.

==============================
CRITICAL RULES (VERY IMPORTANT)
==============================

- ALL angles are RELATIVE.
- NEVER interpret any angle as absolute.
- ALWAYS treat angles as a CHANGE (delta).

- Coordinate system:
  X: right (+) / left (-)
  Y: forward (+) / backward (-)
  Z: up (+) / down (-)

- Output ONLY valid JSON array (no explanation).

- Convert degrees → radians:
  radians = degrees × 0.017453

- ALWAYS include:
  - type
  - target
  - start_time
  - duration (if applicable)

==============================
COMMAND DEFINITIONS
==============================

--------------------------------------------------
1) BOOM_ROTATE (horizontal rotation)

ALWAYS relative rotation.

Example:
"Rotate the boom 60 degrees clockwise within 5 seconds"

Calculation:
- angle = -1.0472
- angular_velocity = -0.20944

Output:
[
  {{
    "type": "boom_rotate",
    "target": "boom",
    "angle": -1.0472,
    "angular_velocity": -0.20944,
    "start_time": 0,
    "duration": 5
  }}
]

--------------------------------------------------
2) BOOM_LUFF (lift up / down)

ALWAYS relative angle change.

Rules:
- Raise / Lift → positive angle
- Lower → negative angle

Example:
"Raise the boom by 20 degrees in 4 seconds"

Calculation:
- angle = +0.3491
- angular_velocity = +0.0873

Output:
[
  {{
    "type": "boom_luff",
    "target": "boom",
    "angle": 0.3491,
    "angular_velocity": 0.0873,
    "start_time": 0,
    "duration": 4
  }}
]

--------------------------------------------------
3) BOOM_EXTEND (extend / retract boom)

ALWAYS relative length change.

Rules:
- Extend → positive velocity
- Retract → negative velocity

Example:
"Extend the boom by 10 meters in 5 seconds"

Calculation:
- velocity = 10 / 5 = 2.0

Output:
[
  {{
    "type": "boom_extend",
    "target": "boom",
    "velocity": 2.0,
    "start_time": 0,
    "duration": 5
  }}
]

--------------------------------------------------
4) EXTERNAL_FORCE

Example:
"Apply 5N force forward for 3 seconds"

Output:
[
  {{
    "type": "external_force",
    "target": "crane",
    "force": [0, 5, 0],
    "start_time": 0,
    "duration": 3
  }}
]

--------------------------------------------------
5) WIRE_INIT_VELOCITY

Example:
"Give the load an initial velocity to the right at 2 m/s"

Output:
[
  {{
    "type": "wire_init_velocity",
    "target": "wire",
    "velocity": [2, 0, 0],
    "start_time": 0
  }}
]

==============================
USER INPUT
==============================

{user_input}

==============================
OUTPUT JSON ONLY
==============================

    """

    # ------------------------------------------------------------------

    def parse_command(self, user_input: str, current_time: float = 0.0) -> List[Command]:

        prompt = self._build_prompt(user_input)

        # =========================
        # Backend switch
        # =========================

        if self.backend == "azure":
            url = f"{self.azure_endpoint}/openai/deployments/{self.azure_deployment}/chat/completions?api-version={self.azure_api_version}"

            headers = {
                "Content-Type": "application/json",
                "api-key": self.api_key,
            }

            payload = {
                "messages": [
                    {"role": "system", "content": "You convert text into JSON commands."},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.0,
            }

        elif self.backend == "qwen":
            url = self.api_url

            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            }

            payload = {
                "model": "qwen",
                "messages": [
                    {"role": "system", "content": "You convert text into JSON commands."},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.0,
            }

        else:  # OpenAI
            url = ""

            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            }

            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": "You convert text into JSON commands."},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.0,
            }

        # =========================
        # Request
        # =========================

        response = requests.post(url, headers=headers, data=json.dumps(payload))
        result_text = response.json()["choices"][0]["message"]["content"].strip()

        try:
            data = json.loads(result_text)
        except Exception:
            print(" JSON parse error:", result_text)
            return []

        commands = []

        for item in data:
            try:
                cmd = Command(
                    type=CommandType(item["type"]),
                    target=item.get("target", ""),
                    start_time=item.get("start_time", current_time),
                    duration=item.get("duration"),
                    force=item.get("force"),
                    velocity=item.get("velocity"),
                    angle=item.get("angle"),
                    angular_velocity=item.get("angular_velocity"),
                )

                commands.append(cmd)

            except Exception:
                print("Command parse error:", item)

        return commands
