## Project Overview
Summary: Three capabilities for crane simulation: 
(1) control crane motion with natural language via an LLM, 
(2) generate crane configurations from natural language, 
(3) run a continuous interactive server for streaming commands and saving replay videos. 

## Features & Where to Find Them
* Natural-language motion control: parse & execute LLM commands (offline examples)
     Files: llm_parser.py:1, physics_executor.py:1, demo_llm_control.py:1, llm_command.py:1
* LLM-based crane configuration: build crane from a text spec and render animation
     File: configuration_llm.py:1
* Continuous interactive server (portal): send multiple commands while service runs; server    
     logs and /debug endpoint provide real-time status; video saved to results/live_crane.mp4
     Server: crane_server.py:1
     Test/verification script: test_crane_server.py:1 (or test_crane_server_v2.py)
     Detailed operations / instructions: SERVER_START_HERE.md:1

## How to validate (brief)
* Offline flows: run the demo or configuration scripts; they save MP4 outputs under results/.
* Server flow: start crane_server.py, send commands to /command, poll /debug for motion_detected and command status, and download /video when ready. Logs print periodic [STATUS], [POSITION], [MOTION], [COMMAND] lines to confirm execution without the video.


## Notes
Due to the limitation of my hardware, my experimental environment cannot show interactive real-time GUI; the code saves rendered animations (MP4) for verification.
The server prints live textual status and provides /debug for programmatic verification so you do not need to rely solely on the video.

## Minimal quick commands
* Start server: python py-crane/llm_interaction/crane_server.py
* Test commands: python py-crane/llm_interaction/test_crane_server.py
* Build/configure crane: python py-crane/llm_interaction/configuration_llm.py
* Offline demo: python py-crane/llm_interaction/demo_llm_control.py
