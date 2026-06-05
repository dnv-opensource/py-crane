## Project Overview
LLM interaction for crane simulation,three major capabilities :
(1) Control crane motion with natural language via LLM
(2) Generate crane configurations from natural language descriptions
(3) Run an interactive server for real-time command processing with live visualization

## Features & Where to Find Them
* **Natural-language motion control**: Parse LLM commands and execute crane physics in real-time with matplotlib visualization
  Files: `llm_parser.py`, `physics_executor.py`, `demo_llm_control.py`, `llm_command.py`

* **LLM-based crane configuration**: Build crane geometry from text specifications and display animation
  File: `crane_configuration.py`

* **Interactive server**: Send commands to running server; real-time matplotlib animation displays motion; `/debug` endpoint provides execution status
  Server: `crane_server.py`
  Test script: `interactive_test.py`

## Real-time Visualization
All flows use matplotlib with FuncAnimation to display crane motion in real-time. Watch the animated matplotlib window as commands execute.

## Configuration
Set environment variables for LLM backend:
- **Azure**: `AZURE_KEY`, `AZURE_ENDPOINT`
- **Qwen**: `QWEN_API_URL`, `QWEN_API_KEY`

## Quick Commands
* **Offline demo**: `python llm_interaction/demo_llm_control.py`
* **Build crane from description**: `python llm_interaction/crane_configuration.py`
* **Start server**: `python llm_interaction/crane_server.py`
* **Test server**: `python llm_interaction/interactive_test.py`