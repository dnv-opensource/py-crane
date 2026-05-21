USAGE GUIDE - START HERE
========================

THE PROBLEM WAS
---------------

Commands were never executing because:
- Command start_time was always 0.0
- But simulation time (sim_time) was 100+ seconds when second command arrived
- Execution logic checks if abs(sim_time - start_time) < 1e-6
- This never matched, so commands never ran
- Video showed no motion (commands not executing)


THE FIX IS
----------

When command is received, update start_time to current sim_time:

  for cmd in cmds:
      cmd.start_time = sim_time

Now commands execute immediately when received.


QUICK START
-----------

Terminal 1 - Start server:
$ cd /nvme/admhonqia/New/py-crane/llm_interaction
$ python crane_server.py

You should see messages like:
[STATUS] SIM_TIME=0.00s | Active_Commands=0 | Timeline_Frames=0
[STATUS] SIM_TIME=1.00s | Active_Commands=0 | Timeline_Frames=50
[STATUS] SIM_TIME=2.00s | Active_Commands=0 | Timeline_Frames=100
...

Terminal 2 - Run test:
$ python test_crane_server_v2.py

Test will:
1. Send 4 commands
2. Monitor execution via /debug API (no video needed)
3. Show motion confirmation for each command
4. Attempt video download
5. Report success/failure


MANUAL VERIFICATION
--------------------

Terminal 3 - Check command execution:

Step 1: Get current status
$ curl http://localhost:9100/state
Result: {"length": 30.0, "polar": 1.047, "azimuth": 0.0}

Step 2: Send a command
$ curl -X POST http://localhost:9100/command \
  -H "Content-Type: application/json" \
  -d '{"text": "Rotate the boom 45 degrees clockwise in 5 seconds"}'
Result: {"status": "ok", "count": 1}

In Terminal 1, you should see:
[COMMAND RECEIVED] Rotate the boom 45 degrees clockwise in 5 seconds
[COMMAND PARSED] Type=boom_rotate, start_time=5.23, duration=5.0
[COMMAND] boom_rotate - EXECUTING (progress=10.0%)
[COMMAND] boom_rotate - EXECUTING (progress=50.0%)
[COMMAND] boom_rotate - COMPLETED at t=10.23s

Step 3: Verify motion happened
$ curl http://localhost:9100/debug | python -m json.tool

Look for motion_detected section:
"motion_detected": {
  "boom_length_delta": 0.0,
  "polar_delta": 0.0,
  "azimuth_delta": 0.785,  <-- This shows 45 degree rotation!
  "wire_delta": 0.0
}

Non-zero azimuth_delta proves rotation happened!

Step 4: Get video
$ curl http://localhost:9100/video -o output.mp4

Video should show the rotation (if generation worked).


UNDERSTANDING THE LOG OUTPUT
-----------------------------

Every ~1 second you see:

[STATUS] SIM_TIME=X.XXs | Active_Commands=N | Timeline_Frames=M
  - X.XXs: Current simulation time
  - N: How many commands are executing right now
  - M: How many frames recorded (M * 0.02s = total time)

[POSITION] Boom_Length=X.XXm | Polar_Angle=X.XXrad | Azimuth=X.XXrad | Wire_Length=X.XXm
  - Current position of all boom parts
  - Changes as commands execute

[MOTION] Boom_Delta=+X.XXm | Polar_Delta=+X.XXrad | Azimuth_Delta=+X.XXrad | Wire_Delta=+X.XXm
  - Change since last output (~1 second ago)
  - + means increase, - means decrease
  - Only appears if motion > threshold
  - THIS PROVES COMMANDS ARE WORKING

[COMMAND] boom_rotate - EXECUTING (progress=50.0%)
  - Command type
  - Current status (WAITING / EXECUTING / COMPLETED)
  - Percentage complete (for executing commands)


WHAT TO LOOK FOR
----------------

Success indicators:

1. [COMMAND RECEIVED] message in logs ✓
2. [COMMAND PARSED] with correct type ✓
3. [COMMAND] ... - EXECUTING messages ✓
4. [MOTION] messages with non-zero deltas ✓
5. [COMMAND] ... - COMPLETED message ✓
6. /debug API shows motion_detected non-zero ✓
7. Next command executes without delay ✓

If you see all of these, commands are executing correctly!


TESTING SEQUENCE
----------------

1. Start server (Terminal 1)
   python crane_server.py

2. Wait for it to stabilize (2-3 seconds of [STATUS] messages)

3. Run test (Terminal 2)
   python test_crane_server_v2.py

4. Watch as test sends commands and verifies:
   - [TEST 1/4] Sending command: Rotate...
   - [SUCCESS] Command accepted
   - [INFO] Waiting for command execution...
   - [VERIFIED] Command 1 execution confirmed
   - [INFO] Waiting for next command...

5. Check Terminal 1 logs for actual execution:
   [COMMAND] boom_rotate - EXECUTING
   [COMMAND] boom_rotate - COMPLETED

6. If everything passed, commands are working correctly!


INDEPENDENT VERIFICATION (No Video Needed)
-------------------------------------------

You can verify command execution without any video:

$ while true; do
    echo "=== $(date '+%H:%M:%S') ==="
    curl -s http://localhost:9100/debug | grep -E '"(simulation_time|active_commands|motion_detected)"' | head -10
    sleep 1
  done

Watch motion_detected values change as commands execute.
Non-zero deltas = commands working.


TROUBLESHOOTING
---------------

If no [MOTION] messages appear:
  1. Check [COMMAND] message says EXECUTING
  2. Run /debug API to check motion_detected
  3. Check that active_commands > 0
  4. Verify command was parsed correctly

If [COMMAND] shows WAITING:
  1. Check start_time matches current sim_time
  2. Command might be in future (wait for time to catch up)
  3. Check command parsing didn't fail

If video doesn't generate:
  1. This is OK - command execution is more important
  2. Video generation is a visualization feature
  3. /debug API proves commands executed
  4. Check [VIDEO] messages in server logs for errors

If server crashes:
  1. Check terminal output for error messages
  2. Verify all Python packages installed
  3. Ensure port 9100 is available
  4. Check disk space in /nvme directory


SUCCESS CRITERIA
----------------

Command execution is successful when:

1. Immediate execution
   - Command starts executing right after receiving it
   - No waiting for t=0

2. Motion in data
   - /debug API shows non-zero motion_detected
   - [MOTION] messages in logs

3. Sequential execution
   - Second command waits for first to complete
   - Then executes immediately

4. Clear status
   - [COMMAND] messages show EXECUTING then COMPLETED
   - No error messages

If all above are true, the fix is working correctly.
Video generation is a bonus, not required for verification.


NEXT STEPS
----------

1. Start crane_server.py
2. Run test_crane_server_v2.py
3. Verify [MOTION] and [COMMAND] messages
4. Use /debug API to confirm motion_detected changes
5. Optional: Check generated video if available

All documentation in this directory:
- PROBLEM_ANALYSIS.md - Why it failed
- EXECUTION_SUMMARY.md - What was fixed
- QUICK_COMMANDS.md - Command reference
- FIXES_APPLIED.md - Technical details
