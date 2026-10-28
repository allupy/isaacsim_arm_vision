Fixed-zone tabletop agent

planner.py: Chinese deterministic planner; LLMPlanner injectable API boundary, unconfigured by default.
orchestrator.py: sequential execution; only a verified successful skill permits the next step.
skills.py: existing ROS grasp runner adapter. No simulator truth is exposed to the planner.
run.py: entry point, exclusive agent lock, plan/events/results persistence.

Local planning: python agent_workflow/run.py --dry-run
Tests: python agent_workflow/test_workflow.py
Workstation (simulation/control containers running with everyday scene and short gripper):
python3 src/agent_workflow/run.py --run-id agent_sort_20261028 --reset-scene

Default instruction: 把苹果放到水果区，再把杯子放到杯子区
Fixed zones use world coordinates: fruit (.35,-.30), cup (.49,-.30).
Surface heights belong to the current scaled object setup. No generic spatial planning is claimed.
Duplicate zone reservations and unsupported instructions are rejected before motion.
No automatic retries or resume after an unknown execution state. Reruns require a fresh run ID.
Do not run legacy motion scripts concurrently; only this agent entry point honors its lock.
LLM provider must return target/destination/skill dictionaries; the same validator is always applied.

Live API and recording:
llm_api.py implements the HTTPS /chat/completions transport. No API key is saved.
Select --planner llm --model qwen3.8-chat and provide ROBOT_LLM_API_KEY in the process environment,
or use --api-key-stdin and send the key over stdin. Never put a key into source or committed config.
The --api-base option selects an OpenAI-compatible HTTPS endpoint.
--record captures one continuous simulator stream across all steps, including model loading between skills.
Example (runtime key supplied separately):
python3 src/agent_workflow/run.py --planner llm --api-key-stdin --record --reset-scene --run-id UNIQUE_RUN_ID
The persisted plan records the actual model response, usage and validated steps, without credentials.
audit.py independently scores the requested objects after execution.
encode_video.py produces a Chinese-captioned 1280x720 video locally on Windows.

Physical destination prototype:
runtime_destinations.json adds a real colliding dish and a raised four-leg cup support.
Plans can select plate or cup_stand. Their world positions are not stored in ZONES.
The skill uses observer RGB ArUco IDs 10/20 plus synchronized depth to estimate support center and height.
These are deliberately marked fixtures, not validated zero-shot plate/stand recognition.
Two distinct fresh observations must agree before execution; missing depth/marker aborts the skill.
Current object-specific grasp-height offsets remain calibrated to the scaled apple and cup.
Independent scoring checks support height and a conservative interior placement margin.
First physical-destination test: apple error 9.2 mm passed; cup error 27.4 mm on the raised support failed the 21 mm interior margin.
The controller reported completion, but the full-task independent audit failed. Original evidence is preserved.
Randomized fixture positions and general occupancy checking have not yet been validated.
