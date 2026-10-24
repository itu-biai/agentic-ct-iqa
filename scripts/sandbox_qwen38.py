"""Qwen3.8-27B (Qwen/Qwen3.8-27B) on the 20 sandbox images, served by vLLM on a Colab A100/H100, in both protocols.

The same two protocols as Qwen3.5-4B (scripts/sandbox_qwen35.py), with the same prompts, permissions and runners:
setups 16 and 17 are one independent request per image (colab/vm/qwen38_single.py), setups 18 and 19 an OpenCode
agent in copies of the setup 1/2 sandboxes (colab/vm/q27_agent.py). The GPU, weights (bf16 or Qwen's FP8 checkpoint)
and context come from the VM's serve.json and are recorded in each run_info.json.

    python -m scripts.sandbox_qwen38 ingest <raw.jsonl>    # setups 16/17 -> results/setup1x_qwen38_27b_*/run_<n>/
    python -m scripts.sandbox_qwen38 agent <agent.zip>     # setups 18/19 -> results/setup1x_qwen38_27b_agent_*/
    python -m scripts.sandbox_qwen38 agent <agent_examples.zip> <serve.json>  # setup 26, on a later VM
"""
import json
import sys
from pathlib import Path

from scripts import sandbox_qwen35

MODEL = "Qwen/Qwen3.8-27B"
AGENT_SETUPS = ("setup18_qwen38_27b_agent_vision_only", "setup19_qwen38_27b_agent_vision_tools",
                "setup26_qwen38_27b_agent_vision_tools_examples")


def serve(raw: Path) -> dict:
    """The server settings the VM recorded (q27_serve.py), from the meta line of the single-request log."""
    for line in raw.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if record.get("type") == "meta":
            return record["serve"]
    raise SystemExit(f"no meta line in {raw}")


if __name__ == "__main__":
    command, path = sys.argv[1], Path(sys.argv[2])
    if command == "ingest":
        sandbox_qwen35.ingest(path, model=MODEL, provider=serve(path)["server"])
    elif command == "agent":
        server = (json.loads(Path(sys.argv[3]).read_text(encoding="utf-8")) if len(sys.argv) > 3
                  else serve(path.with_name("qwen38_raw.jsonl")))["server"]
        sandbox_qwen35.agent(path, setups=AGENT_SETUPS, model=MODEL, server=server)
