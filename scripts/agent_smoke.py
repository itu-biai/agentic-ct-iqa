"""Step 1.5: agent loop end to end on 10 dev images with rubric, ct_view and noise_measure only.
Fills the Step 1.5 check table: finished runs, refusals/failures, fallback runs, tool calls, cost, latency.

Usage: .venv/Scripts/python -m scripts.agent_smoke [--n 10] [--effort high]
Needs Claude API credentials (ANTHROPIC_API_KEY or an `ant auth login` profile). Every run costs money.
"""
import argparse
import statistics
import time

import anthropic

from src.agent import AgentConfig, Case, read_skill_tool, run_claude_agent, system_prompt
from src.config import CONFIG
from src.data import load_ldctiqa_split, read_id_list
from src.skill_tools import LDCTIQA, build_skills
from src.tools import to_png

# Runs needed by the plan's Compute Budget (Step 1.5 excluded), to extrapolate the total API cost.
PLANNED_RUNS = 11_900


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--effort", default="high", choices=["low", "medium", "high", "xhigh", "max"])
    args = parser.parse_args()

    skills = build_skills(LDCTIQA, ["rubric", "ct_view", "noise_measure"])
    rubric, tool_skills = skills[0], skills[1:]
    tools = {t.name: t for s in tool_skills for t in s.tools}
    tools["read_skill"] = read_skill_tool(tool_skills)
    system = system_prompt(rubric.instructions, tool_skills)
    config = AgentConfig(effort=args.effort,
                         trajectory_path=CONFIG.output_dir / "trajectories_step15.jsonl")

    dev = read_id_list(CONFIG.splits_dir / "dev.txt")[: args.n]
    samples = [s for s in load_ldctiqa_split(CONFIG.train_images, CONFIG.train_labels) if s.image_id in set(dev)]
    client = anthropic.Anthropic()
    try:  # free credential and model check before any paid call
        client.models.retrieve(config.model)
    except (TypeError, anthropic.AuthenticationError, anthropic.PermissionDeniedError) as error:
        raise SystemExit(f"No usable Claude API credentials ({error}). Set ANTHROPIC_API_KEY or run `ant auth login`.")

    level, width = LDCTIQA.windows["full"]
    results, latencies = [], []
    for sample in samples:
        start = time.perf_counter()
        first_view = to_png(sample.image, level - width / 2, level + width / 2)
        result = run_claude_agent(client, config, system, tools, Case(sample.image_id, sample.image), first_view,
                                  config_name=f"step15_{args.effort}")
        latencies.append(time.perf_counter() - start)
        results.append((sample, result))
        print(f"{sample.image_id}  MOS {sample.mos:.2f}  score {result.score}  status {result.status}  "
              f"tools {len(result.tool_calls)}  ${result.cost_usd:.4f}  {latencies[-1]:.1f}s")

    finished = [r for _, r in results if r.status == "ok"]
    costs = [r.cost_usd for _, r in results]
    cache_reads = sum(r.usage.get("cache_read_input_tokens", 0) for _, r in results)
    print("\nStep 1.5 check                         observed")
    print(f"Runs finished with a score             {len(finished)} / {len(results)}")
    print(f"Refused or failed runs                 {len(results) - len(finished)}")
    print(f"Runs served by a fallback model        {sum(r.fallback_used for _, r in results)}")
    print(f"Tool calls per image (median)          {statistics.median(len(r.tool_calls) for _, r in results)}")
    print(f"Cost per image (USD, mean)             {statistics.mean(costs):.4f}")
    print(f"Latency per image (median, s)          {statistics.median(latencies):.1f}")
    print(f"Cache read tokens (all runs)           {cache_reads}")
    print(f"Extrapolated total API cost (USD)      {statistics.mean(costs) * PLANNED_RUNS:,.0f}  ({PLANNED_RUNS} runs)")


if __name__ == "__main__":
    main()
