"""Step 1.5: provider-neutral skill types and the Claude backend (manual tool loop).

The loop is manual (not the SDK tool runner) because every tool result can carry an image and the
tool-call budget B must be enforced. History is append-only (full response.content is appended),
which keeps thinking blocks valid and the prompt cache warm.
"""
import base64
import json
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from src.tools import ToolResult


@dataclass(frozen=True)
class Case:
    case_id: str
    image: np.ndarray
    expert_scores: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_schema: dict[str, Any]
    run: Callable[..., ToolResult]  # run(case, **tool_input)


@dataclass(frozen=True)
class Skill:
    name: str
    description: str
    instructions: str
    tools: tuple[ToolSpec, ...]


def load_skill(folder: Path, tools: tuple[ToolSpec, ...]) -> Skill:
    """SKILL.md: first heading is the name, first paragraph the one-line description, the rest the procedure."""
    lines = folder.joinpath("SKILL.md").read_text(encoding="utf-8").strip().splitlines()
    name = lines[0].lstrip("# ").strip()
    description, _, instructions = "\n".join(lines[1:]).strip().partition("\n\n")
    return Skill(name, description.strip(), instructions.strip(), tools)


def read_skill_tool(skills: list[Skill]) -> ToolSpec:
    by_name = {s.name: s for s in skills}

    def run(case: Case, skill: str) -> ToolResult:
        if skill not in by_name:
            return ToolResult({}, text=f"Unknown skill '{skill}'. Available: {', '.join(by_name)}")
        return ToolResult({}, text=by_name[skill].instructions)

    return ToolSpec("read_skill", "Load the full procedure of a skill before using its tools.",
                    {"type": "object", "properties": {"skill": {"type": "string", "enum": sorted(by_name)}},
                     "required": ["skill"], "additionalProperties": False}, run)


def system_prompt(rubric: str, skills: list[Skill]) -> str:
    listing = "\n".join(f"- {s.name}: {s.description}" for s in skills)
    return (f"{rubric}\n\nSkills available (call read_skill before using a skill's tools):\n{listing}\n\n"
            "Finish by calling submit_score exactly once.")


SUBMIT_SCORE: dict[str, Any] = {
    "name": "submit_score",
    "description": "Submit the final quality score on the 0 to 4 MOS scale with the evidence used. Call once, last.",
    "strict": True,
    "input_schema": {
        "type": "object",
        "properties": {
            "score": {"type": "number", "description": "0 (nondiagnostic) to 4 (excellent); decimals allowed"},
            "confidence": {"type": "number", "description": "0 to 1"},
            "evidence": {"type": "array", "items": {"type": "string"},
                         "description": "Short statements citing measured values or views"},
            "trusted_experts": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["score", "confidence", "evidence", "trusted_experts"],
        "additionalProperties": False,
    },
}

# USD per million tokens: (input, output, cache read, 5-minute cache write). Source: claude-api skill, 2026-09-25.
PRICES: dict[str, tuple[float, float, float, float]] = {
    "claude-opus-5-5": (4.00, 20.00, 0.20, 5.00),
    "claude-opus-5": (5.00, 25.00, 0.50, 6.25),
    "claude-opus-4-8": (5.00, 25.00, 0.50, 6.25),
    "claude-sonnet-5-5": (2.00, 10.00, 0.20, 2.50),
}


@dataclass(frozen=True)
class AgentConfig:
    model: str = "claude-opus-5-5"
    effort: str = "high"  # set explicitly: Opus 5.5 defaults to medium; sweep low/medium/high on dev (Step 2.3)
    max_tool_calls: int = 6
    max_turns: int = 10
    max_tokens: int = 16000
    task_prompt: str = ("Assess the diagnostic image quality of this low-dose CT slice on the radiologist scale "
                        "0 (nondiagnostic) to 4 (excellent). Use the skills, then call submit_score.")
    trajectory_path: Path = Path("outputs/agentic_ct_iqa/trajectories.jsonl")


@dataclass
class ToolCall:
    name: str
    input: dict[str, Any]
    values: dict[str, float]
    over_budget: bool = False


@dataclass
class AgentResult:
    case_id: str
    config_name: str
    repeat: int
    score: float | None
    confidence: float | None
    evidence: list[str]
    trusted_experts: list[str]
    tool_calls: list[ToolCall]
    usage: dict[str, int]
    cost_usd: float
    served_by: list[str]
    fallback_used: bool
    status: str  # ok | refusal | no_score | error
    detail: str = ""


def image_block(png: bytes) -> dict[str, Any]:
    return {"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                        "data": base64.standard_b64encode(png).decode("utf-8")}}


def tool_result_block(tool_use_id: str, result: ToolResult) -> dict[str, Any]:
    text = json.dumps(result.values, sort_keys=True) + (f"\n{result.text}" if result.text else "")
    content: list[dict[str, Any]] = [{"type": "text", "text": text}]
    if result.image_png is not None:
        content.append(image_block(result.image_png))
    return {"type": "tool_result", "tool_use_id": tool_use_id, "content": content}


def add_usage(total: dict[str, int], usage: Any) -> None:
    for key in ("input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens"):
        total[key] = total.get(key, 0) + int(getattr(usage, key, 0) or 0)


def cost_usd(model: str, usage: Any) -> float:
    price_in, price_out, price_read, price_write = PRICES.get(model, PRICES["claude-opus-5-5"])
    return ((getattr(usage, "input_tokens", 0) or 0) * price_in + (getattr(usage, "output_tokens", 0) or 0) * price_out
            + (getattr(usage, "cache_read_input_tokens", 0) or 0) * price_read
            + (getattr(usage, "cache_creation_input_tokens", 0) or 0) * price_write) / 1e6


def run_claude_agent(client: Any, config: AgentConfig, system: str, tools: dict[str, ToolSpec], case: Case,
                     first_view_png: bytes, config_name: str = "default", repeat: int = 0) -> AgentResult:
    """One agent episode on one image. Never raises for model-side outcomes (refusal, no score);
    those are logged with a status so the run can be re-tried or reported separately."""
    tool_params = [{"name": t.name, "description": t.description, "input_schema": t.input_schema}
                   for t in tools.values()] + [SUBMIT_SCORE]
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": [image_block(first_view_png), {"type": "text", "text": config.task_prompt}]}]
    calls: list[ToolCall] = []
    usage_total: dict[str, int] = {}
    cost, served_by, fallback_used = 0.0, [], False

    def finish(status: str, detail: str = "", answer: dict[str, Any] | None = None) -> AgentResult:
        answer = answer or {}
        score = float(np.clip(answer["score"], 0.0, 4.0)) if "score" in answer else None
        result = AgentResult(case.case_id, config_name, repeat, score, answer.get("confidence"),
                             answer.get("evidence", []), answer.get("trusted_experts", []), calls, usage_total,
                             round(cost, 6), served_by, fallback_used, status, detail)
        config.trajectory_path.parent.mkdir(parents=True, exist_ok=True)
        with config.trajectory_path.open("a", encoding="utf-8") as log:
            log.write(json.dumps(asdict(result)) + "\n")
        return result

    for _ in range(config.max_turns):
        response = client.beta.messages.create(
            model=config.model,
            max_tokens=config.max_tokens,
            system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            cache_control={"type": "ephemeral"},  # also cache the growing conversation inside the episode
            tools=tool_params,
            messages=messages,
            thinking={"type": "adaptive"},
            output_config={"effort": config.effort},
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )
        add_usage(usage_total, response.usage)
        cost += cost_usd(response.model, response.usage)
        served_by.append(response.model)
        fallback_used |= any(getattr(i, "type", "") == "fallback_message" for i in (getattr(response.usage, "iterations", None) or []))
        if response.stop_reason == "refusal":
            details = getattr(response, "stop_details", None)
            return finish("refusal", f"{getattr(details, 'category', None)}: {getattr(details, 'explanation', '')}")
        messages.append({"role": "assistant", "content": response.content})

        results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            if block.name == "submit_score":
                return finish("ok", answer=dict(block.input))
            if len(calls) >= config.max_tool_calls:
                calls.append(ToolCall(block.name, dict(block.input), {}, over_budget=True))
                results.append({"type": "tool_result", "tool_use_id": block.id, "is_error": True,
                                "content": "Tool budget reached. Call submit_score now."})
                continue
            spec = tools.get(block.name)
            try:
                if spec is None:
                    raise KeyError(f"unknown tool {block.name}")
                outcome = spec.run(case, **block.input)
                calls.append(ToolCall(block.name, dict(block.input), outcome.values))
                results.append(tool_result_block(block.id, outcome))
            except Exception as error:  # a failing tool must not end the episode
                calls.append(ToolCall(block.name, dict(block.input), {}))
                results.append({"type": "tool_result", "tool_use_id": block.id, "is_error": True,
                                "content": f"Tool error: {error}"})
        messages.append({"role": "user", "content": results or [{"type": "text", "text": "Call submit_score to finish."}]})
    return finish("no_score", f"no submit_score after {config.max_turns} turns")
