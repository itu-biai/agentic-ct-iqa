import json
from types import SimpleNamespace

import numpy as np
import pytest

from src.agent import AgentConfig, Case, run_claude_agent, read_skill_tool, system_prompt
from src.skill_tools import LDCTIQA, build_skills
from src.tools import to_png


def block(name, tool_input, block_id):
    return SimpleNamespace(type="tool_use", name=name, input=tool_input, id=block_id)


def reply(*content, stop_reason="tool_use", model="claude-opus-5-5"):
    usage = SimpleNamespace(input_tokens=1000, output_tokens=200, cache_read_input_tokens=500,
                            cache_creation_input_tokens=0, iterations=None)
    return SimpleNamespace(content=list(content), stop_reason=stop_reason, model=model, usage=usage, stop_details=None)


class FakeClient:
    """Returns scripted responses and records every request."""
    def __init__(self, responses):
        self.responses, self.requests = list(responses), []
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        self.requests.append({**kwargs, "messages": list(kwargs["messages"])})  # snapshot: the loop keeps appending
        return self.responses.pop(0)


@pytest.fixture
def setup(tmp_path):
    skills = build_skills(LDCTIQA, ["rubric", "ct_view", "noise_measure"])
    tools = {t.name: t for s in skills for t in s.tools}
    tools["read_skill"] = read_skill_tool(skills)
    image = np.clip(np.random.default_rng(0).normal(0.65, 0.04, (512, 512)), 0, 1).astype(np.float32)
    case = Case("0001", image)
    config = AgentConfig(trajectory_path=tmp_path / "traj.jsonl", max_tool_calls=3, max_turns=4)
    system = system_prompt(skills[0].instructions, skills[1:])
    return config, system, tools, case, to_png(image, 0, 1)


def test_normal_episode_logs_score_tools_and_cost(setup):
    config, system, tools, case, png = setup
    client = FakeClient([
        reply(block("read_skill", {"skill": "noise_measure"}, "a"), block("high_pass_energy", {}, "b")),
        reply(block("submit_score", {"score": 4.7, "confidence": 0.8, "evidence": ["hp 0.04"], "trusted_experts": []}, "c")),
    ])
    result = run_claude_agent(client, config, system, tools, case, png)
    assert result.status == "ok" and result.score == 4.0  # clipped to the scale
    assert [c.name for c in result.tool_calls] == ["read_skill", "high_pass_energy"]
    assert "high_pass_std" in result.tool_calls[1].values
    assert result.cost_usd == pytest.approx(2 * (1000 * 4 + 200 * 20 + 500 * 0.2) / 1e6)
    logged = json.loads(config.trajectory_path.read_text().splitlines()[-1])
    assert logged["case_id"] == "0001" and logged["served_by"] == ["claude-opus-5-5", "claude-opus-5-5"]


def test_request_uses_fallback_effort_thinking_and_cache(setup):
    config, system, tools, case, png = setup
    client = FakeClient([reply(block("submit_score", {"score": 2, "confidence": 0.5, "evidence": [], "trusted_experts": []}, "a"))])
    run_claude_agent(client, config, system, tools, case, png)
    request = client.requests[0]
    assert request["model"] == "claude-opus-5-5" and request["fallbacks"] == "default"
    assert request["betas"] == ["server-side-fallback-2026-07-01"]
    assert request["output_config"] == {"effort": "high"} and request["thinking"] == {"type": "adaptive"}
    assert request["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert request["tools"][-1]["name"] == "submit_score" and request["tools"][-1]["strict"] is True
    assert "tool_choice" not in request  # forced tool choice is a 400 on Opus 5.5


def test_budget_is_enforced(setup):
    config, system, tools, case, png = setup
    many = [block("high_pass_energy", {}, f"t{i}") for i in range(5)]
    client = FakeClient([reply(*many),
                         reply(block("submit_score", {"score": 1, "confidence": 0.5, "evidence": [], "trusted_experts": []}, "s"))])
    result = run_claude_agent(client, config, system, tools, case, png)
    assert sum(not c.over_budget for c in result.tool_calls) == config.max_tool_calls
    assert sum(c.over_budget for c in result.tool_calls) == 2
    second_turn_results = client.requests[1]["messages"][-1]["content"]
    assert sum(r.get("is_error", False) for r in second_turn_results) == 2


def test_refusal_and_missing_score_are_logged_not_raised(setup):
    config, system, tools, case, png = setup
    refused = run_claude_agent(FakeClient([reply(stop_reason="refusal")]), config, system, tools, case, png)
    assert refused.status == "refusal" and refused.score is None
    silent = FakeClient([reply(stop_reason="end_turn") for _ in range(config.max_turns)])
    result = run_claude_agent(silent, config, system, tools, case, png)
    assert result.status == "no_score" and len(silent.requests) == config.max_turns


def test_tool_error_does_not_end_episode(setup):
    config, system, tools, case, png = setup
    client = FakeClient([reply(block("zoom_view", {"row": "bad", "col": 0}, "z")),
                         reply(block("submit_score", {"score": 3, "confidence": 0.5, "evidence": [], "trusted_experts": []}, "s"))])
    result = run_claude_agent(client, config, system, tools, case, png)
    assert result.status == "ok"
    assert client.requests[1]["messages"][-1]["content"][0]["is_error"] is True
