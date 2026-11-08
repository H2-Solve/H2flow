"""Arm A: the raw model. One chat call per problem, zero-shot chain of thought, answer in \\boxed{}."""

from __future__ import annotations

from ..datasets import Problem
from ..ollama_client import OllamaError
from . import Arm, Attempt, RunContext


def ask_model(problem: Problem, seed: int, ctx: RunContext) -> Attempt:
    prompt = ctx.prompt_template.replace("{problem}", problem.problem)  # not .format: prompts contain \boxed{}
    try:
        res = ctx.client.chat(ctx.model, [{"role": "user", "content": prompt}], {**ctx.options, "seed": seed})
    except OllamaError as exc:
        return Attempt(response="", llm_calls=1, prompt_tokens=None, completion_tokens=None, error=str(exc))
    return Attempt(
        response=res.content,
        llm_calls=1,
        prompt_tokens=res.prompt_tokens,
        completion_tokens=res.completion_tokens,
        latency_s=res.latency_s,
        truncated=res.truncated,
    )


class BaseArm(Arm):
    name = "base"
    description = "raw model, zero-shot step-by-step prompt, final answer in \\boxed{}"

    def setup(self, ctx: RunContext) -> dict:
        return {"prompt": ctx.prompt_name}

    def solve(self, problem: Problem, seed: int, ctx: RunContext) -> Attempt:
        return ask_model(problem, seed, ctx)
