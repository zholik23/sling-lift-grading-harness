"""The one LLM entry point harnesses use for judge calls.

One line of indirection on purpose: `llm_judge` names this module and not
a provider, so the route can change without touching a judge.

THE IMPORT IS THE POINT OF FAILURE, and deliberately so. There is no stub
and no fallback: a machine without the `anthropic` SDK (or without an
Azure credential) cannot import this, and a harness that names it becomes
unimportable there. That is why no harness
imports it directly -- `common/judge_runner` does the import inside the
call and records its absence as a fact, so `--score-from` keeps working on
a host with no judge instead of grading against placeholder scores.
"""
from common.call_claude import call_claude as judge


def call_llm(prompt):
    return judge(prompt)
