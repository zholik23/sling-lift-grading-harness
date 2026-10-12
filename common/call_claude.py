"""Claude route: Claude Sonnet 5 or Claude Fable 5.1, on Azure.

AZURE ONLY. There is no Bedrock path in this module any more, and no AWS
credential is read anywhere in it. Both models are deployments on the same
`jenni-m7rybsi6-eastus2` resource that serves the GPT route, reached with
`AnthropicFoundry` -- Azure is "Microsoft Foundry" to the Anthropic SDK, and
it takes the resource NAME, not the `*.cognitiveservices.azure.com` URL (a
base_url of that form 404s).

THIS NO LONGER USES THE CLAUDE AGENT SDK. That SDK is the Claude Code
harness: it shells out to a bundled `claude` CLI, and it only knows how to
reach a model through the Anthropic API, Bedrock or Vertex -- there is no
Foundry backend to point it at. So the agent loop here is the same manual
one the GPT and Gemini routes run, over the same four tools from
`agent_workspace`. Three consequences worth knowing:

  * All three routes now get an identical tool surface and an identical
    prompt, which is what makes their scores comparable. Claude used to get
    the Agent SDK's Write/Read/Glob/Grep/Bash instead.
  * The `claude` CLI is no longer a dependency of this repo.
  * Model ids are first-party ids (`claude-sonnet-5`), not Bedrock's
    `us.anthropic.*` inference-profile ids.

THINKING IS CONFIGURED PER MODEL AND THE TWO MODELS DISAGREE. Fable 5.1
thinks unconditionally and rejects any explicit thinking config, so the
parameter is omitted for it entirely; Sonnet 5 needs `{"type": "adaptive"}`
to think at all. `budget_tokens` is dead on both -- verified, not assumed:
it comes back 400 `"thinking.type.enabled" is not supported`. Depth is set
with `output_config.effort` on both.
"""
from __future__ import annotations

import base64
import functools
import mimetypes
import os
import shutil
import tempfile
import time
from pathlib import Path

import dotenv
from anthropic import AnthropicFoundry

from common.agent_workspace import (
    AGENT_SOLUTION_FILENAME,
    BINARY_DATA_EXTS,
    BINARY_DATA_HINT,
    DEFAULT_AGENT_MAX_TURNS,
    TOOL_SPECS,
    agent_task_suffix,
    attach_files,
    attached_files_note,
    clip,
    dispatch,
    read_back_solution,
)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
dotenv.load_dotenv(REPO_ROOT / ".env")

#: Deployment names on the Azure resource, which happen to equal the
#: first-party model ids. Overridable because a second resource would very
#: likely label them differently.
MODELS = {
    "sonnet5": os.getenv("CLAUDE_MODEL_SONNET5", "claude-sonnet-5"),
    "fable51": os.getenv("CLAUDE_MODEL_FABLE51", "claude-fable-5-1"),
}

DEFAULT_MODEL = os.getenv("CLAUDE_DEFAULT_MODEL", "sonnet5")

#: `xhigh` is the sweet spot for agentic/coding work on both of these; the
#: judge path drops to `high`, since it only has to justify a score.
EFFORT = {"sonnet5": "xhigh", "fable51": "xhigh"}

DISPLAY_NAMES = {"sonnet5": "Claude Sonnet 5", "fable51": "Claude Fable 5.1"}

AZURE_RESOURCE = (os.getenv("AZURE_CLAUDE_RESOURCE")
                  or "jenni-m7rybsi6-eastus2")
AZURE_API_KEY = os.getenv("AZURE_API_KEY")

#: Non-streaming ceiling. The SDK wants streaming for anything much larger,
#: and neither a judge call nor a solver turn needs it.
DEFAULT_MAX_TOKENS = int(os.getenv("CLAUDE_MAX_TOKENS", "16000"))

CALL_RETRIES = int(os.getenv("CLAUDE_CALL_RETRIES", "5"))
CALL_RETRY_BACKOFF_S = int(os.getenv("CLAUDE_CALL_BACKOFF_S", "10"))

#: SOME FAILURES ARE NOT WORTH A SECOND TRY, and retrying them is not free.
#: Measured twice: task 59 passed an alias the endpoint does not have and
#: paid 340 s per model in retries of a 404; task 75 ran on a machine whose
#: proxy is a socks4 URL httpx will never accept and paid 315 s per model
#: of the identical error. Five inner attempts with a growing backoff,
#: inside three outer ones, is fifteen repetitions of a sentence that was
#: true the first time.
#:
#: A transport fault -- a timeout, a reset, a rate limit, a 5xx -- is worth
#: repeating. A configuration fault is not: nothing between two attempts
#: changes a deployment name, an API key or a proxy scheme. These are
#: matched against the exception's text, lower-cased.
FATAL_SIGNATURES = (
    "unknown scheme for proxy",
    "deploymentnotfound",
    "invalid api key",
    "invalid_api_key",
    "authentication_error",
    "unsupported_country_region_territory",
    "no such deployment",
)


def is_configuration_error(exc) -> bool:
    """True when asking again cannot possibly help."""
    text = f"{type(exc).__name__}: {exc}".lower()
    return any(sign in text for sign in FATAL_SIGNATURES)

AGENT_READ_TOOL = "read_file"
AGENT_BASH_TOOL = "bash"

#: Claude now drives the same tools as the other two routes, so it gets the
#: same wording naming them.
AGENT_TASK_SUFFIX = agent_task_suffix(bash_tool=AGENT_BASH_TOOL)

#: BINARY_DATA_EXTS / BINARY_DATA_HINT / attach_files / attached_files_note
#: are re-exported above: `judge_runner` imports them from this module.


def resolve_model(model: str) -> tuple[str, str | None]:
    """'sonnet5'/'fable51' -> (deployment id, effort).

    Returns a pair rather than a string because callers unpack it; a full
    id passes through with the default effort.
    """
    if model in MODELS:
        return MODELS[model], EFFORT.get(model)
    return model, EFFORT.get(DEFAULT_MODEL)


_resolve_model = resolve_model


@functools.lru_cache(maxsize=1)
def client():
    """The Foundry client, built once and reused."""
    if not AZURE_API_KEY:
        raise RuntimeError(
            "no Azure credential for Claude: set AZURE_API_KEY in .env")
    return AnthropicFoundry(api_key=AZURE_API_KEY, resource=AZURE_RESOURCE)


def thinking_config(model: str):
    """Fable 5.1 rejects an explicit thinking config; Sonnet 5 requires one.

    Returns None to mean "omit the parameter", which is not the same as
    disabling thinking -- on Fable 5.1 omitting it is the only way to get
    its always-on thinking without a 400.
    """
    if MODELS.get(model, model) == MODELS["fable51"]:
        return None
    return {"type": "adaptive"}


#: `cache_control` marks a prefix boundary: everything BEFORE it is cached
#: and re-read at a fraction of the input price on the next turn. The API
#: allows at most four, and this module uses three -- tools, the opening
#: user message, and one rolling marker on the newest tool results.
CACHE_MARK = {"type": "ephemeral"}


def _tools(cache: bool = True):
    """The shared specs as Anthropic tool definitions.

    Tools render before system and messages, never change during a run, and
    are several KB of JSON -- so the last one carries a cache breakpoint and
    the whole block is read from cache on every turn after the first.
    """
    defs = [{"name": s["name"], "description": s["description"],
             "input_schema": s["parameters"]} for s in TOOL_SPECS]
    if cache and defs:
        defs[-1] = {**defs[-1], "cache_control": CACHE_MARK}
    return defs


def _mark_rolling_cache(messages) -> None:
    """Move the rolling breakpoint to the newest user message.

    An agent transcript only ever grows at the end, so the whole history up
    to the last tool result is a stable prefix. Marking it means each turn
    re-reads the conversation from cache instead of paying full input price
    for it again -- which is the thing that makes a 40-turn CAD run
    affordable. The previous marker is cleared first: the cap is four, and
    leaving one per turn would blow through it by turn five.
    """
    last_list = None
    for msg in messages:
        content = msg.get("content") if isinstance(msg, dict) else None
        if not isinstance(content, list):
            continue
        for block in content:
            if isinstance(block, dict):
                block.pop("cache_control", None)
        last_list = content
    if last_list and isinstance(last_list[-1], dict):
        last_list[-1]["cache_control"] = CACHE_MARK


def _media_block(path):
    """An image or PDF as a content block; anything else as labelled text."""
    path = Path(path)
    mime = mimetypes.guess_type(path.name)[0] or ""
    if mime.startswith("image/"):
        return {"type": "image", "source": {
            "type": "base64", "media_type": mime,
            "data": base64.b64encode(path.read_bytes()).decode("ascii")}}
    if mime == "application/pdf":
        return {"type": "document", "source": {
            "type": "base64", "media_type": "application/pdf",
            "data": base64.b64encode(path.read_bytes()).decode("ascii")}}
    return {"type": "text",
            "text": f"--- {path.name} ---\n"
                    f"{path.read_text(errors='replace')}"}


def _create(messages, *, model, tools, max_tokens, effort, system=None):
    model_id, default_effort = resolve_model(model)
    kwargs = dict(model=model_id, max_tokens=max_tokens, messages=messages,
                  output_config={"effort": effort or default_effort})
    think = thinking_config(model)
    if think is not None:
        kwargs["thinking"] = think
    if tools:
        kwargs["tools"] = tools
    if system:
        kwargs["system"] = system
    return client().messages.create(**kwargs)


def cache_usage(resp) -> dict:
    """The two numbers that say whether caching is actually working.

    `cache_read_input_tokens` staying at zero across turns means something
    is invalidating the prefix, and that is invisible without looking.
    """
    u = getattr(resp, "usage", None)
    return {
        "input": getattr(u, "input_tokens", None),
        "cache_read": getattr(u, "cache_read_input_tokens", None),
        "cache_write": getattr(u, "cache_creation_input_tokens", None),
        "output": getattr(u, "output_tokens", None),
    }


def _refusal_error(resp):
    """Safety classifiers decline with HTTP 200 and stop_reason 'refusal'.

    Worth naming rather than letting it surface as an empty answer: a
    refused judge call and a model that simply said nothing need different
    responses from whoever reads the log.
    """
    if getattr(resp, "stop_reason", None) != "refusal":
        return None
    det = getattr(resp, "stop_details", None)
    return RuntimeError(
        f"refused by the model (category={getattr(det, 'category', None)}): "
        f"{getattr(det, 'explanation', '') or 'no explanation given'}")


def _text_of(resp) -> str:
    return "".join(b.text for b in resp.content
                   if getattr(b, "type", "") == "text").strip()


def run_agent(prompt: str, *, cwd=None, model: str = DEFAULT_MODEL,
              tools=None, max_turns: int = DEFAULT_AGENT_MAX_TURNS,
              effort: str | None = None, files=(), inline_files=(),
              announce_files: bool = True, require_file: str | None = None,
              max_tokens: int = DEFAULT_MAX_TOKENS, system: str | None = None,
              verbose: bool = True):
    """Tool-using agent loop. Returns (final_text, meta).

    tools=[] means "no tools at all" -- that is how a one-turn call asks for
    a plain answer, and it is why this defaults to None rather than to the
    tool list.

    files: copied into cwd and announced, as in the other two routes.
    inline_files: sent as content blocks instead (images and PDFs, which
    `read_file` cannot open).
    """
    if files and cwd is None:
        raise ValueError("run_agent(files=...) needs a cwd to copy them into")
    if cwd is not None:
        cwd = Path(cwd)
        cwd.mkdir(parents=True, exist_ok=True)

    if files:
        names = attach_files(cwd, files)
        if announce_files:
            prompt = prompt + attached_files_note(
                names, read_tool=AGENT_READ_TOOL, bash_tool=AGENT_BASH_TOOL)

    content = [{"type": "text", "text": prompt}]
    content += [_media_block(f) for f in inline_files]
    # The opening message carries the whole task: instructions, the staged
    # file list, and any drawings. It never changes again, so it is worth a
    # breakpoint of its own.
    content[-1] = {**content[-1], "cache_control": CACHE_MARK}
    messages = [{"role": "user", "content": content}]

    tool_defs = _tools() if tools is None else list(tools)
    cache_stats = []
    transcript = []
    final_text, nudged, turn = "", False, 0

    while turn < max_turns:
        turn += 1
        _mark_rolling_cache(messages)
        resp = _create(messages, model=model, tools=tool_defs,
                       max_tokens=max_tokens, effort=effort, system=system)
        cache_stats.append(cache_usage(resp))

        refusal = _refusal_error(resp)
        if refusal is not None:
            raise refusal

        text = _text_of(resp)
        calls = [b for b in resp.content
                 if getattr(b, "type", "") == "tool_use"]
        transcript.append({"role": "assistant", "content": text,
                           "tool_calls": [{"name": c.name, "input": c.input}
                                          for c in calls],
                           "stop_reason": resp.stop_reason})

        # The whole content list goes back, not the text pulled out of it:
        # thinking blocks have to be echoed unchanged on the same model.
        messages.append({"role": "assistant", "content": resp.content})

        if not calls:
            if (require_file and cwd is not None
                    and not (cwd / require_file).is_file()
                    and not nudged and turn < max_turns):
                nudged = True
                note = (f"`{require_file}` does not exist in the working "
                        "directory yet -- your work is only graded from "
                        "that file.  Please continue and save it.")
                messages.append({"role": "user", "content": note})
                transcript.append({"role": "user", "content": note})
                if verbose:
                    print(f"    [claude turn {turn}] stopped without "
                          f"{require_file}; nudged once")
                continue
            final_text = text
            break

        results = []
        for c in calls:
            result = clip(dispatch(cwd or Path.cwd(), c.name, dict(c.input)))
            if verbose:
                preview = str(dict(c.input))[:110]
                print(f"    [claude turn {turn}] {c.name}({preview})"
                      f" -> {len(result)} chars")
            results.append({"type": "tool_result", "tool_use_id": c.id,
                            "content": result})
            transcript.append({"role": "tool", "name": c.name,
                               "content": result})
        messages.append({"role": "user", "content": results})
    else:
        if verbose:
            print(f"    [claude] turn budget exhausted ({max_turns})")

    model_id, _ = resolve_model(model)
    return final_text, {"model": model_id, "turns": turn,
                        "transcript": transcript, "nudged": nudged,
                        "cache": cache_stats}


def label(model: str = DEFAULT_MODEL) -> str:
    model_id, effort = resolve_model(model)
    pretty = DISPLAY_NAMES.get(model, model)
    return f"{pretty} (agent, Azure, {model_id}, effort={effort})"


def solve(user_text, image_files=(), pdf_pages=(), pdf_files=(),
          code_files=(), pointcloud_files=(), *, model: str = DEFAULT_MODEL):
    workdir = tempfile.mkdtemp(prefix="claude_agent_")
    try:
        files = list(code_files) + list(pointcloud_files)
        text, meta = run_agent(
            user_text + AGENT_TASK_SUFFIX,
            cwd=workdir, model=model, files=files,
            inline_files=(list(image_files) + list(pdf_pages)
                          + list(pdf_files)),
            require_file=AGENT_SOLUTION_FILENAME,
        )
        return read_back_solution(workdir, text), meta
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def call_claude(prompt: str, *, model: str = DEFAULT_MODEL,
                images=(), effort: str = "high",
                retries: int = CALL_RETRIES) -> str:
    """Judge call: text (and optionally pictures) in, text out, no tools."""
    last = None
    for attempt in range(retries):
        try:
            content = [{"type": "text", "text": prompt}]
            content += [_media_block(i) for i in images]
            resp = _create([{"role": "user", "content": content}],
                           model=model, tools=None,
                           max_tokens=DEFAULT_MAX_TOKENS, effort=effort)
            refusal = _refusal_error(resp)
            if refusal is not None:
                raise refusal
            text = _text_of(resp)
            if text:
                return text
            last = RuntimeError(
                f"empty reply (stop_reason={resp.stop_reason})")
        except Exception as exc:                            # noqa: BLE001
            last = exc
            if is_configuration_error(exc):
                raise RuntimeError(
                    f"call_claude cannot run here, and trying again will "
                    f"not change it: {exc}") from exc
        if attempt < retries - 1:
            time.sleep(CALL_RETRY_BACKOFF_S * (attempt + 1))
    raise RuntimeError(
        f"call_claude failed after {retries} attempts: {last}") from last


if __name__ == "__main__":
    import sys

    which = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL
    print(label(which))
    print(call_claude("In one sentence, what is a living hinge?", model=which))
