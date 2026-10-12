"""Provider-neutral scratch-directory scaffolding shared by the three routes.

All three routes drive their own agent loop over the four tools implemented
here -- bash / write_file / read_file / list_dir against a scratch cwd. None
of the three providers supplies a tool runtime we use: Claude reaches Azure
through the plain Messages API, which has no built-in tools either.

THIS MODULE EXISTS SO THE THREE ROUTES SHARE A PROMPT, NOT JUST A SIGNATURE.
An eval that gives Claude "inspect the files, run your script, save
solution.py" and GPT something worded differently is comparing prompts as
much as models. `agent_task_suffix()` and `attached_files_note()` are the
single source of both, parameterised only by what each provider calls its
tools.

It deliberately does NOT import any provider SDK. Each `call_*` module
imports its own at module scope -- see the note in `call_llm` about the
import being the point of failure -- so putting the shared pieces in one of
them would make the other two unimportable wherever that one SDK is
missing, which is exactly the coupling the route split is meant to avoid.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

# One turn = one API round trip, which may contain several tool calls.
DEFAULT_AGENT_MAX_TURNS = 40

TOOL_OUTPUT_LIMIT = 15000
BASH_TIMEOUT_DEFAULT_S = 300
BASH_TIMEOUT_MAX_S = 900

AGENT_SOLUTION_FILENAME = "solution.py"

_SHELL = "cmd.exe" if os.name == "nt" else "/bin/sh"


def clip(text: str, limit: int = TOOL_OUTPUT_LIMIT) -> str:
    """Keep the head and (mostly) the tail of an over-long tool result.

    Tail-weighted on purpose: when a build script fails, the traceback is at
    the end, and a head-only clip throws away the only part worth reading.
    """
    if len(text) <= limit:
        return text
    head, tail = text[: limit // 4], text[-(limit * 3 // 4):]
    return head + f"\n...[{len(text) - limit} chars clipped]...\n" + tail


# --------------------------------------------------------------------------
# Tool implementations. Every one takes cwd first and returns a plain string;
# a tool that raises is reported back to the model as text rather than
# killing the run, since a recoverable mistake (bad path, bad JSON) is
# something the model can act on and a crash is not.
# --------------------------------------------------------------------------

#: After the tree is killed, how long to wait for the pipes to close.
BASH_DRAIN_S = 10


def _kill_tree(proc):
    """Kill the shell AND everything it spawned.

    Killing only the direct child is not enough, and the difference is not
    theoretical: a SolidWorks run wedged for 286 minutes because the shell
    launched ScriptRunner.exe, ScriptRunner blocked on a COM call, and the
    timeout killed `cmd.exe` while the orphan lived on.
    """
    if os.name == "nt":
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                       capture_output=True)
    else:
        import signal
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError):
            pass
    try:
        proc.kill()
    except OSError:
        pass


def tool_bash(cwd, command, timeout_s=None):
    """Run a command, and be genuinely killable.

    NOT `subprocess.run(..., timeout=)`, which does not survive contact with
    a command that outlives its shell. On timeout it kills the direct child
    and then reads the pipes again -- and a surviving grandchild still holds
    the write end, so the read blocks forever. The timeout looks present and
    does nothing; the agent loop hangs with no turn, no error and no ceiling.
    Popen plus an explicit tree kill is what actually bounds the call.
    """
    t = min(int(timeout_s or BASH_TIMEOUT_DEFAULT_S), BASH_TIMEOUT_MAX_S)

    # A COMMAND WITH EMBEDDED NEWLINES DOES NOT RUN ON WINDOWS, and fails in
    # the worst possible way: cmd.exe breaks at the first newline, the body
    # never executes, and the call returns a bare `exit=0` with no output and
    # no stderr. That is indistinguishable from "ran fine, printed nothing".
    # One agent read it as success forty turns in a row while doing nothing
    # at all. Refusing it with an explanation costs one turn; the silence
    # cost forty.
    if os.name == "nt" and chr(10) in command:
        return ("ERROR: this command was NOT executed. It contains newlines, "
                "and cmd.exe splits on them -- the body after the first line "
                "never runs, which is why such commands appear to succeed "
                "with empty output. Write the script to a file with "
                "write_file and run that file instead, e.g. "
                "write_file('probe.py', ...) then bash('python probe.py').")

    kw = {} if os.name == "nt" else {"start_new_session": True}
    proc = subprocess.Popen(command, shell=True, cwd=str(cwd),
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, errors="replace", **kw)
    try:
        out, err = proc.communicate(timeout=t)
    except subprocess.TimeoutExpired:
        _kill_tree(proc)
        try:
            out, err = proc.communicate(timeout=BASH_DRAIN_S)
        except subprocess.TimeoutExpired:
            return (f"TIMEOUT: killed after {t}s, but something it spawned "
                    "still holds the output pipe, so its output is lost. "
                    "Whatever you started is detached -- do not wait on it "
                    "again; check for a result file on disk instead.")
        partial = (out or "")
        if err:
            partial += ("\n--- stderr ---\n" if partial else
                        "--- stderr ---\n") + err
        head = (f"TIMEOUT: command killed after {t}s, along with every "
                "process it started")
        return (head + ("\n" + partial if partial.strip() else "")).strip()

    out = out or ""
    if err:
        out += ("\n--- stderr ---\n" if out else "--- stderr ---\n") + err
    return f"exit={proc.returncode}\n{out}".strip()


def tool_write_file(cwd, path, content):
    p = Path(cwd) / path
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return f"wrote {len(content)} chars to {path}"


def tool_read_file(cwd, path, start_line=None, line_count=None):
    p = Path(cwd) / path
    if not p.is_file():
        return f"ERROR: {path} does not exist"
    try:
        lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception as exc:
        return f"ERROR reading {path}: {exc}"
    start = max(int(start_line or 1), 1) - 1
    stop = start + int(line_count) if line_count else len(lines)
    return "\n".join(lines[start:stop]) or "(empty)"


def tool_list_dir(cwd, pattern=None):
    cwd = Path(cwd)
    rows = []
    for p in sorted(cwd.glob(pattern or "*")):
        try:
            size = "<dir>" if p.is_dir() else str(p.stat().st_size)
        except OSError:
            size = "?"
        rows.append(f"{p.relative_to(cwd)}  {size}")
    return "\n".join(rows) or "(no matches)"


TOOL_IMPL = {
    "bash": tool_bash,
    "write_file": tool_write_file,
    "read_file": tool_read_file,
    "list_dir": tool_list_dir,
}


# Neutral JSON-Schema specs. `call_gpt` wraps these in OpenAI's
# {"type": "function", ...} envelope and `call_gemini` converts them to
# `types.FunctionDeclaration`, so both models see the same four tools with
# the same descriptions.
TOOL_SPECS = [
    {
        "name": "bash",
        "description": (
            f"Run a shell command ({_SHELL}) in the working directory and "
            "return exit code + stdout/stderr.  `python` is on PATH.  Prefer "
            "writing scripts with write_file and running them here over long "
            "one-liners."),
        "parameters": {"type": "object", "properties": {
            "command": {"type": "string",
                        "description": "the command line to run"},
            "timeout_s": {"type": "integer",
                          "description": f"seconds (default "
                                         f"{BASH_TIMEOUT_DEFAULT_S}, max "
                                         f"{BASH_TIMEOUT_MAX_S})"},
        }, "required": ["command"]},
    },
    {
        "name": "write_file",
        "description": "Write a UTF-8 text file (path relative to the "
                       "working directory), creating parent dirs.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "relative file path"},
            "content": {"type": "string", "description": "full file contents"},
        }, "required": ["path", "content"]},
    },
    {
        "name": "read_file",
        "description": "Read a UTF-8 text file (path relative to the working "
                       "directory).  Optional 1-based start line and line "
                       "count.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "relative file path"},
            "start_line": {"type": "integer", "description": "1-based"},
            "line_count": {"type": "integer",
                           "description": "how many lines to return"},
        }, "required": ["path"]},
    },
    {
        "name": "list_dir",
        "description": "List files matching a glob pattern relative to the "
                       "working directory (default '*'; '**/*' recurses).",
        "parameters": {"type": "object", "properties": {
            "pattern": {"type": "string", "description": "glob, e.g. '**/*'"},
        }},
    },
]


def dispatch(cwd, name, args):
    impl = TOOL_IMPL.get(name)
    if impl is None:
        return f"ERROR: unknown tool {name!r}"
    try:
        return impl(cwd, **args)
    except TypeError as exc:
        return f"ERROR: bad arguments for {name}: {exc}"
    except Exception as exc:
        return f"ERROR: {name} raised {type(exc).__name__}: {exc}"


# --------------------------------------------------------------------------
# Prompt scaffolding
# --------------------------------------------------------------------------

BINARY_DATA_EXTS = {".ply", ".stl", ".step", ".stp", ".obj", ".3mf"}

BINARY_DATA_HINT = (
    "Do NOT open it with {read} (it's binary) -- inspect it from {bash} with "
    "Python, e.g. `python3 -c \"import trimesh, numpy as np; "
    "v = np.asarray(trimesh.load('{name}').vertices); print(v.shape, "
    "v.min(0), v.max(0))\"`. You have trimesh/numpy available, so you can "
    "measure the real coordinates (extents, cross-sections, clusters) "
    "rather than estimating from renders."
)


def attach_files(cwd, files) -> list[str]:
    """Copy reference files into the scratch dir, keeping their own names."""
    cwd = Path(cwd)
    cwd.mkdir(parents=True, exist_ok=True)
    names = []
    for f in files:
        f = Path(f)
        dest = cwd / f.name
        if f.resolve() != dest.resolve():
            shutil.copy2(f, dest)
        names.append(f.name)
    return names


def attached_files_note(names, *, read_tool: str = "Read",
                        bash_tool: str = "Bash") -> str:
    names = list(names)
    if not names:
        return ""
    lines = ["\n\nThese files are in your current working directory:"]
    for name in names:
        if Path(name).suffix.lower() in BINARY_DATA_EXTS:
            lines.append(f"- `{name}` -- " + BINARY_DATA_HINT.format(
                name=name, read=read_tool, bash=bash_tool))
        else:
            lines.append(f"- `{name}`")
    return "\n".join(lines)


def agent_task_suffix(*, bash_tool: str = "Bash",
                      solution_filename: str = AGENT_SOLUTION_FILENAME) -> str:
    return (
        "\n\nYou're working in a scratch directory that has the reference "
        "material above copied in as files (listed below) -- inspect them "
        f"yourself if that helps. Feel free to run your script with "
        f"{bash_tool} to check it actually builds before finalizing (e.g. "
        "`python3 script.py`). When you're done, write your final, complete, "
        f"self-contained script to a file named exactly "
        f"`{solution_filename}` in the current directory."
    )


def read_back_solution(workdir, text: str,
                       filename: str = AGENT_SOLUTION_FILENAME) -> str:
    """The graded artefact is the file on disk, not the chat text.

    An agent that writes solution.py and then says "done" would otherwise be
    scored on the word "done"; when the file is missing we fall back to the
    final message so a one-shot-style answer still grades.
    """
    p = Path(workdir) / filename
    if p.is_file():
        return f"```python\n{p.read_text(encoding='utf-8', errors='replace')}\n```"
    return text
