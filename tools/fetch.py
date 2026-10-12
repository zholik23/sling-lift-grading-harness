"""Materialize large CAD assets that live in Azure Blob Storage, not git.

Walks every task.toml under the repo, reads its [[metadata.assets]] entries
(path, url, sha256), and downloads any file that is missing on disk or fails
its checksum. Run once after cloning (or after tools/sync.py changes the remote):

    python3 tools/fetch.py            # fetch everything missing
    python3 tools/fetch.py FreeCAD    # restrict to a subtree
    python3 tools/fetch.py --check    # report what is missing, download nothing

Flags may be combined with a subtree, in any order. A subtree is matched
against the POSIX form of the task path, so both slash styles work on
Windows.
"""
import hashlib
import sys
import time
import tomllib
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load_toml(path: Path) -> dict:
    """TOML is UTF-8 by definition -- read it as bytes and let tomllib say so.

    Path.read_text() without an explicit encoding uses the LOCALE default,
    which on a non-UTF-8 Windows console codepage silently mangles every
    non-ASCII byte. That is not a cosmetic bug here: asset names carry Turkish
    letters, the mangled name goes straight into the blob URL, and every one
    of them 404s while the ASCII names sail through. Reading in binary removes
    the locale from the picture entirely.
    """
    with path.open("rb") as f:
        return tomllib.load(f)


def _ok(dest: Path, sha256: str) -> bool:
    return _status(dest, sha256) == "ok"


def _status(dest: Path, sha256: str) -> str:
    """"ok" | "absent" | "stale".

    Absent and stale both mean "fetch would download it", but they are very
    different facts about the working copy, and collapsing them hides the
    interesting one: SolidWorks rewrites part files when it opens an assembly
    for edit, so a folder that has been graded once reads as `stale` while
    being complete. Reporting that as "missing" invites re-downloading over
    measurements already taken.
    """
    if not dest.exists():
        return "absent"
    if hashlib.sha256(dest.read_bytes()).hexdigest() != sha256:
        return "stale"
    return "ok"


MISSING: list[str] = []
LOCKED: list[str] = []
RENORMALISED: list[str] = []
CORRUPT: list[str] = []
STALE: list[str] = []
PENDING: list[tuple[str, int, str]] = []  # (folder label, bytes, status)


def _name_variants(rel: str):
    """The blob name, then its other Unicode normalisation.

    Composed and decomposed spellings of the same name are different blob
    names, and this dataset genuinely uses both: the `environment` folder is
    stored decomposed, `solution` composed. Manifests match today, but a
    re-upload from a different OS flips it -- macOS hands out NFD, Windows
    NFC. Trying the alternate form turns that from a silent 404 into a hit.
    """
    seen = [rel]
    for form in ("NFC", "NFD"):
        alt = unicodedata.normalize(form, rel)
        if alt not in seen:
            seen.append(alt)
    return seen


def _safe_url(url: str) -> str:
    """Percent-encode the PATH of an already-composed URL, once.

    `_download` builds `f"{url_base}/{quote(name)}"`, which quotes the
    manifest name and leaves `url_base` exactly as task.toml wrote it.
    That held until a task pinned folders whose names contain spaces --
    task 14's `environment/01. Base Frame` -- and then the composed URL
    carries a raw space in the directory and `%20` in the file name, and
    `http.client` refuses the whole request:

        InvalidURL: URL can't contain control characters.
        '/assets/.../environment/01. Base Frame/01.%20MAIN_SIDE_FRAME.SLDASM'

    Fixing it here rather than at either call site covers folder pins and
    single-file pins together, and covers a `url` that was already written
    correctly: `safe="/%"` keeps the separators and leaves existing `%XX`
    escapes alone, so quoting an already-quoted URL is a no-op rather than
    turning `%20` into `%2520`.

    The cost of that choice is a literal `%` in a blob name, which would
    pass through unencoded. A name that really contains a percent sign
    would need per-segment quoting and a way to tell the two cases apart,
    which is not worth carrying for a case this repository does not have.

    Scheme, host, query and fragment are left untouched -- a SAS token in
    the query must not be re-encoded.
    """
    parts = urllib.parse.urlsplit(url)
    if not parts.scheme:                       # a bare path, not a URL
        return url
    return urllib.parse.urlunsplit(
        parts._replace(path=urllib.parse.quote(parts.path, safe="/%")))


def _swap_in(tmp: Path, dest: Path) -> bool:
    """Move the verified download over `dest`. On WSL (/mnt/c) the rename
    fails with EACCES while a Windows process -- SolidWorks, typically --
    holds the old file open; retry briefly, then leave the download beside
    it and report rather than crash."""
    for attempt in range(5):
        try:
            tmp.replace(dest)
            return True
        except PermissionError:
            time.sleep(1 + attempt)
    try:
        dest.unlink()
        tmp.replace(dest)
        return True
    except OSError:
        return False


def _download(url_base: str, rel: str, dest: Path, sha256: str) -> bool:
    """url_base is the folder prefix; rel is the manifest name (unquoted)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    # Download beside the target and swap in only on success. A stale local
    # file is still the best available copy when the blob 404s or fails its
    # checksum; deleting it (as this used to) left the tree with nothing.
    tmp = dest.with_name(dest.name + ".fetch-part")
    variants = _name_variants(rel) if url_base else [rel]
    last_code = None
    for idx, name in enumerate(variants):
        url = _safe_url(
            f"{url_base}/{urllib.parse.quote(name)}" if url_base else rel)
        for attempt in range(4):
            try:
                urllib.request.urlretrieve(url, tmp)
                if idx:
                    RENORMALISED.append(str(dest.relative_to(ROOT)))
                if not _ok(tmp, sha256):
                    # One bad blob used to raise SystemExit, which killed the
                    # run and left every later asset silently un-attempted --
                    # the partial tree then looks exactly like a finished one.
                    # Report it and keep going instead.
                    CORRUPT.append(str(dest.relative_to(ROOT)))
                    tmp.unlink(missing_ok=True)
                    return False
                if not _swap_in(tmp, dest):
                    LOCKED.append(f"{dest.relative_to(ROOT)} (downloaded copy left at "
                                  f"{tmp.name}; close the file in SolidWorks and re-run)")
                    return False
                return True
            except urllib.error.HTTPError as exc:
                # A pin can point at a blob that has not been synced up yet
                # (or was deleted remotely). Before believing that, try the
                # other normalisation of the name.
                last_code = exc.code
                tmp.unlink(missing_ok=True)
                break
            except (urllib.error.URLError, ConnectionError, TimeoutError) as exc:
                if attempt == 3:
                    MISSING.append(
                        f"{dest.relative_to(ROOT)} (network: {exc})")
                    tmp.unlink(missing_ok=True)
                    return False
                time.sleep(2 ** attempt)
    MISSING.append(f"{dest.relative_to(ROOT)} ({last_code})")
    return False


def _fetch_folder(task_dir: Path, asset: dict, opts: "Opts") -> int:
    """Folder pin: download each manifest row (relpath, sha256, bytes)
    under the entry's url prefix.

    Blob names and FOLDER names may both contain spaces; every path
    segment of the composed URL is percent-quoted by `_safe_url`. Only
    the manifest name used to be, which is why task 14's
    `environment/01. Base Frame` raised InvalidURL on its first row."""
    fetched = 0
    base = asset["url"].rstrip("/")
    label = str((task_dir / asset["path"]).relative_to(ROOT))
    for rel, sha, size in asset["manifest"]:
        dest = task_dir / asset["path"] / rel
        state = _status(dest, sha)
        if state == "ok":
            continue
        if opts.check:
            PENDING.append((label, int(size), state))
            continue
        print(f"fetching {dest.relative_to(ROOT)} ({size} bytes)")
        if _download(base, rel, dest, sha):
            fetched += 1
            if state == "stale":   # only counted once actually replaced
                STALE.append(str(dest.relative_to(ROOT)))
    return fetched


class Opts:
    """Command-line switches, so the walk does not read sys.argv itself."""

    def __init__(self, subtree=None, check=False):
        self.subtree = subtree
        self.check = check


def _matches(task_dir: Path, subtree: str | None) -> bool:
    """Compare in POSIX form, both sides.

    str(PurePath) uses the platform separator, so on Windows the task path
    reads `SolidWorks\\17_industrial_electrical_panel` while anyone typing
    the argument writes it with forward slashes (that is how it appears in
    every URL and in the repo's own docs). The substring test then failed
    for every task, the walk selected nothing, and the run printed
    `done; 0 file(s) fetched` -- which reads exactly like "already complete"
    and is how a half-materialised dataset passes for a whole one.
    """
    if not subtree:
        return True
    want = subtree.replace("\\", "/").strip("/")
    return want in task_dir.relative_to(ROOT).as_posix()


def fetch_all(subtree: str | None = None, check: bool = False) -> int:
    opts = Opts(subtree, check)
    fetched = 0
    seen_tasks = 0
    for toml_path in sorted(ROOT.glob("*/*/task.toml")):
        task_dir = toml_path.parent
        if not _matches(task_dir, subtree):
            continue
        seen_tasks += 1
        meta = _load_toml(toml_path).get("metadata", {})
        for asset in meta.get("assets", []):
            if "manifest" in asset:
                fetched += _fetch_folder(task_dir, asset, opts)
                continue
            dest = task_dir / asset["path"]
            state = _status(dest, asset["sha256"])
            if state == "ok":
                continue
            if opts.check:
                PENDING.append((str(dest.relative_to(ROOT)),
                                int(asset.get("bytes", 0) or 0), state))
                continue
            print(f"fetching {dest.relative_to(ROOT)} "
                  f"({asset.get('bytes', '?')} bytes)")
            if _download("", asset["url"], dest, asset["sha256"]):
                fetched += 1
                if state == "stale":   # only counted once actually replaced
                    STALE.append(str(dest.relative_to(ROOT)))

    # A subtree that matched no task is a typo, not an empty result. Saying
    # so is the whole point: silence here is indistinguishable from success.
    if subtree and not seen_tasks:
        print(f"no task matched subtree {subtree!r}; nothing was checked")
        _report()
        return 0

    if check:
        _print_check(seen_tasks)
    else:
        print(f"done; {fetched} file(s) fetched across {seen_tasks} task(s)")
    _report()
    return fetched


def _print_check(seen_tasks: int) -> None:
    """Absent and stale are listed apart -- see _status."""
    if not PENDING:
        print(f"nothing to fetch across {seen_tasks} task(s)")
        return
    for state, word in (("absent", "absent"),
                        ("stale", "present but not matching the manifest")):
        rows = [(lab, b) for lab, b, st in PENDING if st == state]
        if not rows:
            continue
        total = sum(b for _, b in rows)
        print(f"{len(rows)} file(s) {word}, {total / 1e6:.0f} MB:")
        by_dir: dict[str, list[int]] = {}
        for lab, b in rows:
            by_dir.setdefault(lab, []).append(b)
        for d in sorted(by_dir):
            sizes = by_dir[d]
            print(f"  {d}: {len(sizes)} file(s), {sum(sizes) / 1e6:.0f} MB")
    if any(st == "stale" for _, _, st in PENDING):
        print("note: a plain fetch OVERWRITES the stale files above with the "
              "pinned blobs. SolidWorks rewrites parts when it opens an "
              "assembly for edit, so a folder already measured reads as "
              "stale; narrow the subtree if you mean to keep it.")


def _report() -> None:
    if STALE:
        print(f"{len(STALE)} file(s) were present but did not match the "
              "manifest, and were overwritten with the pinned blob:")
        for m in STALE[:10]:
            print(f"  {m}")
    if RENORMALISED:
        print(f"{len(RENORMALISED)} file(s) only resolved under the other "
              "Unicode normalisation of their name -- the manifest and the "
              "blob disagree on NFC/NFD:")
        for m in RENORMALISED[:10]:
            print(f"  {m}")
    if CORRUPT:
        print(f"{len(CORRUPT)} file(s) downloaded but failed their checksum "
              "(the blob does not match the manifest):")
        for m in CORRUPT:
            print(f"  {m}")
    if LOCKED:
        print(f"{len(LOCKED)} file(s) downloaded but could not replace the "
              "existing file (it is open in another program):")
        for m in LOCKED:
            print(f"  {m}")
    if MISSING:
        print(f"{len(MISSING)} pinned file(s) not in blob storage "
              "(not yet synced up, or deleted remotely):")
        for m in MISSING:
            print(f"  {m}")


if __name__ == "__main__":
    # The report prints blob names; a legacy console codepage would mangle
    # them on the way out, which is how a name bug hides in plain sight.
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    argv = sys.argv[1:]
    known = {"--check"}
    # A mistyped flag must not be read as a subtree, and must not be ignored
    # either: both failure modes end in a run that prints something reassuring
    # and does nothing. Same reason the subtree filter now speaks up (_matches).
    unknown = [a for a in argv if a.startswith("-") and a not in known]
    if unknown:
        raise SystemExit(f"unknown option(s): {' '.join(unknown)}\n"
                         f"usage: fetch.py [--check] [SUBTREE]")
    rest = [a for a in argv if not a.startswith("-")]
    if len(rest) > 1:
        raise SystemExit(f"expected at most one subtree, got: {' '.join(rest)}")
    fetch_all(rest[0] if rest else None, check="--check" in argv)
