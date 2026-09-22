#!/usr/bin/env python3
"""
selfcheck.py — mechanical consistency checks for this toolkit.

Run it after editing, renaming or porting anything:

    python selfcheck.py            # from the repo root
    python selfcheck.py --strict   # exit 1 on WARN as well as FAIL

It verifies the invariants that break SILENTLY — the ones where a mistake produces
an empty run or a skipped model rather than a traceback. It is deliberately
repo-agnostic: nothing is hardcoded to a particular filename, so it works
unchanged in a renamed or scrubbed copy.

WHAT IT CHECKS

  1. compile        every .py parses
  2. extractor ref  each EXTRACTOR = ...with_name("X") points at a file that exists
  3. residual I/O   the filenames the extractor WRITES are exactly the ones the
                    consumers READ                                  <-- the silent one
  4. helper drift   scripts carrying their own copy of the weight-edit / snapshot /
                    restore helpers agree on which modules they touch
  5. doc flags      every --flag cited in README/USAGE exists in that script's argparse
  6. doc scripts    every script named in the docs exists, and vice versa
  7. figure claims  dpi values and png+pdf pairing match what the docs assert

WHY THIS EXISTS

  A scrub that genericised variable names once rewrote string literals too, so a
  consumer looked for `a_residuals.pt` while the extractor wrote `prs_residuals.pt`.
  Nothing crashed: every model hit an `if not exists: continue` branch and the run
  finished with no output and no error. Check 3 exists for that class of bug.
"""

import argparse
import ast
import re
import sys
from fnmatch import fnmatch
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS = []


def record(level, check, msg):
    RESULTS.append((level, check, msg))


def scripts():
    return sorted(p for p in ROOT.glob("*.py") if p.name != Path(__file__).name)


# ---------------------------------------------------------------- 1. compile
def check_compile():
    for p in scripts():
        try:
            ast.parse(p.read_text())
        except SyntaxError as e:
            record("FAIL", "compile", f"{p.name}: line {e.lineno}: {e.msg}")
    if not any(r[1] == "compile" for r in RESULTS):
        record("PASS", "compile", f"{len(scripts())} scripts parse")


# ------------------------------------------------------- 2. extractor refs
EXTRACTOR_RE = re.compile(r'EXTRACTOR\s*=\s*Path\(__file__\)\.with_name\(\s*"([^"]+)"')


def check_extractor_refs():
    n = 0
    for p in scripts():
        for target in EXTRACTOR_RE.findall(p.read_text()):
            n += 1
            if not (ROOT / target).exists():
                record("FAIL", "extractor-ref",
                       f"{p.name} imports '{target}' which does not exist")
    if n:
        record("PASS", "extractor-ref", f"{n} cross-script reference(s) resolve")
    else:
        record("WARN", "extractor-ref", "no EXTRACTOR references found — pattern changed?")


# --------------------------------------------------------- 3. residual I/O
SAVE_RE = re.compile(r'torch\.save\([^)]*?["\']([a-zA-Z0-9_]+\.pt)["\']')
LOAD_RE = re.compile(r'torch\.load\([^)]*?["\']([a-zA-Z0-9_]+\.pt)["\']')
EXISTS_RE = re.compile(r'["\']([a-zA-Z0-9_]+\.pt)["\']\s*\)\s*\.exists\(\)')


def check_residual_io():
    written, read = {}, {}
    for p in scripts():
        src = p.read_text()
        for f in SAVE_RE.findall(src):
            written.setdefault(f, set()).add(p.name)
        for f in set(LOAD_RE.findall(src)) | set(EXISTS_RE.findall(src)):
            read.setdefault(f, set()).add(p.name)

    if not written:
        record("WARN", "residual-io", "no torch.save of a .pt file found")
        return

    orphans = {f: v for f, v in read.items() if f not in written}
    if orphans:
        for f, users in sorted(orphans.items()):
            record("FAIL", "residual-io",
                   f"'{f}' is read by {', '.join(sorted(users))} but written by nothing "
                   f"(producers write: {', '.join(sorted(written))}) — "
                   f"this fails SILENTLY via the exists() guard")
    unused = {f for f in written if f not in read}
    if unused:
        record("WARN", "residual-io",
               f"written but never read: {', '.join(sorted(unused))}")
    if not orphans:
        record("PASS", "residual-io",
               f"all read filenames are produced ({', '.join(sorted(written))})")


# --------------------------------------------------------- 4. helper drift
MODULE_RE = re.compile(r'\b\w+\.(?:attn|self_attn|mlp)\.(\w+_proj)\b')


def check_helper_drift():
    per_file = {}
    for p in scripts():
        mods = set(MODULE_RE.findall(p.read_text()))
        if mods:
            per_file[p.name] = mods
    if not per_file:
        record("WARN", "helper-drift", "no residual-writing projections found")
        return
    allsets = {frozenset(v) for v in per_file.values()}
    if len(allsets) > 1:
        detail = "; ".join(f"{k}: {{{', '.join(sorted(v))}}}" for k, v in sorted(per_file.items()))
        record("FAIL", "helper-drift",
               f"scripts disagree on which projections to edit — {detail}")
    else:
        record("PASS", "helper-drift",
               f"{len(per_file)} script(s) agree on {{{', '.join(sorted(next(iter(allsets))))}}}")


# ------------------------------------------------------------- 5. doc flags
def docs():
    return [p for p in (ROOT / "README.md", ROOT / "USAGE.md") if p.exists()]


def check_doc_flags():
    checked = 0
    for doc in docs():
        text = doc.read_text()
        # A command is its first line plus any lines joined by a trailing backslash.
        # Matching greedily across blank lines attributes the NEXT command's flags to
        # this one -- which produced three false failures the first time this ran.
        for m in re.finditer(r'python\s+([a-z_]+\.py)((?:[^\n]*\\\n)*[^\n]*)', text):
            name, tail = m.group(1), m.group(2)
            target = ROOT / name
            if not target.exists():
                record("FAIL", "doc-scripts", f"{doc.name} invokes '{name}' which does not exist")
                continue
            real = set(re.findall(r'"(--[a-z-]+)"', target.read_text()))
            cited = set(re.findall(r'(?<![\w-])(--[a-z][a-z-]*)', tail))
            bogus = {c for c in cited if c not in real and len(c) > 3}
            checked += 1
            for b in sorted(bogus):
                record("FAIL", "doc-flags", f"{doc.name} passes {b} to {name}, which has no such flag")
    if checked:
        record("PASS", "doc-flags", f"{checked} documented invocation(s) use only real flags")


# ----------------------------------------------------------- 6. doc scripts
def check_doc_coverage():
    named, invoked, globs = set(), set(), []
    for doc in docs():
        text = doc.read_text()
        named |= set(re.findall(r'`([a-z_0-9]+\.py)`', text))
        invoked |= set(re.findall(r'python\s+([a-z_0-9]+\.py)', text))
        # Docs legitimately refer to timestamped scripts by wildcard
        # (`ppigplm_residuals_*.py`); treat that as covering the matching files.
        globs += re.findall(r'`([a-z_0-9]+\*\.py)`', text)
    have = {p.name for p in scripts()}
    for g in globs:
        named |= {n for n in have if fnmatch(n, g)}
    # Only an INVOKED script must exist here. A backticked name in prose may be the
    # user's own model definition (e.g. `model.py`), which lives outside this repo.
    missing = invoked - have - {Path(__file__).name}
    undocumented = have - named - {Path(__file__).name}
    for m in sorted(missing):
        record("FAIL", "doc-scripts", f"docs reference '{m}' which does not exist")
    for u in sorted(undocumented):
        record("WARN", "doc-scripts", f"'{u}' exists but is not mentioned in the docs")
    if not missing and not undocumented:
        record("PASS", "doc-scripts", f"docs and repo agree on {len(have)} script(s)")


# --------------------------------------------------------- 7. figure claims
def check_figure_claims():
    dpis, pairs, savefigs = set(), 0, 0
    for p in scripts():
        src = p.read_text()
        dpis |= {int(d) for d in re.findall(r'dpi=(\d+)', src)}
        pairs += len(re.findall(r'for ext in \("png", "pdf"\)', src))
        savefigs += len(re.findall(r'\.savefig\(', src))
    if savefigs and pairs < savefigs:
        record("WARN", "figures",
               f"{savefigs} savefig call(s) but only {pairs} png+pdf loop(s) — "
               f"some figures may be one format only")
    for doc in docs():
        text = doc.read_text()
        for claimed in {int(d) for d in re.findall(r'(\d{3})\s*dpi', text)}:
            if claimed not in dpis:
                record("FAIL", "figures",
                       f"{doc.name} claims {claimed} dpi; code uses "
                       f"{', '.join(map(str, sorted(dpis)))}")
    # Only flag undocumented dpi if the docs discuss dpi at all; many repos do not.
    mentions_dpi = any(re.search(r'\bdpi\b', doc.read_text(), re.I) for doc in docs())
    undocumented = ({d for d in dpis if not any(str(d) in doc.read_text() for doc in docs())}
                    if mentions_dpi else set())
    if undocumented:
        record("WARN", "figures",
               f"dpi value(s) {', '.join(map(str, sorted(undocumented)))} used in code "
               f"but not mentioned in the docs")
    if not any(r[1] == "figures" for r in RESULTS):
        record("PASS", "figures", f"dpi values {sorted(dpis)} consistent with the docs")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--strict", action="store_true", help="exit non-zero on WARN too")
    args = ap.parse_args()

    for fn in (check_compile, check_extractor_refs, check_residual_io,
               check_helper_drift, check_doc_flags, check_doc_coverage,
               check_figure_claims):
        fn()

    width = max(len(c) for _, c, _ in RESULTS)
    icon = {"PASS": "ok  ", "WARN": "warn", "FAIL": "FAIL"}
    for level in ("FAIL", "WARN", "PASS"):
        for lv, check, msg in RESULTS:
            if lv == level:
                print(f"  [{icon[lv]}] {check:<{width}}  {msg}")

    fails = sum(1 for lv, _, _ in RESULTS if lv == "FAIL")
    warns = sum(1 for lv, _, _ in RESULTS if lv == "WARN")
    print(f"\n{fails} failure(s), {warns} warning(s)")
    if fails or (warns and args.strict):
        sys.exit(1)


if __name__ == "__main__":
    main()
