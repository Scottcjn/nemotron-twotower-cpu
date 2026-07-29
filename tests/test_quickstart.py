# SPDX-License-Identifier: AGPL-3.0-or-later
"""The documented setup must actually produce a runnable inference_cpu.py.

CI cannot download NVIDIA's gated 120 GB checkpoint, so the patch surface used
to be untested end to end. It does not have to stay that way: a unified diff
already carries its own pre-image (the context and '-' lines, at known line
numbers), so we can synthesise a stand-in file that the patch must apply to,
run the exact commands the README prints, and check what comes out. No model
download, no NVIDIA source added to this repo, no network.
"""
import os
import re
import shutil
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATCHES = os.path.join(ROOT, "patches")
MODELING_PATCH = os.path.join(PATCHES, "modeling_nemotron_h.cpu.patch")
INFERENCE_PATCH = os.path.join(PATCHES, "inference.cpu.patch")
FILLER = "# not part of any hunk\n"

pytestmark = pytest.mark.skipif(
    shutil.which("patch") is None, reason="GNU patch not installed"
)


# --------------------------------------------------------------------------
# helpers: rebuild the pre-/post-image of a unified diff from the diff itself
# --------------------------------------------------------------------------

def parse_hunks(patch_path):
    """-> [(old_start, [old lines], new_start, [new lines])], 1-based starts."""
    hunks = []
    header = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
    cur = None
    with open(patch_path, encoding="utf-8") as fh:
        for line in fh:
            m = header.match(line)
            if m:
                cur = (int(m.group(1)), [], int(m.group(3)), [])
                hunks.append(cur)
                continue
            if cur is None or line.startswith(("---", "+++", "\\")):
                continue
            tag, body = line[:1], line[1:]
            if tag == " ":
                cur[1].append(body)
                cur[3].append(body)
            elif tag == "-":
                cur[1].append(body)
            elif tag == "+":
                cur[3].append(body)
    assert hunks, f"no hunks parsed from {patch_path}"
    return hunks


def _image(hunks, side):
    """Synthesise a file containing the hunks' old (side=0) or new (side=1) text
    at their recorded line numbers, filler everywhere else."""
    start_i, lines_i = (0, 1) if side == 0 else (2, 3)
    size = max(h[start_i] - 1 + len(h[lines_i]) for h in hunks)
    out = [FILLER] * size
    for h in hunks:
        at = h[start_i] - 1
        out[at:at + len(h[lines_i])] = h[lines_i]
    return "".join(out)


def preimage(patch_path):
    return _image(parse_hunks(patch_path), 0)


def postimage(patch_path):
    return _image(parse_hunks(patch_path), 1)


def run(cmd, cwd, **kw):
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, **kw)


def read(path):
    with open(path, encoding="utf-8") as fh:
        return fh.read()


def write(path, text):
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)


def readme_quickstart():
    """The first ```bash fence under '## Quick start'."""
    text = read(os.path.join(ROOT, "README.md"))
    body = text.split("## Quick start", 1)[1]
    return body.split("```bash", 1)[1].split("```", 1)[0]


# --------------------------------------------------------------------------
# the patches apply at all
# --------------------------------------------------------------------------

def test_modeling_patch_applies_to_its_own_preimage(tmp_path):
    d = tmp_path / "twotower"
    d.mkdir()
    target = d / "modeling_nemotron_h.py"
    write(str(target), preimage(MODELING_PATCH))

    r = run(["patch", "-p0"], str(d), stdin=open(MODELING_PATCH, "rb"))
    assert r.returncode == 0, r.stdout + r.stderr
    assert read(str(target)) == postimage(MODELING_PATCH)


def test_modeling_patch_replaces_every_list_level_device_access():
    """The whole point of the patch: no `.ssm_states.device` / `.conv_states.device`
    / `.conv_states.zero_()` left on the new side."""
    new = postimage(MODELING_PATCH)
    for bad in (".conv_states.device", ".ssm_states.device",
                ".conv_states.zero_()", ".ssm_states.zero_()"):
        assert bad not in new, f"{bad} survives the patch"
    assert "self.conv_states[layer_idx].device" in new
    assert "cache_params.ssm_states[self.layer_idx].device" in new


# --------------------------------------------------------------------------
# the inference patch has to produce inference_cpu.py -- the file run.sh runs
# --------------------------------------------------------------------------

def test_inference_patch_headers_name_inference_cpu():
    """Regression guard. With `--- inference.py` on the old side, GNU patch
    picks inference.py as the target even when inference_cpu.py exists, so the
    documented sequence silently patched NVIDIA's file and never created the
    one run.sh needs."""
    lines = read(INFERENCE_PATCH).splitlines()
    assert lines[0].startswith("--- inference_cpu.py"), lines[0]
    assert lines[1].startswith("+++ inference_cpu.py"), lines[1]


def test_readme_quickstart_produces_inference_cpu(tmp_path):
    """Run the README's step 2 verbatim and check the artefact exists."""
    root = tmp_path / "repo"
    shutil.copytree(PATCHES, str(root / "patches"))
    model = root / "twotower"
    model.mkdir()
    write(str(model / "inference.py"), preimage(INFERENCE_PATCH))
    write(str(model / "modeling_nemotron_h.py"), preimage(MODELING_PATCH))
    pristine = read(str(model / "inference.py"))

    quickstart = readme_quickstart()
    steps = [ln.strip() for ln in quickstart.splitlines()
             if ln.strip().startswith(("patch ", "cp "))]
    assert steps, "no patch/cp commands found in the README quick start"

    for step in steps:
        r = run(["bash", "-c", step], str(model))
        assert r.returncode == 0, f"README step failed: {step}\n{r.stdout}{r.stderr}"

    assert (model / "inference_cpu.py").exists(), (
        "the README quick start does not produce inference_cpu.py, "
        "which README step 3 and run.sh both execute"
    )
    assert read(str(model / "inference_cpu.py")) == postimage(INFERENCE_PATCH)
    assert read(str(model / "inference.py")) == pristine, (
        "NVIDIA's inference.py must be left untouched"
    )


def test_patched_inference_has_no_cuda_placement_left():
    new = postimage(INFERENCE_PATCH)
    assert ".cuda()" not in new
    assert new.count('.to("cpu")') == 3


# --------------------------------------------------------------------------
# the scripts and the README agree on the same filename
# --------------------------------------------------------------------------

def test_setup_run_and_readme_agree_on_inference_cpu():
    setup = read(os.path.join(ROOT, "setup.sh"))
    runsh = read(os.path.join(ROOT, "run.sh"))
    quickstart = readme_quickstart()
    assert "inference_cpu.py" in setup
    assert "inference_cpu.py" in runsh
    assert "inference_cpu.py" in quickstart


def test_readme_patch_paths_exist():
    """Every `patch -p0 < X` in the quick start must point at a real file,
    resolved from inside twotower/ as the README instructs."""
    for path in re.findall(r"patch -p0 <\s*(\S+)", readme_quickstart()):
        resolved = os.path.normpath(os.path.join(ROOT, "twotower", path))
        assert os.path.isfile(resolved), f"README references missing patch {path}"


def test_setup_applies_the_patch_file_rather_than_its_own_rewrite():
    """setup.sh used to sed `.cuda()` itself instead of applying
    patches/inference.cpu.patch, so the two routes could drift apart -- and did."""
    setup = read(os.path.join(ROOT, "setup.sh"))
    assert "patches/inference.cpu.patch" in setup
    assert "sed 's/\\.cuda()" not in setup


def test_setup_does_not_swallow_a_failed_patch():
    code = [ln for ln in read(os.path.join(ROOT, "setup.sh")).splitlines()
            if not ln.lstrip().startswith("#")]
    for ln in code:
        assert "|| true" not in ln, f"a failed patch must not be ignored: {ln}"


# --------------------------------------------------------------------------
# behaviour of the scripts themselves
# --------------------------------------------------------------------------

def test_run_sh_reports_missing_inference_cpu(tmp_path):
    empty = tmp_path / "model"
    empty.mkdir()
    env = dict(os.environ, TT_MODEL=str(empty))
    r = run(["bash", os.path.join(ROOT, "run.sh"), "hello"], ROOT, env=env)
    assert r.returncode != 0
    combined = r.stdout + r.stderr
    assert "inference_cpu.py" in combined and "setup.sh" in combined, combined
    assert "Traceback" not in combined


def test_run_sh_keeps_an_existing_pythonpath():
    runsh = read(os.path.join(ROOT, "run.sh"))
    assert "PYTHONPATH:+" in runsh, "run.sh must append to PYTHONPATH, not clobber it"


@pytest.mark.parametrize("script", ["run.sh", "setup.sh"])
def test_scripts_parse(script):
    r = run(["bash", "-n", os.path.join(ROOT, script)], ROOT)
    assert r.returncode == 0, r.stderr


# --------------------------------------------------------------------------
# the three-way dry-run test setup.sh's apply_patch depends on
# --------------------------------------------------------------------------

def test_patch_dry_run_distinguishes_fresh_applied_and_broken(tmp_path):
    d = tmp_path / "twotower"
    d.mkdir()
    target = d / "modeling_nemotron_h.py"

    def dry(flag):
        with open(MODELING_PATCH, "rb") as fh:
            return run(["patch", "-p0", flag, "--silent", "--dry-run"],
                       str(d), stdin=fh).returncode

    write(str(target), preimage(MODELING_PATCH))
    assert dry("--forward") == 0, "fresh file: forward dry-run should succeed"

    write(str(target), postimage(MODELING_PATCH))
    assert dry("--forward") != 0, "already applied: forward dry-run should fail"
    assert dry("--reverse") == 0, "already applied: reverse dry-run should succeed"

    write(str(target), FILLER * 900)
    assert dry("--forward") != 0 and dry("--reverse") != 0, "garbage: both fail"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
