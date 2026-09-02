import os
import subprocess
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
INSTALLER = REPOSITORY_ROOT / ".github/scripts/install-torch-stack.sh"
WORKFLOW = REPOSITORY_ROOT / ".github/workflows/ci.yml"


def _run_installer(tmp_path, *, wheel_failure):
    command_log = tmp_path / "commands.log"
    fake_micromamba = tmp_path / "micromamba"
    fake_micromamba.write_text(
        "#!/usr/bin/env bash\n"
        'printf \'%s\\n\' "$*" >> "${FAKE_COMMAND_LOG}"\n'
        'if [[ "$*" == *"--only-binary=:all:"* '
        '&& "${FAKE_WHEEL_FAILURE}" == "1" ]]; then\n'
        "  exit 1\n"
        "fi\n"
    )
    fake_micromamba.chmod(0o755)
    env = os.environ.copy()
    env.update(
        {
            "ENV_NAME": "test-environment",
            "FAKE_COMMAND_LOG": str(command_log),
            "FAKE_WHEEL_FAILURE": "1" if wheel_failure else "0",
            "PATH": f"{tmp_path}{os.pathsep}{env['PATH']}",
            "PYG_WHEEL_URL": "https://example.invalid/pyg-wheels.html",
            "TORCH_VERSION": "2.9.0",
        }
    )
    subprocess.run(["bash", str(INSTALLER)], env=env, check=True)
    return command_log.read_text().splitlines()


def test_torch_installer_prefers_cpu_and_binary_wheels(tmp_path):
    commands = _run_installer(tmp_path, wheel_failure=False)

    assert any("download.pytorch.org/whl/cpu" in line for line in commands)
    assert any("--only-binary=:all:" in line for line in commands)
    assert not any("--no-build-isolation" in line for line in commands)


def test_torch_installer_falls_back_to_bounded_source_build(tmp_path):
    commands = _run_installer(tmp_path, wheel_failure=True)

    assert any("--only-binary=:all:" in line for line in commands)
    assert any(
        "MAX_JOBS=2" in line and "--no-build-isolation" in line
        for line in commands
    )


def test_docs_job_collects_only_backend_neutral_example_modules():
    workflow = WORKFLOW.read_text()
    docs_job = workflow.split("\n  docs:\n", 1)[1].split(
        "\n  readthedocs-smoke:\n", 1
    )[0]

    assert "test_docs_transformations_basic.py" in docs_job
    assert "test_docs_trainset.py" in docs_job
    assert "src/aenet/geometry/tests src/aenet/tests" not in docs_job
