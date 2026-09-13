"""The bootstrap CLI must remain usable without the optional ML environment."""

from pathlib import Path
import subprocess
import sys


def test_help_does_not_import_optional_ml_packages() -> None:
    # Block optional imports even on developer machines with the ML extra.
    # A subprocess avoids changing import state for other tests.
    script = """
import importlib.abc
import runpy
import sys

class BlockOptionalML(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'PIL', 'torch', 'ultralytics', 'onnx', 'onnxruntime'}:
            raise AssertionError(f'CLI help imported optional package: {fullname}')
        return None

sys.meta_path.insert(0, BlockOptionalML())
sys.argv = ['bootstrap_candidate', '--help']
runpy.run_module('tools.perception.bootstrap_candidate', run_name='__main__')
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "--training-scene" in result.stdout
    assert "--checkpoint-sha256" in result.stdout
