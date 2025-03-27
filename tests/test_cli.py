"""
Test stanhf CLI
===============
"""

import os

from click.testing import CliRunner
from stanhf.cli import cli


CWD = os.path.dirname(os.path.realpath(__file__))
EXAMPLE = os.path.normpath(os.path.join(CWD, "..", "examples", "test.json"))


def test_cli():
  runner = CliRunner()
  result = runner.invoke(cli, [EXAMPLE])
  assert result.exit_code == 0