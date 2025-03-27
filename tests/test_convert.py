"""
Test elements of converter from hf to Stan
==========================================
"""

import os
import re

import pytest

from stanhf import Convert


CWD = os.path.dirname(os.path.realpath(__file__))
EXAMPLE = os.path.normpath(os.path.join(CWD, "..", "examples", "test.json"))

CON = Convert(EXAMPLE)
BLOCKS = ["functions_block", "data_block",
          "transformed_data_block", "pars_block", "transformed_pars_block"]


def write_expected(block):
    """
    Write expected block from example input with C++ style comments removed
    """
    file_name = os.path.join(CWD, f"{block}.stan")
    with open(file_name, "w", encoding="utf-8") as block_file:
        block_file.write(call(block))


def strip_cpp_comments(code):
    """
    @returns Code with C++ style comments removed
    """
    def replacer(match):
        s = match.group(0)
        return " " if s.startswith('/') else s

    pattern = re.compile(
        r'//.*?$|/\*.*?\*/|\'(?:\\.|[^\\\'])*\'|"(?:\\.|[^\\"])*"',
        re.DOTALL | re.MULTILINE
    )

    return re.sub(pattern, replacer, code)


def call(block):
    """
    @returns Block found by converting input with C++ comments removed
    """
    return strip_cpp_comments(getattr(CON, block)())


@pytest.mark.parametrize("block", BLOCKS)
def test_attribute(block):
    """
    @returns Test whether conversion agrees with expected result on disk

    May differ by C++ comments
    """
    file_name = os.path.join(CWD, f"{block}.stan")
    with open(file_name, "r", encoding="utf-8") as block_file:
        expected = block_file.read()
    assert call(block) == expected


if __name__ == "__main__":
    for b in BLOCKS:
        write_expected(b)
