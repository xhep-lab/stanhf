"""
Test elements of converter from hf to Stan
==========================================
"""

import os

import pytest

from stanhf import Convert


CWD = os.path.dirname(os.path.realpath(__file__))
EXAMPLE = os.path.normpath(os.path.join(CWD, "..", "examples", "example.json"))

CON = Convert(EXAMPLE)
BLOCKS = ["functions_block", "metadata", "data_block",
          "transformed_data_block", "pars_block", "transformed_pars_block"]


def write_expected(block):
    """
    Write expected block from example input
    """
    file_name = os.path.join(CWD, f"{block}.stan")
    with open(file_name, "w", encoding="utf-8") as block_file:
        block_file.write(call(block))


def call(block):
    """
    @returns Block found by converting input with commented lines removed
    """
    out = getattr(CON, block)()
    return "\n".join(line for line in out.split("\n") if not line.startswith("//"))


@pytest.mark.parametrize("block", BLOCKS)
def test_attribute(block):
    """
    @returns Test whether conversion agrees with expected result on disk
    """
    file_name = os.path.join(CWD, f"{block}.stan")
    with open(file_name, "r", encoding="utf-8") as block_file:
        expected = block_file.read()
    assert call(block) == expected


if __name__ == "__main__":
    for b in BLOCKS:
        write_expected(b)
