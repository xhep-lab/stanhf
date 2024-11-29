"""
Test elements of converter from hf to Stan
==========================================
"""

import os

from stanhf import Convert


CWD = os.path.dirname(os.path.realpath(__file__))
EXAMPLE = os.path.join(CWD, "..", "examples", "example.json")

con = Convert(EXAMPLE)


def test_functions():
    con.functions_block()

def test_metadata():
    con.metadata()
    
def test_data():
    con.data_block()

def test_transformed_data():
    con.transformed_data_block()

def test_pars():
    con.pars_block()

def test_transformed_pars():
    con.transformed_pars_block()

def test_model():
    con.model_block()

def test_generated_quantities():
    con.generated_quantities_block()

def test_data_card():
    con.data_card()

def test_init_card():
    con.init_card()
