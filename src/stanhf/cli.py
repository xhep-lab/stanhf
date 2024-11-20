"""
CLI for stanhf
==============
"""

import importlib.metadata

import click
from cmdstanpy import cmdstan_path

from . import install, build, convert, validate


VERSION = importlib.metadata.version(__package__)


def print_cmdstan_path(ctx, param, value):
    """
    Print cmdstan path
    """
    if not value or ctx.resilient_parsing:
        return
    click.echo(cmdstan_path())
    ctx.exit()


@click.command()
@click.argument('hf_json_file_name', type=click.Path(exists=True))
@click.version_option(VERSION, message="%(version)s")
@click.option('--overwrite/--no-overwrite', default=True,
              help="Overwrite exising files.")
@click.option('--cmdstan-path', is_flag=True, callback=print_cmdstan_path, expose_value=False, is_eager=True)
def cli(hf_json_file_name, overwrite):
    """
    Convert, build and validate HF_JSON_FILE_NAME as a Stan model.
    """
    stan_path = install()
    print(f"- Stan installed at {stan_path}")

    root = convert(hf_json_file_name, overwrite)
    print(f"- Stan model files created at {root}*")

    build(root)
    print(f"- Stan executable created at {root}")

    validate(root)
    print("- Validated parameter names & target")

    cmd = f"{root} sample num_chains=4 data file={root}_data.json init={root}_init.json"
    print(f"- Try e.g., {cmd}")
