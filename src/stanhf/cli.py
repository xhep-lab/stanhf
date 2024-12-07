"""
CLI for stanhf
==============
"""

import importlib.metadata
import os

import click
from cmdstanpy import cmdstan_path

from .run import install, build as stan_build, validate
from .convert import convert, par_names


VERSION = importlib.metadata.version(__package__)


def print_cmdstan_path(ctx, _, value):
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
              help="Overwrite existing files.")
@click.option('--compile/--no-compile', default=True,
              help="Compile Stan program.")
@click.option('--cmdstan-path', is_flag=True, callback=print_cmdstan_path,
              expose_value=False, is_eager=True)
def cli(hf_json_file_name, overwrite, build):
    """
    Convert, build and validate HF_JSON_FILE_NAME as a Stan model.
    """
    root = convert(hf_json_file_name, overwrite)
    print(f"- Stan model files created at {root}*")

    par, fixed, null = par_names(hf_json_file_name)
    print(f"- Identified {len(par)} parameters,"
          f" {len(fixed)} fixed parameters and"
          f" {len(null)} null parameters")

    if not build:
        return

    stan_path = install()
    print(f"- Stan installed at {stan_path}")

    local = os.path.join(stan_path, "build", "local")
    print(f"- Build settings controlled at {local}")

    stan_build(root)
    print(f"- Stan executable created at {root}")

    validate(root, par, fixed, null)
    print("- Validated parameter names & target")

    cmd = f"{root} sample num_chains=4 data file={root}_data.json init={root}_init.json"
    print(f"- Try e.g., {cmd}")
