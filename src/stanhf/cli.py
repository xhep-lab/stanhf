"""
CLI for stanhf
==============
"""

import importlib.metadata
import os

import click
from cmdstanpy import cmdstan_path

from .run import install, build as stan_build, validate as stan_validate
from .convert import convert


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
@click.option('--build/--no-build', default=True,
              help="Build Stan program.")
@click.option('--validate/--no-validate', default=True,
              help="Validate Stan program.")
@click.option('--cmdstan-path', is_flag=True, callback=print_cmdstan_path,
              expose_value=False, is_eager=True)
def cli(hf_json_file_name, build, validate):
    """
    Convert, build and validate HF_JSON_FILE_NAME as a Stan model.
    """
    root, convert_ = convert(hf_json_file_name)
    click.echo(f"- Stan model files created at {root}*")
    click.echo(convert_)

    if build:

        stan_path = install()
        click.echo(f"- Stan installed at {stan_path}")

        local = os.path.join(stan_path, "build", "local")
        click.echo(f"- Build settings controlled at {local}")

        stan_build(root)
        click.echo(f"- Stan executable created at {root}")

        cmd = f"{root} sample num_chains=4 data file={root}_data.json init={root}_init.json"
        click.echo(f"- Try e.g., {cmd}")

    if validate:

        stan_validate(root, convert_)
        click.echo("- Validated parameter names & target")
