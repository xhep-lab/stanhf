"""
CLI for stanhf
==============
"""

import importlib.metadata
import os

import click
from cmdstanpy import cmdstan_path

from .run import install
from .convert import Convert


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
@click.argument('hf_file_name', type=click.Path(exists=True))
@click.version_option(VERSION, message="%(version)s")
@click.option('--build/--no-build', default=True,
              help="Build Stan program.")
@click.option('--validate/--no-validate', default=True,
              help="Validate Stan program.")
@click.option('--cmdstan-path', is_flag=True, callback=print_cmdstan_path,
              expose_value=False, is_eager=True)
@click.option('--patch', default=None, nargs=2, type=(click.Path(exists=True), click.IntRange(0)))
def cli(hf_file_name, build, validate, patch):
    """
    Convert, build and validate HF_JSON_FILE_NAME as a Stan model.
    """
    convert = Convert(hf_file_name, patch)
    click.echo(convert)

    stan_file_name, data_file_name, init_file_name = convert.write_to_disk()
    click.echo(
        f"- Stan files created at {stan_file_name}, {data_file_name} and {init_file_name}")

    if build:
        stan_path = install()
        click.echo(f"- Stan installed at {stan_path}")

        local = os.path.join(stan_path, "build", "local")
        click.echo(f"- Build settings controlled at {local}")

        exe_file_name = convert.build(stan_file_name)
        click.echo(f"- Stan executable created at {exe_file_name}")

        cmd = f"{exe_file_name} sample num_chains=4 data file={data_file_name} init={init_file_name}"
        click.echo(f"- Try e.g., {cmd}")
    else:
        exe_file_name = None

    if validate:
        convert.validate_par_names(stan_file_name)
        click.echo("- Validated parameter names")

        convert.validate_target(exe_file_name, stan_file_name, data_file_name, init_file_name)
        click.echo("- Validated target")
