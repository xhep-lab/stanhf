"""
CLI for stanhf
==============
"""

import importlib.metadata
import os
import warnings

import click
from click_help_colors import HelpColorsCommand, version_option

from cmdstanpy import cmdstan_path

from .run import install
from .convert import Convert


VERSION = importlib.metadata.version(__package__)
CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])


def print_cmdstan_path(ctx, _, value):
    """
    Print cmdstan path
    """
    if not value or ctx.resilient_parsing:
        return
    click.echo(cmdstan_path())
    ctx.exit()


@click.command(cls=HelpColorsCommand,
               help_headers_color='yellow',
               help_options_color='green',
               context_settings=CONTEXT_SETTINGS,
               epilog="Check out https://github.com/xhep-lab/stanhf for more details or to report issues")
@click.argument('hf_file_name', type=click.Path(exists=True))
@version_option(VERSION,
                prog_name="stanhf",
                message="%(prog)s version %(version)s",
                version_color='green')
@click.option('--build/--no-build', default=True,
              help="Build Stan program.")
@click.option('--validate-par-names/--no-validate-par-names', default=True,
              help="Validate Stan program parameter names.")
@click.option('--validate-target/--no-validate-target', default=True,
              help="Validate Stan program target.")
@click.option('--cmdstan-path', is_flag=True, callback=print_cmdstan_path,
              expose_value=False, is_eager=True, help="Show path to cmdstan.")
@click.option('--patch', type=(click.Path(exists=True), click.IntRange(0)),
              default=None, nargs=2, help="Apply a patch to the model.",  metavar='<path to patchset> <number>')
def cli(hf_file_name, build, validate_par_names, validate_target, patch):
    """
    Convert, build and validate a histfactory json file HF_FILE_NAME as a Stan model.
    """
    if validate_target and not build:
        warnings.warn("Cannot validate target as not building")
        validate_target = False

    convert = Convert(hf_file_name, patch)
    click.echo(convert)

    stan_path = install()
    click.echo(f"- Stan installed at {stan_path}")

    stan_file_name, data_file_name, init_file_name = convert.write_to_disk()
    click.echo(
        f"- Stan files created at {stan_file_name}, {data_file_name} and {init_file_name}")

    if validate_par_names:
        convert.validate_par_names(stan_file_name)
        click.echo("- Validated parameter names")

    if build:
        local = os.path.join(stan_path, "build", "local")
        click.echo(f"- Build settings controlled at {local}")

        exe_file_name = convert.build(stan_file_name)
        click.echo(f"- Stan executable created at {exe_file_name}")

        cmd = f"{exe_file_name} sample num_chains=4 data file={data_file_name} init={init_file_name}"
        click.echo(f"- Try e.g., {cmd}")

        if validate_target:
            convert.validate_target(
                exe_file_name, stan_file_name, data_file_name, init_file_name)
            click.echo("- Validated target")
