"""
CLI for stanhf
==============
"""

import logging
import click

from . import builder, converter, validator, __version__


@click.command()
@click.argument('hf_json_file_name', type=click.Path(exists=True))
@click.version_option(__version__, message="%(version)s")
@click.option('--overwrite/--no-overwrite', default=True,
              help="Overwrite exising files.")
@click.option('--log/--no-log', default=False,
              help="Show logging from e.g., cmdstanpy.")
def cli(hf_json_file_name, overwrite, log):
    """
    Convert, build and validate HF_JSON_FILE_NAME as a Stan model.
    """
    logger = logging.getLogger("cmdstanpy")
    logger.disabled = not log

    root = converter(hf_json_file_name, overwrite)
    print(f"- Stan model files at {root}*")

    builder(root)
    print(f"- Stan executable at {root}")

    validator(root)
    print("- Validated parameter names & target")


if __name__ == "__main__":
    cli()
