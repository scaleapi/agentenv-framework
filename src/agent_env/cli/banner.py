"""Shared ASCII banner for CLI commands."""

import shutil

import click

BANNER = r"""
                            _
  __ _   __ _   ___  _ __  | |_    ___  _ __ __   __
 / _` | / _` | / _ \| '_ \ | __|  / _ \| '_ \\ \ / /
| (_| || (_| ||  __/| | | || |_  |  __/| | | |\ V /
 \__,_| \__, | \___||_| |_| \__|  \___||_| |_| \_/
        |___/
""".strip("\n")


def print_banner():
    lines = BANNER.splitlines()
    banner_width = max(len(line) for line in lines)
    term_width = shutil.get_terminal_size().columns
    pad = max((term_width - banner_width) // 2, 0)
    prefix = " " * pad
    for line in lines:
        click.echo(f"{prefix}{line}")
    click.echo()
