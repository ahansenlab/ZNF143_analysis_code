#!/usr/bin/env python
"""
__main__.py -- launch batch analyses from the command line
"""
import click
from .ImageViewer import launch

@click.group()
def cli():
    pass

@cli.command()
@click.argument('path', type=str)
def main(path):
    """
    launches main gui
    """
    launch(path)

if __name__ == '__main__':
    cli()
