from argparse import ArgumentParser

import yaml


def args_provider(filename: str, parser: ArgumentParser) -> ArgumentParser:
    with open(filename, 'r') as f:
        args_dict = yaml.safe_load(f)
    args_dict = {key.replace('-', '_'): val for key, val in args_dict.items()}
    parser.set_defaults(**args_dict)
    return parser
