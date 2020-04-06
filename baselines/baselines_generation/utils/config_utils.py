"""Config utility functions."""

import argparse
import configparser

def recieve_cmd_config(config_dict):
    parser = argparse.ArgumentParser()
    parser.add_argument('-baseline', dest='baseline')
    args = vars(parser.parse_args())

    config_dict.update({k: v for k, v in args.items() if v is not None})  # Update if v is not None
