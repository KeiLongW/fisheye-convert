import argparse


def parse_args():
  arg_parser = argparse.ArgumentParser()
  arg_parser.add_argument("--input_path", type=str)
  arg_parser.add_argument("--output_path", type=str)
  arg_parser.add_argument("--label_dir_prefix", type=str, default="label")
  arg_parser.add_argument("--image_dir_prefix", type=str, default="image")
  arg_parser.add_argument("--pencil_dilatation_size", type=int, default=2)
  return arg_parser.parse_args()