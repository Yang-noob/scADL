import argparse

parser = argparse.ArgumentParser(description="My script description")

parser.add_argument("--arg1", type=str, required=True, help="arg1 help")
parser.add_argument("--arg2", type=int, default=0, help="arg2 help")
parser.add_argument("--arg3", type=float, default=0.0, help="arg3 help")

args = parser.parse_args()
