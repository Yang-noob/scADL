import sys

sys.path.append("..")

import options

args = options.parse_common_args()
print("arg1:", args.arg1)
print("arg2:", args.arg2)
print("arg3:", args.arg3)
