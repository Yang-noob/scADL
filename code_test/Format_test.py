import sys

sys.path.append("..")

import utils
from options import format_convert_parser

# par = argparse.ArgumentParser()
# par.add_argument("--in_path", "-i", type=str, help="数据集输入路径", default='./checkpoints/999.pth')
# par.add_argument("--out_path", "-o", type=str, help="数据集输出路径, 默认为和原文件同目录")
# par.add_argument("--rename", type=str, help="文件重命名, 默认: 原文件名_pred.h5")
# arg = par.parse_args()
par = format_convert_parser()
arg = par.parse_args()

in_path = arg.in_path
out_path = arg.out_path
rename = arg.rename

obj = utils.Format_Convert(in_path, out_path, rename)
obj.convert()
