import argparse


def parse_common_args():
    Parser = argparse.ArgumentParser(description="My script description")
    Parser.add_argument('--model_path', type=str, default='/checkpoints/base_model_pref/0.pth', help='已有模型加载路径')
    Parser.add_argument('--data_path', type=str, default='/data', help='数据集路径')
    Parser.add_argument('--epochs', type=int, default='200', help='训练轮数')
    args = Parser.parse_args()
    return args
