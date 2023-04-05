import argparse


# 定义训练命令行参数
def get_train_parser(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser(description="训练参数")
    parser.add_argument("--pre_model_path", "-pmp", type=str, help="预训练模型路径", default='./checkpoints/0.pth')
    parser.add_argument("--train_set", "-trs", type=str, help="训练集文件路径", default='./data/')
    parser.add_argument("--train_label", "-trl", type=str, help="训练集标签文件路径", default='./data/')
    parser.add_argument("--epoch", "-ep", type=int, help="训练轮数 (默认: 50)", default=50)
    parser.add_argument("--batch_size", "-bs", type=int, help="批量大小 (默认: 64)", default=64)
    parser.add_argument("--model_sava_path", "-msp", type=str, help="模型保存位置", default='./checkpoints')
    parser.add_argument("--device", "-dv", type=str, help="训练设备——GPU或CPU (默认: GPU)", default='GPU')
    parser.add_argument("--learning_rate", "-lr", type=float, help="学习率 (默认: 0.0001)", default=0.0001)
    parser.add_argument("--print_loss", "-pl", type=bool, help="是否打印损失值 (默认: True)", default=True)
    parser.add_argument("--save_logs", "-sl", type=bool, help="是否保存训练日志 (默认：True)", default=True)
    return parser


# 定义测试命令行参数
def get_test_parser(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser(description="测试参数")
    parser.add_argument("--test_model_path", "-tmp", type=str, help="用来测试的模型的路径", default='./checkpoints/999.pth')
    parser.add_argument("--test_set", "-ts", type=str, help="测试数据集路径", default='./data/')
    parser.add_argument("--test_label", "-tl", type=str, help="测试标签路径", default='./data/')
    parser.add_argument("--print_accuracy", "-pa", type=bool, help="是否打印测试准确率 (默认: True)", default=True)
    parser.add_argument("--result_sava_path", "-rsp", type=str, help="测试结果保存位置", default='./results/')
    return parser


# 定义预测命令行参数
def get_predict_parser(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser(description="预测参数")
    parser.add_argument("--model_path", "-mp", type=str, help="用于预测的模型的路径", default='./checkpoints/999.pth')
    parser.add_argument("--predict_set", "-ps", type=str, help="用来预测的数据集路径", default='./data/')
    parser.add_argument("--result_sava_path", "-rsp", type=str, help="预测结果保存位置")
    return parser


# 数据集格式转换命令行参数
def format_convert_parser(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser(description="数据集格式转换参数")
    parser.add_argument("--in_path", "-i", type=str, help="数据集输入路径", default='./checkpoints/999.pth')
    parser.add_argument("--out_path", "-o", type=str, help="数据集输出路径, 默认为和原文件同目录")
    parser.add_argument("--rename", type=str, help="文件重命名, 默认: 原文件名_pred.h5")
    return parser
