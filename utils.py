import numpy as np
import pandas as pd
import scipy.io
import os
import fnmatch
import torch

'''
数据集格式转换类
将10X_V2, 10X_V3, TXT, CSV格式的数据集转为H5格式
'''


class Format_Convert:
    def __init__(self, in_path, out_path, rename):
        self.in_path = in_path
        self.out_path = out_path
        self.rename = rename

    def format_identify(self):
        if os.path.isdir(self.in_path):
            # files = [os.path.join(self.in_path, f) for f in os.listdir(self.in_path)]
            # txt_exists = any(fnmatch.fnmatch(f, '*.txt') for f in os.listdir(self.in_path))
            if any(fnmatch.fnmatch(f, '*.mtx') or fnmatch.fnmatch(f, '*.tsv') for f in os.listdir(self.in_path)):
                print('*' * 100)
                print("数据集格式: 10X_V2")
                return '10X_V2'
            elif any(
                    fnmatch.fnmatch(f, '*.mtx.gz') or fnmatch.fnmatch(f, '*.tsv.gz') for f in os.listdir(self.in_path)):
                print('*' * 100)
                print("数据集格式: 10X_V3")
                return '10X_V3'
            else:
                print("此目录下没有符合要求的数据集！")
            # txt_files = [f for f in files if fnmatch.fnmatch(f, '*.mtx')]
        elif os.path.isfile(self.in_path):
            file_name, file_ext = os.path.splitext(self.in_path)
            if file_ext == '.txt':
                print('*' * 100)
                print("数据集格式: txt")
                return 'txt'
            elif file_ext == '.csv':
                print('*' * 100)
                print("数据集格式: csv")
                return 'csv'
            else:
                print('*' * 100)
                print("文件提取错误!")
                print('*' * 100)
        else:
            print('*' * 100)
            print(f'路径”{self.in_path}“不存在!')
            print('*' * 100)

    def convert(self):
        data_format = self.format_identify()
        if data_format == '10X_V2':
            if self.in_path[-1] != "/":
                self.in_path += "/"
            matrix = scipy.io.mmread(os.path.join(self.in_path, "matrix.mtx"))
            genes = list(pd.read_csv(self.in_path + "genes.tsv", header=None, sep='\t')[1])
            barcodes = list(pd.read_csv(self.in_path + "barcodes.tsv", header=None)[0])
            matrix = pd.DataFrame(np.array(matrix.todense()), index=genes, columns=barcodes)
            matrix.fillna(0, inplace=True)
            uniq_index = np.unique(matrix.index, return_index=True)[1]
            matrix = matrix.iloc[uniq_index,]
            print("去除全为零的行前的矩阵维数:", matrix.shape)
            matrix = matrix.loc[matrix.sum(axis=1) > 0, :]
            print("去除全为零的行后的矩阵维数:", matrix.shape)
            if self.out_path[-1] != "/":
                self.out_path += "/"
            if self.rename is None:
                matrix.to_hdf(self.out_path + "matrix_pred" + ".h5", key="dge", mode="w", complevel=3)
                print("处理成功，已保存到路径: ”{}“  文件名: matrix_pred.h5".format(self.out_path))
                print('*' * 100)
            else:
                if self.rename[-3:] != ".h5":
                    self.rename += ".h5"
                matrix.to_hdf(self.out_path + self.rename, key="dge", mode="w", complevel=3)
                print("处理成功，已保存到路径: ”{}“  文件名: {}".format(self.out_path, self.rename))
                print('*' * 100)

        if data_format == '10X_V3':
            if self.in_path[-1] != "/":
                self.in_path += "/"
            matrix = scipy.io.mmread(os.path.join(self.in_path, "matrix.mtx.gz"))
            genes = list(pd.read_csv(self.in_path + "genes.tsv.gz", header=None, sep='\t')[1])
            barcodes = list(pd.read_csv(self.in_path + "barcodes.tsv.gz", header=None)[0])
            matrix = pd.DataFrame(np.array(matrix.todense()), index=genes, columns=barcodes)
            matrix.fillna(0, inplace=True)
            uniq_index = np.unique(matrix.index, return_index=True)[1]
            matrix = matrix.iloc[uniq_index,]
            print("去除全为零的行前的矩阵维数:", matrix.shape)
            matrix = matrix.loc[matrix.sum(axis=1) > 0, :]
            print("去除全为零的行后的矩阵维数:", matrix.shape)
            if self.out_path[-1] != "/":
                self.out_path += "/"
            if self.rename is None:
                matrix.to_hdf(self.out_path + "matrix_pred" + ".h5", key="dge", mode="w", complevel=3)
                print("处理成功，已保存到路径: ”{}“  文件名: matrix_pred.h5".format(self.out_path))
                print('*' * 100)
            else:
                if self.rename[-3:] != ".h5":
                    self.rename += ".h5"
                matrix.to_hdf(self.out_path + self.rename, key="dge", mode="w", complevel=3)
                print("处理成功，已保存到路径: ”{}“  文件名: {}".format(self.out_path, self.rename))
                print('*' * 100)

        if data_format == 'txt':
            new_data = pd.read_csv(self.in_path, index_col=0, sep="\t")
            uniq_index = np.unique(new_data.index, return_index=True)[1]
            new_data = new_data.iloc[uniq_index,]
            print("去除全为零的行前的矩阵维数:", new_data.shape)
            new_data = new_data.loc[new_data.sum(axis=1) > 0, :]
            print("去除全为零的行后的矩阵维数:", new_data.shape)
            if self.out_path[-1] != "/":
                self.out_path += "/"
            if self.rename is None:
                file_name, file_ext = os.path.splitext(self.in_path)
                new_data.to_hdf(self.out_path + file_name + "_pred.h5", key="dge", mode="w", complevel=3)
                print("处理成功，已保存到路径: “{}”  文件名: {}_pred.h5".format(self.out_path, file_name))
                print('*' * 100)
            else:
                if self.rename[-3:] != ".h5":
                    self.rename += ".h5"
                new_data.to_hdf(self.out_path + self.rename, key="dge", mode="w", complevel=3)
                print("处理成功，已保存到路径: ”{}“  文件名: {}".format(self.out_path, self.rename))
                print('*' * 100)

        if data_format == 'csv':
            new_data = pd.read_csv(self.in_path, index_col=0)
            uniq_index = np.unique(new_data.index, return_index=True)[1]
            new_data = new_data.iloc[uniq_index,]
            print("去除全为零的行前的矩阵维数:", new_data.shape)
            new_data = new_data.loc[new_data.sum(axis=1) > 0, :]
            print("去除全为零的行后的矩阵维数:", new_data.shape)
            if self.out_path[-1] != "/":
                self.out_path += "/"
            if self.rename is None:
                file_name, file_ext = os.path.splitext(self.in_path)
                new_data.to_hdf(self.out_path + file_name + "_pred.h5", key="dge", mode="w", complevel=3)
                print("处理成功，已保存到路径: ”{}“  文件名: {}_pred.h5".format(self.out_path, file_name))
                print('*' * 100)
            else:
                if self.rename[-3:] != ".h5":
                    self.rename += ".h5"
                new_data.to_hdf(self.out_path + self.rename, key="dge", mode="w", complevel=3)
                print("处理成功，已保存到路径: ”{}“  文件名: {}".format(self.out_path, self.rename))
                print('*' * 100)


class Datasets_Process:
    @staticmethod
    def capitalize_genes_name(*datasets):
        dataset_list = list(datasets)
        for i in range(len(dataset_list)):
            dataset_list[i].index = [s.upper() for s in dataset_list[i].index]
        if len(dataset_list) == 1:
            return dataset_list[0]
        else:
            return dataset_list

    @staticmethod
    def filt_duplicate_rows(*datasets):
        datasets_list = list(datasets)
        for i in range(len(datasets_list)):
            datasets_list[i] = datasets_list[i].loc[~datasets_list[i].index.duplicated(keep='first')]
        if len(datasets_list) == 1:
            return datasets_list[0]
        else:
            return datasets_list

    @staticmethod
    def get_common_genes(*datasets):
        datasets_list = list(datasets)
        if len(datasets_list) == 1:
            return datasets_list[0].loc[~datasets_list[0].index.duplicated(keep='first')].index.tolist()
        common_genes = set(datasets_list[0].index)
        for i in range(1, len(datasets_list)):
            common_genes = set.intersection(set(datasets_list[i].index), common_genes)
        common_genes = sorted(list(common_genes))
        return common_genes

    @staticmethod
    def merge_datasets(*datasets, common_genes=None, return_original_sep=False):
        datasets_list = list(datasets)
        if common_genes is None:
            common_genes = Datasets_Process.get_common_genes(*datasets)
        sep_point = [0]
        for i in range(len(datasets_list)):
            datasets_list[i] = datasets_list[i].loc[common_genes,]
            sep_point.append(datasets_list[i].shape[1])
        merged_dataset = pd.concat(datasets_list, axis=1, sort=False)
        if return_original_sep:
            return merged_dataset, sep_point
        return merged_dataset

    @staticmethod
    def normalize(*datasets, scale_factor=10000, low_range=1, high_range=99):
        datasets_list = list(datasets)
        for i in range(len(datasets_list)):
            datasets_list[i] = np.array(datasets_list[i])
            datasets_list[i] = np.divide(datasets_list[i], np.sum(datasets_list[i], axis=0, keepdims=True)) * scale_factor
            datasets_list[i] = np.log2(datasets_list[i] + 1)
            expr = np.sum(datasets_list[i], axis=1)
            datasets_list[i] = datasets_list[i][
                np.logical_and(expr >= np.percentile(expr, low_range), expr <= np.percentile(expr, high_range)),]
            cv = np.std(datasets_list[i], axis=1) / np.mean(datasets_list[i], axis=1)
            datasets_list[i] = datasets_list[i][
                np.logical_and(cv >= np.percentile(cv, low_range), cv <= np.percentile(cv, high_range)),]
        if len(datasets_list) == 1:
            return datasets_list[0]
        return datasets_list

    @staticmethod
    def split_datasets(dataset, group_size, by_scale=False, discard_end=False):
        if group_size[0] != 0:
            group_size = [0] + group_size
        if isinstance(dataset, pd.DataFrame):
            data_len = dataset.shape[1]
            sets = []
            if ~by_scale:
                if data_len != sum(group_size):
                    print("数据集宽度和分组总宽度不匹配！")
                    print("数据集宽度: {} ,分组总宽度: {}".format(data_len, sum(group_size)))
                    print("请重新设置拆分数据!")
                    return []
                for i in range(len(group_size) - 1):
                    sets.append(dataset.iloc[:, sum(group_size[:(i + 1)]):sum(group_size[:(i + 2)])])
                return sets
            if data_len != sum(group_size):
                print("数据集宽度和分组总宽度不匹配！")
                print("数据集宽度: {} ,分组总宽度: {}".format(data_len, sum(group_size)))
                print("按照比例进行拆分中...")
                if discard_end:
                    for i in range(len(group_size)):
                        group_size[i] = int((group_size[i] / sum(group_size)) * data_len)
                    for i in range(len(group_size) - 1):
                        sets.append(dataset.iloc[:, sum(group_size[:(i + 1)]):sum(group_size[:(i + 2)])])
                        return sets
                for i in range(len(group_size)-1):
                    group_size[i] = int((group_size[i]/sum(group_size)) * data_len)
                group_size[len(group_size)-1] = data_len-sum(group_size[0:len(group_size)-2])
                for i in range(len(group_size)-1):
                    sets.append(dataset.iloc[:, sum(group_size[:(i + 1)]):sum(group_size[:(i + 2)])])
                    return sets
            for i in range(len(group_size) - 1):
                sets.append(dataset.iloc[:, sum(group_size[:(i + 1)]):sum(group_size[:(i + 2)])])
            return sets
        elif isinstance(dataset, np.ndarray):
            new_group_size = []
            for i in range(len(group_size)):
                new_group_size.append(sum(group_size[:i+1]))
            del new_group_size[0]
            del new_group_size[-1]
            array_sets_list = np.split(dataset,new_group_size, axis=1)
            return array_sets_list
        else:
            print("暂不支持拆分此格式的数据集！")


class Labels_Process:
    @staticmethod
    def merge_labels(*label, return_original_sep=False):
        labels_list = list(label)
        sep_point = [0]
        for i in range(len(labels_list)):
            sep_point.append(labels_list[i].shape[0])
        merged_label = pd.concat(labels_list, axis=0, sort=False)
        if return_original_sep:
            return merged_label, sep_point
        return merged_label

    @staticmethod
    def split_labels(merged_label, group_size):
        if group_size[0] != 0:
            group_size = [0] + group_size
        merged_label_len = merged_label.shape[0]
        if merged_label_len != sum(group_size):
            print("标签数量和分组标签数量不匹配！")
            print("标签数量: {} ,分组标签总数: {}".format(merged_label_len, sum(group_size)))
            print("请重新设置拆分数据!")
            return []
        labels = []
        for i in range(len(group_size) - 1):
            labels.append(merged_label[sum(group_size[:(i + 1)]):sum(group_size[:(i + 2)])])
        return labels

    @staticmethod
    def type_to_label_dict(cell_type):
        cell_type = cell_type.iloc[:, 1]
        all_type = list(sorted(set(cell_type)))
        type_to_label_dict = {}
        for i in range(len(all_type)):
            type_to_label_dict[all_type[i]] = i
        return type_to_label_dict

    @staticmethod
    def label_to_type_dict(type_to_label_dict):
        label_to_type_dict = {v: k for k, v in type_to_label_dict.items()}
        return label_to_type_dict

    @staticmethod
    def convert_type_to_label(cell_type, type_to_label_dict, return_labels_only=True):
        cell_type_columns = cell_type.iloc[:, 1]
        cell_types = list(cell_type_columns)
        labels = []
        for types in cell_types:
            labels.append(type_to_label_dict[types])
        if return_labels_only:
            return labels
        else:
            cell_type[2] = pd.Series(labels)
            return cell_type

    @staticmethod
    def one_hot_matrix(labels):
        num_class = len(set(labels))
        one_hot_matrix = torch.nn.functional.one_hot(torch.tensor(labels), num_class).float().numpy()
        return one_hot_matrix, num_class


# 功能:将数据集和标签是否一一对应，防止数据集和标签不配对造成错误
# 输入:数据集的dataframe对象，标签dataframe对象，check_common_cell:按照提取共同细胞的方式配对，顺序会打乱
def dataset_label_match(dataset, label, check_common_cell=False):
    if not check_common_cell:
        cell_num = dataset.shape[1]
        label_num = label.shape[0]
        dataset = dataset.loc[:, ~dataset.columns.duplicated()]
        label = label.loc[~label[0].duplicated(),]
        if cell_num > label_num:
            dataset = dataset.loc[:, label[0]]
            return dataset, label
        else:
            label = label.set_index(label.iloc[:, 0], inplace=False)
            label = label.loc[dataset.columns.tolist(),]
            label = label.reset_index(drop=True)
            return dataset, label
    else:
        common_cells = sorted(list(set.intersection(set(dataset.columns.tolist()), set(label[0]))))
        dataset = dataset.loc[:, common_cells]
        label = label.set_index(label.iloc[:, 0], inplace=False)
        label = label.loc[common_cells,]
        label = label.reset_index(drop=True)
        return dataset, label
