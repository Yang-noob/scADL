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
    def __init__(self, *datasets):
        self.common_genes = None
        self.datasets = list(datasets)

    def capitalize_genes_name(self):
        for i in range(len(self.datasets)):
            self.datasets[i].index = [s.upper() for s in self.datasets[i].index]
        return self.datasets

    def filt_duplicate_rows(self):
        for i in range(len(self.datasets)):
            self.datasets[i] = self.datasets[i].loc[~self.datasets[i].index.duplicated(keep='first')]
        return self.datasets

    def get_common_genes(self):
        self.common_genes = set(self.datasets[0].index)
        for i in range(1, len(self.datasets)):
            self.common_genes = set.intersection(set(self.datasets[i].index), self.common_genes)
        self.common_genes = sorted(list(self.common_genes))
        return self.common_genes

    def merge_datasets(self, genes):
        # if genes != self.get_common_genes():
        #     print("genes != common_genes")
        #     genes = self.get_common_genes()
        sep_point = [0]
        for i in range(len(self.datasets)):
            self.datasets[i] = self.datasets[i].loc[genes,]
            sep_point.append(self.datasets[i].shape[1])
        merged_dataset = np.array(pd.concat(self.datasets, axis=1, sort=False), dtype=np.float32)
        return merged_dataset, sep_point

    def normalize(self, merged_datasets):
        merged_datasets = np.array(merged_datasets)
        normalized_datasets = np.divide(merged_datasets, np.sum(merged_datasets, axis=0, keepdims=True)) * 10000
        normalized_datasets = np.log2(normalized_datasets + 1)
        expr = np.sum(normalized_datasets, axis=1)
        normalized_datasets = normalized_datasets[
            np.logical_and(expr >= np.percentile(expr, 1), expr <= np.percentile(expr, 99)),]
        cv = np.std(normalized_datasets, axis=1) / np.mean(normalized_datasets, axis=1)
        normalized_datasets = normalized_datasets[
            np.logical_and(cv >= np.percentile(cv, 1), cv <= np.percentile(cv, 99)),]
        return normalized_datasets

    def split_datasets(self, dataset, group_size):
        if group_size[0] != 0:
            group_size = [0] + group_size
        data_len = dataset.shape[1]
        if data_len != sum(group_size):
            print("数据集宽度和分组宽度不匹配！")
            print("数据集宽度: {} ,分组总宽度: {}".format(data_len, sum(group_size)))
            print("请重新设置拆分数据!")
            return []
        sets = []
        for i in range(len(group_size) - 1):
            sets.append(dataset[:, sum(group_size[:(i + 1)]):sum(group_size[:(i + 2)])])
        return sets


class Labels_Process:
    def __init__(self):
        self.a = 0

    def type_to_label_dict(self, cell_type):
        cell_type = cell_type.iloc[:, 1]
        all_type = list(sorted(set(cell_type)))
        type_to_label_dict = {}
        for i in range(len(all_type)):
            type_to_label_dict[all_type[i]] = i
        return type_to_label_dict

    def label_to_type_dict(self, type_to_label_dict):
        label_to_type_dict = {v: k for k, v in type_to_label_dict.items()}
        return label_to_type_dict

    def convert_type_to_label(self, cell_type, type_to_label_dict):
        cell_type_columns = cell_type.iloc[:, 1]
        cell_types = list(cell_type_columns)
        labels = []
        for types in cell_types:
            labels.append(type_to_label_dict[types])
        cell_type[2] = pd.Series(labels)
        return cell_type

    def one_hot_matrix(self, labels):
        num_class = len(set(labels))
        one_hot_matrix = torch.nn.functional.one_hot(torch.tensor(labels), num_class).float().numpy()
        return one_hot_matrix, num_class


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
            # label.reset_index(inplace=True)
            # label.drop(0, axis=1, inplace=True)
            label = label.reset_index(drop=True)
            return dataset, label
    else:
        common_cells = sorted(list(set.intersection(set(dataset.columns.tolist()), set(label[0]))))
        dataset = dataset.loc[:, common_cells]
        label = label.set_index(label.iloc[:, 0], inplace=False)
        label = label.loc[common_cells, ]
        label = label.reset_index(drop=True)
        return dataset, label
