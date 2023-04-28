# import anndata as ad
import numpy as np
import scanpy as sc
from scipy import sparse

panglao = sc.read_h5ad("G:/dataset/PanglaoDB/panglao_10000.h5ad")
data = sc.read_h5ad("G:/dataset/PanglaoDB/Zheng68K.h5ad")
print("panglao shape:", panglao.shape)
print("panglao:",panglao)
print("panglao.X shape:",panglao.X.shape)
print("panglao.X:",panglao.X)
print("*"*100)
print("data shape:", data.shape)
print("data:",data)
print("data.X shape:",data.X.shape)
print("data.X:",data.X)
print("*"*100)

counts = sparse.lil_matrix((data.X.shape[0],panglao.X.shape[1]),dtype=np.float32)
print("counts shape:", counts.shape)
print("counts:", counts)

ref = panglao.var_names.tolist()
obj = data.var_names.tolist()
print("ref len:", len(ref))
print("ref:", ref)
print("obj len",len(obj))
print("obj:", obj)


# for i in range(len(ref)):
#     if ref[i] in obj:
#         loc = obj.index(ref[i])
#         counts[:,i] = data.X[:,loc]
# print(counts.shape)
# print(counts)
# print("&"*100)
#
# counts = counts.tocsr()
# print(counts.shape)
# print(counts)
# print("&"*100)
#
# new = ad.AnnData(X=counts)
# print(new.shape)
# print(new)
# print("-"*100)
#
# new.var_names = ref
# new.obs_names = data.obs_names
# new.obs = data.obs
# new.uns = panglao.uns
# print(new.shape)
# print(new)
# print("@"*100)
#
# sc.pp.filter_cells(new, min_genes=200)
# sc.pp.normalize_total(new, target_sum=1e4)
# sc.pp.log1p(new, base=2)
# print(new.shape)
# print(new)
# print("@"*100)
#
# new.write("G:/dataset/PanglaoDB/preprocessed_data.h5ad")
# print("保存成功！")

