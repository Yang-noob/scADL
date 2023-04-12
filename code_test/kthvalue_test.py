import numpy as np
import torch


data = np.random.rand(20000)
tensor = torch.from_numpy(data).float()

cv_low = torch.kthvalue(tensor, int(len(tensor) * 1 / 100)).values
cv_high = torch.kthvalue(tensor, int(len(tensor) * 99 / 100)).values
cv_mask = (tensor >= cv_low) & (tensor <= cv_high)

print(cv_low)
print(cv_high)
print(cv_mask)