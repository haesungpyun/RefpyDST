import os
import torch

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
print(torch.cuda.current_device())  # 0