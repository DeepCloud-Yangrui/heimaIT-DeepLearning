"""
案例
    演示张量常用的运算函数
需要记的
    sum() max() min() mean()
有dim参数的：sum() max() min() mean()        dim=0表示对列操作，dim=1表示对行操作
"""
import torch

# 1. 定义张量
t1 = torch.tensor([
    [1, 2, 3],
    [4, 5, 6]
])
# 2. 演示有dim参数的函数
print(t1.sum(dim=0))  # 按列求和
print(t1.sum(dim=1))  # 按行求和
print(t1.sum())  # 整体求和，所有元素相加
# 3. 剩余三个函数，同理就不演示了