"""
案例
    演示张量的点乘、矩阵乘法
点乘：
    要求：两个张量的维度保持一致，对应元素直接做相应的操作
    API：
        t1 * t2
        t1.mul(t2)
矩阵乘法：
    要求: 两个张量，第一个张量的列数等于第二个张量的行数（A列 = B行）
    结果：A行B列
    API：
        t1 @ t2
        t1 matmul(t2)
"""
# 导包
import torch


# 1. 定义函数，演示张量点乘
def dm01():
    t1 = torch.tensor([[1, 2], [3, 4]])
    t2 = torch.tensor([[5, 6], [7, 8]])
    # t3=t1*t2
    t3 = torch.mul(t1, t2)
    print(t3)


# 2. 定义函数，演示张量矩阵乘法
def dm02():
    t1 = torch.tensor([[1, 2, 3]])
    t2 = torch.tensor([[5, 6], [7, 8], [9, 10]])
    # t3=t1@t2
    t3 = t1.matmul(t2)
    print(t3)


# 3. 测试
if __name__ == '__main__':
    # dm01()
    dm02()
