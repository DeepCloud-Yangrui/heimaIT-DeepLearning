"""
案例：
    1、演示tensor和numpy之间如何相互转换
    2、演示如何从标量张量中提取其内容

涉及到的API：
    场景1：张量 -> numpy  nd数组对象
        张量对象.numpy()        共享内存    浅拷贝
        张量对象.numpy().copy   不共享内存  深拷贝
    场景2：numpy -> 张量  张量.
        from numpy()            共享内存
        torch.tensor(numpy)     不共享内存
    场景3：从标量张量中提取其内容
        标量张量.item()

掌握：(实际上只需要掌握三个就够了)
    张量 -> numpy     张量对象.numpy()
    numpy -> 张量：    torch.tensor(nd数组)
    从标量张量中提取内容  标量张量.item()
"""
import torch
import numpy as np


# 1. 定义函数，演示张量 -> numpy
def dm01():
    # 1. 创建张量
    t1 = torch.tensor([1, 2, 3, 4, 5])
    # 2. 张量 -> numpy
    # n1 = t1.numpy()         # 共享内存
    n1 = t1.numpy().copy()
    print(f't1:{t1},tpye:{type(t1)}')  # t1:tensor([1, 2, 3, 4, 5]),tpye:<class 'torch.Tensor'>

    print(f'n1:{n1},tpye:{type(n1)}')  # n1:[1 2 3 4 5],tpye:<class 'numpy.ndarray'>
    # 3. 演示上述方式是否共享内存
    n1[0] = 100
    print(t1)
    print(n1)


# 2. 定义函数，演示numpy -> 张量
def dm02():
    # 1. 创建numpy
    n1 = np.array([11, 22, 33])
    print(f'n1:{n1},type:{type(n1)}')
    # 2. 把上述的ndarray数组转成tensor
    t1 = torch.tensor(n1)
    print(f't1:{t1},type:{type(t1)}')
    t2 = torch.from_numpy(n1)  # 共享内存
    # 3. 演示上述方式是否共享内存
    n1[0] = 111
    print(t1)  # 不共享内存
    print(t2)  # 共享内存


# 3. 定义函数，演示从张量中提取内容
def dm03():
    # 1. 创建张量
    t1=torch.tensor(100)            # 可以
    #t1=torch.tensor([100,])        # 可以
    #t1=torch.tensor([100,200])     # 不可以
    # 2. 从张量中提取内容
    item = t1.item()
    print(item)


# 4. 测试
if __name__ == '__main__':
    # dm01()
    # dm02()
    dm03()