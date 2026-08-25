"""
案例：
    演示张量的拼接操作
涉及到的API
    cat         不改变维度数，拼接张量，除了拼接的那个维度外，其他维度数必须保持一致
    stack       改变增加维度数，拼接张量，所有的维度都必须保持一致，

"""
# 导包
import torch


# 设置随机种子
torch.manual_seed(10)


# 1. 创建两个张量
t1=torch.randint(1,10, (2, 3))
print(t1)
t2=torch.randint(1,10, (2, 3))
print(t2)

# 2. 演示张量拼接
# 2.1 cat函数拼接张量
# 按照行dim=0来拼接

t3=torch.cat((t1,t2), dim=0)
print(t3)
# 按照列dim=1来拼接
t4=torch.cat((t1,t2), dim=1)
print(t4)

# 按照不存在的维度dim=2来拼接
# t5=torch.cat((t1,t2), dim=2)        # 报错
# print(f't5:{t5}')
print('-'*30)


# 2.2 stack函数拼接张量   可以是新维度，但是无论新旧维度，所有维度都必须保持一致

# 按照新维度dim=0来拼接
t6=torch.stack((t1,t2), dim=0)
print(t6)
# 按照新维度dim=1来拼接
t7=torch.stack((t1,t2), dim=1)
print(t7)
# 按照新维度dim=2来拼接
t8=torch.stack((t1,t2), dim=2)
print(t8)