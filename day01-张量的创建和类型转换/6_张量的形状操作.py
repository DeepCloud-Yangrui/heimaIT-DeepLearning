"""
案例：
    演示张量的形状操作
涉及到的API：
    reshape()：      在不改变张量内容的前提下，对其形状做改变
    unsqueeeze()     在指定的轴上增加一个（1）维度，等价于升维
    squeeze         删除所有为1 的维度，等价于降维
    transpose       一次只能交换两个维度
    permute         一次可以同时交换多个维度
    view            只能修改连续的张量的形状，   连续张量 = 内存中的存储顺序 和 在张量中显示的顺序相同
    contiguous      把不连续的张量 -> 连续的张量，即：基于张量中显示的顺序，修改内存中的存储顺序
    is_contiguous   判断张量是否是连续的
需要掌握的函数
    reshape
    unsqueeze
    permute
    view
"""
# 导包
import torch

# 指定随机种子
torch.manual_seed(24)


# 1. 定义函数，演示reshape函数
def dm01():
    # 1. 定义一个2行3列的张量
    t1 = torch.randint(1, 10, (2, 3))
    print(f't1:{t1},shape:{t1.shape}')
    # 2. 通过reshape函数把t1转成->3行2列，1行6列，6行1列
    # t2=t1.reshape(3,2)
    t2 = t1.reshape(1, 6)
    print(f't1:{t2},shape:{t2.shape}')

    # 3. 尝试通过reshape把t1转成2行5列的结果
    t3 = t1.reshape(2, 5)  # 报错，转之前6个元素，转之后10个元素，不一致
    print(f't3:{t2},shape:{t3.shape}')


# 2. 定义函数，演示unsqueeze、squeeze函数
def dm02():
    # 1. 创建张量
    t1 = torch.randint(1, 10, (2, 3))
    print(f't1:{t1},shape:{t1.shape}')
    # 2. 在0维上添加一个维度
    t2 = t1.unsqueeze(0)
    print(f't2:{t2},shape:{t2.shape}')  # (1,2,3)
    # 3. 在1维上添加一个维度
    t3 = t1.unsqueeze(1)
    print(f't3:{t3},shape:{t3.shape}')  # (2,1,3)
    # 4. 在2维上添加一个维度
    t4 = t1.unsqueeze(2)
    print(f't4:{t4},shape:{t4.shape}')  # (2,3,1)
    # 5. 在3维上添加一个维度
    # t5=t1.unsqueeze(3)                      # 报错，越界
    # print(f't5:{t5},shape:{t5.shape}')      # (2,3,*,1)

    # 6. 删除所有为1的维度
    t6 = torch.randint(1, 10, (2, 1, 3, 1, 1))
    print(f't6:{t6},shape:{t6.shape}')

    t7 = t6.squeeze()
    print(f't7:{t7},shape:{t7.shape}')
    print('-' * 30)

# 3. 定义函数，演示transpose、permute函数
def dm03():
    # 1. 定义张量
    t1=torch.randint(1,10,(2,3,4))
    print(f't1:{t1},shape:{t1.shape}')
    # 2. 改变维度，从(2,3,4)->(3,2,4)
    t2=t1.transpose(0,1)
    print(f't2:{t2},shape:{t2.shape}')

    # 3. 改变维度，从(2,3,4)->(4,2,3)
    t3=t1.permute(2,0,1)
    print(f't3:{t3},shape:{t3.shape}')



# 4. 定义函数，演示view、contiguous、is_contiguous函数
def dm04():
    # 思路：演示view无法改变不连续张量的形状，可以通过is_contiguous()判断张量是否连续,也可以通过contiguous()函数将不连续张量转换为连续张量
    # 1. 定义张量
    t1=torch.randint(1,10,(2,3))
    print(f't1:{t1},shape:{t1.shape}')
    # 2. 判断张量是否连续,即张量中显示的顺序和内存中的存储顺序是否是一致的
    print(t1.is_contiguous())
    # 3. 通过view函数修改上述张量的形状,(2,3) -> (3,2)
    t2 = t1.view(3, 2)
    print(t2.is_contiguous())       # True
    # 4. 通过transpose交换维度 -> 交换之后，不连续了
    t3=t1.transpose(0,1)
    print(f't3:{t3},shape:{t3.shape}')
    print(t3.is_contiguous())
    # 5. 尝试把t3张量从（3，2）通过view转成（2，3）
    # t4=t3.view(2,3)         # 报错，因为t3不连续
    # 6. 通过contiguous函数，把t3张量变成连续的
    t5=t3.contiguous().view(2,3)
    print(f't5:{t5},shape:{t5.shape}')


# 5. 测试
if __name__ == '__main__':
    # dm01()
    # dm02()
    # dm03()
    dm04()