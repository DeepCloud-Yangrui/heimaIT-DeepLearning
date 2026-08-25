"""
案例
    演示自动微分模块 循环实现 计算梯度 更新参数

需求
    #求y=x**2+20 的极小值点并打印y是最小值时 w的值(梯度)
    #1 定义点 x=10 requires_grad=True dtype-torch. float32
    #2定义函数y=x**2+20
    #3利用梯度下降法循环迭代1000 求最优解
    #3-1正向计算(前向传播)
    #3-2梯度清零x.grad.zero_
    #3-3反向传播
    #3-4 梯度更新 x.data = x.data - 0.01 * x.grad
"""
# 导包
import torch

# 1 定义点 x=10 requires_grad=True dtype-torch.float32  这里x就是w

# 参1：初始值、初始权重   参2：自动微分     参3：数据类型，浮点型
w = torch.tensor(10, requires_grad=True, dtype=torch.float32)

# 2定义函数y=x**2+20    这里y就是loss
loss = w ** 2 + 20  # 求导 loss’=2w
print(w.grad)
# 3利用梯度下降法循环迭代100次 求最优解
print(f'初始权重:{w},初始损失:{loss}')

print('-' * 30)
# 迭代100次求最优解
for i in range(1, 101):
    # 3-1正向计算(前向传播)
    loss = w ** 2 + 20
    # 3-2梯度清零x.grad.zero_   默认梯度会累加
    # 至此第一次的时候还没有计算梯度，所以w.grad为None,要做非空判断
    if w.grad != None:
        w.grad.zero_()

    # 3-3反向传播
    loss.sum().backward()

    # 3-4 梯度更新 x.data = x.data - 0.01 * x.grad
    w.data = w.data - 0.01 * w.grad

    # 3.5 打印每次迭代后的结果
    print(f'第{i}次更新，梯度值为{w.grad}，损失值为{loss}')

    # 4 打印最终结果
    print(f'最终结果：权重:{w:.5f},梯度:{w.grad:.5f},损失:{loss:.5f}')