"""
案例：
    演示自动微分的真实应用场景

结论：
    1. 先前向传播（正向传播），计算出预测值z
    2. 基于损失函数，结合预测值z 和真实值y 来计算梯度
    3. 结合权重更新公式 W新 = W旧 - 学习率 * 梯度，     来更新权重
"""
# 导包
import torch

# 1. 定义x ,表示特征（输入的数据），假设2行5列，全1矩阵
x = torch.ones(2, 5)
print(f'x = {x}')

# 2. 定义y,表示标签（真实值），假设2行3列，全0矩阵
y = torch.zeros(2, 3)
print(f'y:{y}')

# 3. 初始化（可自动微分的）权重和偏置
w = torch.randn(5, 3, requires_grad=True)  # X @ w + b  , 为什么要w=torch.randn()

print(f'w:{w}')

b = torch.randn(3, requires_grad=True)  # 这里有几个b跟 x@w 的列数一致，(2,5) @ (5,3) = (2,3),所以b的行数是3
print(f'b:{b}')

# 4. 基于前向传播计算出预测值z
z = torch.matmul(x, w) + b
# 也可以写成这样 z=x@w+b
print(f'z:{z}')

# 5. 定义损失函数，计算损失        在这里定义损失函数一般用criterion
criterion = torch.nn.MSELoss()   # nn -> neural network： 神经网络
loss =criterion(z,y)            # loss = 损失
print(f'loss:{loss}')

# 6. 进行自动微分，求导，结合反向传播，更新权重
loss.backward()         # note：前面打印出来的loss只有一个值，这里就可以不用写sum(),如果loss不是一个值，那就要写loss.sum().backward()

# 7. 打印w,b 用来更新的梯度
print(f'w的梯度：{w.grad}')
print(f'b的梯度：{b.grad}')

# 后续就是：W新=W旧-学习率*w的梯度， b新=b旧-学习率*b的梯度  来更新权重
w.data=w.data-0.01*w.grad
b.data=b.data-0.01*b.grad
print(f'更新后的w:{w.data}')
print(f'更新后的b:{b.data}')