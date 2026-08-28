# 导入相关模块
from cProfile import label

import torch
from pandas.core.common import random_state
from torch.utils.data import TensorDataset  # 构造数据集对象
from torch.utils.data import DataLoader  # 数据加载器
from torch import nn  # nn模块中有平方损失函数和假设函数
from torch import optim  # optim模块中有优化器函数
from sklearn.datasets import make_regression  # 创建线性回归模型数据集  这个包以后大概率不用，以后再写程序数据都是已有的，这一次需要自己造
import matplotlib.pyplot as plt  # 可视化

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


# 1. 定义函数，创建线性回归样本数据
def create_data():
    # 1. 创建数据集对象
    x, y, coef = make_regression(
        n_samples=100,  # 样本数量
        n_features=1,  # 特征数量
        noise=10,  # 噪声
        coef=True,  # 系数  ,是否返回系数，默认为False,返回值为None
        random_state=24,
        bias=14.5  # 偏置
    )

    # 2. 把上述的数据集对象封装成张量对象
    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)

    # 3. 返回结果
    return x, y, coef


# 2. 定义函数，训练模型
def train(x, y, coef):
    # 1. 创建数据集对象，把tensor(x,y) -> 数据集对象 -> 数据加载器
    dataset = TensorDataset(x, y)

    # 2. 创建数据加载器对象
    # 参1： 数据集对象     参2： 批次大小     参3： 是否打乱数据（训练集打乱，测试集不打乱）
    dataLoader = DataLoader(dataset, batch_size=16, shuffle=True)

    # 3. 创建初始的线性回归模型
    # 参1： 输入的特征维度  (输入的每一个样本有几个特征？)     参2： 输出的特征维度(对每一个样本，模型需要对该样本预测几个标签？)
    model = nn.Linear(1, 1)
    print("==== 训练前的初始状态 ====")
    print(f'初始权重 (Weight): {model.weight.data}')
    print(f'初始偏置 (Bias): {model.bias.data}')
    print("==================================")
    # 4. 创建损失函数对象
    criterion = nn.MSELoss()

    # 5. 创建优化器对象
    # 参1：模型参数   参2：学习率
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # 6. 具体的训练过程
    # 6.1 定义变量，分别表示：训练轮数，每轮的平均损失值（总损失数/样本数） ，训练总损失值， 训练的样本批次数
    epochs, loss_list, total_loss, total_sample = 100, [], 0.0, 0
    # 6.2 开始训练,按轮训练
    for epoch in range(epochs):
        # 6.3 每轮分批次训练，所以从数据加载器中 获取 批次数据
        for x_train, y_train in dataLoader:     # 7批，16，16，16，16，16，16，4
            # 6.4 模型预测
            y_pred = model(x_train)
            # 6.5 计算每一批的平均损失
            loss =criterion(y_pred ,y_train.reshape(-1,1))
            # 6.6 计算总损失 和 样本批次数
            total_loss += loss.item()
            total_sample+=1
            # 6.7 梯度清零 + 反向传播 + 梯度更新
            optimizer.zero_grad()           # 梯度清零
            loss.sum().backward()           # 反向传播，计算梯度
            optimizer.step()                # 梯度更新
        # 6.8 把本轮的平均损失值添加到列表中
        loss_list.append(total_loss/total_sample)
        print(f'轮数：{epoch+1},平均损失值：{total_loss/total_sample}')

    # 7. 打印最终的训练结果
    print(f'训练完成！，{epochs}轮的平均损失分别为：{loss_list},')
    print(f'模型参数为：权重：{model.weight},偏置：{model.bias}')

    # 8. 绘制损失曲线
    # 8.1 绘制损失曲线
    #               100轮            每轮的平均损失值
    plt.plot(range(epochs), loss_list)
    plt.title('损失值曲线的变化图')
    plt.grid()
    plt.show()

    # 9. 绘制预测值和真实值的关系
    # 9.1 绘制样本点分布情况
    plt.scatter(x, y)
    # 9.2 绘制训练模型的预测值
    # x: 100个样本点的特征
    y_pred=torch.tensor(data=[v* model.weight+model.bias for v in x])
    # 9.3 计算真实值
    y_true=torch.tensor(data=[v* coef +14.5 for v in x])
    # 9.4 绘制预测值和真实值的折线图
    plt.plot(x,y_pred,color='red',label='预测值')
    plt.plot(x,y_true,color='blue',label='真实值')
    # 9.5 图例
    plt.legend()
    plt.grid()
    # 9.6 显示
    plt.show()

    plt.show()

# 3. 测试
if __name__ == '__main__':
    # 3.1 创建数据集
    x, y, coef = create_data()
    print(f'x:{x}, y:{y}, coef:{coef}')

    # 3.2 训练模型
    train(x, y, coef)
    #


"""
从运行结果来看：为什么初始权重是0.8712，和模型最终学到的权重（74.95）和真实答案（76.50)差别这么大？

这个问题问得非常好，它触及了深度学习最核心的本质——“从无知到已知”的过程。

初始权重和最终权重之间之所以有这么大的天壤之别，是因为初始权重是模型在完全“蒙眼瞎猜”，而最终权重是模型经过了 100 次“纠错学习”的结果。

我们可以把整个过程拆解为三个阶段来看：

1. 初始权重 (0.8712)：完全是“盲盒瞎猜”
当你写下 model = nn.Linear(1, 1) 的那一瞬间，模型还没有见过任何一条数据，它根本不知道你要预测的是房价、温度还是什么东西。
为了打破对称性（前面我们讲过），PyTorch 会在 [−1,1] 之间随便掷一个骰子，恰好这次掷出了 0.8712。
这就好比我拿出一个外星水果让你猜价格，你完全没概念，随口瞎猜了一个“8毛7”。

2. 训练过程：一步步打醒模型 (0.87 ➔ 74.95)
在这个瞎猜的基础上，模型开始了它的第一轮预测：

模型用 0.87 的权重去算预测值，结果算出来错得离谱。

这时候 criterion 算出了高达 6364 的 Loss，相当于狠狠地给了模型一巴掌：“你猜得太低了，错得太离谱了！”

接着 loss.backward() 算出了梯度，告诉优化器：“赶紧把权重往上调！”

优化器 optimizer.step() 执行命令，把权重从 0.87 加大了一点点。

经过整整 100 轮这样的“预测 ➔ 挨打 ➔ 调整”，模型硬生生把权重从 0.87 缓慢且坚定地推到了 74.95。

3. 真实答案 (76.50)：藏在数据背后的真理
76.50 是你用 make_regression 造数据时，上帝视角的真实规则。模型永远无法直接“看”到这个数字，它只能通过观察那 100 个带有噪声的数据样本（X 和 Y），靠着反向传播一点点去逼近这个真理。
"""