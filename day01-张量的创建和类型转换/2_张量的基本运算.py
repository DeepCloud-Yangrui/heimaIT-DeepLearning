"""
案例：
    演示张量的基本运算
涉及到的API：
    add()、sub()、mul()、div()、neg()、      ->      加减乘除，取反
    add_()、sub_()、mul_()、div_()、neg_()、 ->      功能同上，只不过可以修改原数据，类似于pandas部分的 inplace = true
"""

# 导包
import torch
# 1. 创建张量
t1=torch.tensor([1,2,3])
# 2. 演示加减乘除取反
# t2=t1.add(10)
t2=t1.add_(10)
print(t1)
print(t2)
# 其他运算API效果同上