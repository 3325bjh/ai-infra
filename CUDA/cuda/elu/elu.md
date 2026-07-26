### ELU 函数

ELU（Exponential Linear Unit）是一种激活函数，由 Djork-Arné Clevert 等人于 2015 年提出。其数学表达式为：

$$
\text{ELU}(x) = 
\begin{cases}
x, & x > 0 \\
\alpha (e^x - 1), & x \leq 0
\end{cases}
$$

其中：

- \( x \) 是输入值（实数）
- \( \alpha \) 是超参数，通常取 \( \alpha = 1.0 \)
- 输出范围在 \( (-\alpha, +\infty) \) 之间
- 在 \( x = 0 \) 处连续且可导（当 \( \alpha = 1 \) 时）

---

### 导数形式

ELU 函数的导数为：

$$
\text{ELU}'(x) = 
\begin{cases}
1, & x > 0 \\
\alpha e^x, & x \leq 0
\end{cases}
$$

由于 \( \text{ELU}(x) + \alpha = \alpha e^x \)（当 \( x \leq 0 \) 时），导数也可表示为：

$$
\text{ELU}'(x) = 
\begin{cases}
1, & x > 0 \\
\text{ELU}(x) + \alpha, & x \leq 0
\end{cases}
$$

---

### 代码示例（Python）

```python
import numpy as np

def elu(x, alpha=1.0):
    return np.where(x > 0, x, alpha * (np.exp(x) - 1))

def elu_derivative(x, alpha=1.0):
    return np.where(x > 0, 1, alpha * np.exp(x))

# 使用示例
x = np.array([-2, -1, 0, 1, 2])
print(elu(x))            # [-0.8647, -0.6321, 0, 1, 2]
print(elu_derivative(x)) # [0.1353, 0.3679, 1, 1, 1]
```

---

### ELU vs ReLU 对比

| 特性 | ReLU | ELU |
|------|------|-----|
| 负值输出 | ❌ 全部为 0 | ✅ 负值（趋近于 -α） |
| 均值漂移 | 可能偏移 | 更接近 0 均值 |
| 梯度消失 | 负区完全消失 | 负区有梯度（软饱和） |
| 计算开销 | 简单（比较/取零） | 稍高（指数运算） |
| 死神经元 | 容易产生 | 不易产生 |

---

### 适用场景

✅ ELU 适合的场景：
- 深层神经网络（缓解梯度消失）
- 需要负值响应的任务
- 对训练速度敏感的模型

❌ ELU 不适合的场景：
- 计算资源受限的环境（比 ReLU 慢）
- 不需要负值输出的简单任务

> 💡 **提示**：ELU 的变体还包括 **SELU**（Scaled ELU），通过固定 \( \alpha \approx 1.6733 \) 和缩放因子 \( \lambda \approx 1.0507 \) 实现自归一化特性。
```