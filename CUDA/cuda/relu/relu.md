### ReLU 函数

ReLU（Rectified Linear Unit）是一种常用的激活函数，其数学表达式为：

$$
\text{ReLU}(x) = \max(0, x) = 
\begin{cases}
x, & x > 0 \\
0, & x \leq 0
\end{cases}
$$

其中：

- \( x \) 是输入值（实数）
- 输出范围在 \([0, +\infty)\) 之间
- 在 \( x = 0 \) 处不可导，通常使用次梯度

---

### 导数形式

ReLU 函数在 \( x \neq 0 \) 处的导数为：

$$
\text{ReLU}'(x) = 
\begin{cases}
1, & x > 0 \\
0, & x < 0
\end{cases}
$$

在 \( x = 0 \) 处不可导，通常取次梯度为 0 或 1。

---

### 代码示例（Python）

```python
import numpy as np

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

# 使用示例
x = np.array([-2, -1, 0, 1, 2])
print(relu(x))            # [0, 0, 0, 1, 2]
print(relu_derivative(x)) # [0, 0, 0, 1, 1]
```

---
