### Sigmoid 函数

Sigmoid 函数是一种常用的 S 型激活函数，其数学表达式为：

$$
\sigma(x) = \frac{1}{1 + e^{-x}}
$$

其中：

- \( x \) 是输入值（实数）
- \( e \) 是自然对数的底数（约等于 2.71828）
- \( \sigma(x) \) 的输出范围在 (0, 1) 之间

---

### 导数形式

Sigmoid 函数的一个优秀性质是其导数可以用自身表示：

$$
\sigma'(x) = \sigma(x) \cdot (1 - \sigma(x))
$$

---

### 代码示例（Python）

```python
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
```

---
