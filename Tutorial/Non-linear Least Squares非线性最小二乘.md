# Non-linear Least Squares非线性最小二乘

## 介绍

Ceres可以解决有约束的鲁棒非线性最小二乘问题，问题形式如下式所示

$$
\min_{x} \frac{1}{2}\sum_i\rho_i(\|f_i(x_{i_{1},...,x_{x_k}}) \|^2) \\ s.t. \quad l_j \le x_j \le u_j
$$

上述形式的问题在科学和工程领域广泛出现——从统计学中的曲线拟合（Curve Fitting）问题到计算机视觉中利用照片进行3D模型重建的任务，你都能看到非线性最小二乘问题的身影。

在本章节中，我们将学习如何使用Ceres Solver求解上述问题。本章以及后续章节里示例的完整代码可以在Ceres Solver工程下的[examples](https://github.com/ceres-solver/ceres-solver/tree/master/examples)目录中找到。

表达式中$\rho_i(\|f_i(x_{i_{1},...,x_{x_k}}) \|^2)$这部分被称作`ResidualBlock`，其中$f_i(\cdot)$叫做`CostFunction`，其具体形式由若干参数块（parameter blocks）$[x_{i_1}, \dots,x_{i_k}]$决定。在大部分优化问题中，若干个标量参数会以小组的形式出现。例如，平移向量中的3个分量以及四元数的4个分量，组成了表示相机姿态的两组参数。我们将这一组参数称为一个参数块`ParameterBlock`。当然，一个`ParameterBlock`也可以指单个参数。$l_j$和$u_j$表示参数块$x_j$的下界和上界。

$\rho_i$表示一个`LossFunction`。`LossFunction`是一个标量函数，用于在求解非线性最小二乘问题时减少离群值带来的影响。

特别的，当$\rho_i(x)=x$，即取恒等变换，且满足$l_j=-\infty$$和u_j=\infty$时，我们得到了非线性最小二乘问题[更常见的形式](https://en.wikipedia.org/wiki/Non-linear_least_squares)。

$$
\frac{1}{2}\sum\|f_i(x_{i_1},\dots,f_{i_k})\|^2
$$




