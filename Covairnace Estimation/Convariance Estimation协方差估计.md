# Covariance Estimation协方差估计

## 介绍

评估非线性最小二乘优化问题的结果的一个方法是分析协方差。

让我们考虑这样一个非线性回归问题。

$$y = f(x) + N(0, I)$$

即，观测值$y$是均值为$f(x)$的自由变量$x$的随机非线性函数，同时它的协方差是单位矩阵。那么给定观测值$y$的$x$最大似然估计就是非线性最小二乘问题的解：

$$x^* = \arg \min_x \|f(x) - y\|^2$$

$x^*$的协方差由下式给出

$$C(x^*) = \left(J'(x^*)J(x^*)\right)^{-1}$$

这里$J(x^*)$是$f$在$x^*$处的Jacobian。上式假设$J(x^*)$列满秩。

如果$J(x^*)$是非满秩的，协方差矩阵$C(x^*)$也是非满秩，可以通过Moore-Penrose伪逆求得：

$$C(x^*) =  \left(J'(x^*)J(x^*)\right)^{\dagger}$$

请注意，在上面的讨论组，我们假设$y$的协方差矩阵是单位矩阵。这是一个重要的假设。如果情况并非如此，我们有

$$y = f(x) + N(0, S)$$

相应的协方差估计由下式给出

$$C(x^*) = \left(J'(x^*) S^{-1} J(x^*)\right)^{-1}$$

因此，如果观测值具有非单位阵的协方差矩阵，则ceres solver用户需要正确使用协方差矩阵放缩相应的CostFunction，例如在上述情况下，该问题的成本函数应使用$S^{-1/2} f(x)$而不仅仅是$f(x)$，其中$S^{-1/2}$是协方差矩阵的平方根的逆矩阵。

## 尺度不变性

在运动结构（3D重建）问题中，重建在相似变换之前都是不明确的。这称为尺度模糊度。正确处理尺度需要使用SVD或自定义反演算法。对于小规模问题，用户可以使用密集算法求解。有关更多详细信息，请参阅 Kanatani 和 Morris[KanataniMorris](http://ceres-solver.org/bibliography.html#kanatanimorris)的工作。


**`Covariance`**

`Covariance`允许用户评估非线性最小二乘问题的协方差，并提供对其块的随机访问。协方差计算过程假设成本函数计算了残差，使得它们的协方差是单位矩阵。

由于协方差矩阵的计算可能计算大规模矩阵的逆矩阵，因此可能产生相当大的时间和内存消耗。然而，一般情况下用户只对协方差矩阵的一小部分感兴趣。通常只是块对角线。`Covariance`允许用户指定她感兴趣的协方差矩阵部分，然后使用此信息仅计算和存储协方差矩阵的这些部分

## Jacobian的秩

正如我们上面提到的，如果雅可比行列式是秩亏的，那么$J'J$的逆就没有定义，而是需要计算伪逆。

$J$的不满秩可能是结构性的——总是已知为零的列，或数值性的——具体取决于雅可比行列式中的确切值。


当问题包含恒定常量的参数块时，就会出现结构性的不满秩。这个类（Covariance）正确地处理了这样的结构等级缺陷。

数值性的不满秩，即矩阵的秩无法通过其稀疏结构来预测，并且需要查看其数值，这种情况更为复杂。这里又分为两种情况：

- 不满秩由过度参数化引起。例如，用于参数化$SO(3)$的四维四元数，它是一个三维流形。在这种情况下，用户应该使用`LocalParameterization`。这不仅能够求解器拥有更好的数值稳定性，还会向`Covariance`对象暴露不满秩，以便它能够正确处理该问题。
- 雅可比行列式中更一般的数值秩缺陷需要计算$J'J$的所谓奇异值分解 (SVD)。对于小型和中等规模的问题，可以使用稠密线性代数来完成。但是，我们不知道如何有效地对大型稀疏矩阵执行此操作。


`Covariance::Options`

> class Covariance::Options

> int Covariance:Options::num_threads

默认值:1

用于计算雅克比和协方差数值的线程数量

> SparseLinearAlgebraLibraryType Covariance::Options::sparse_linear_algebra_library_type

默认值:`SUITE_SPARSE`Ceres Solver构建时支持`SuiteSparse`和`EIGEN_SPARSE`。`EIGEN_SPARSE`总是可用。

> CovarianceAlgorithmType Covariance::Options::algorithm_type

默认值：SPARSE_QR

Ceres支持两种不同的协方差估计算法，它们在速度、准确性和可靠性方面进行了权衡。

1. `SPARSE_QR`使用稀疏 QR 分解算法来计算分解

$$ \begin{split}QR &= J\\
\left(J^\top J\right)^{-1} &= \left(R^\top R\right)^{-1}\end{split} $$

该算法的速度取决于所使用的稀疏线性代数库。`Eigen`的稀疏 QR 分解是一种速度适中的算法，适用于中小型矩阵。为了获得最佳性能，我们建议使用   `SuiteSparseQR`，它是通过将 `Covariance::Options::sparse_linear_algebra_library_type`设置为`SUITE_SPARSE`来启用的。

如果`Jacobian`秩亏，`SPARSE_QR`无法计算协方差。

2. `DENSE_SVD`使用`Eigen`的`JacobiSVD`来执行计算。它计算奇异值分解

$$U D V^\top = J$$

 然后用它来计算 J'J 的伪逆：
 
 $$(J'J)^{\dagger} = V  D^{2\dagger}  V^\top$$
 
 这是一种准确但缓慢的方法，仅适用于中小型问题。  它可以处理满秩和缺秩雅可比行列式。

> double Covariance::Options::column_pivot_threshold

默认值: -1

在 QR 分解过程中，如果遇到欧几里得范数小于`column_pivot_threshold`的列，则将其视为零。

如果`column_pivot_threshold < 0`，则使用自动默认值$20*(m+n)*eps*sqrt(max(diag(J'*J)))$。这里$m$和$n$分别是雅可比矩阵$(J)$的行数和列数。

这是一个高级选项，适用于对雅可比矩阵有足够了解的用户，他们可以确定比默认值更好的值

>int Covariance::Options::min_reciprocal_condition_number

默认值： $10^{-14}$

如果雅可比矩阵接近奇异，则求$J'J$的逆将导致不可靠的结果，例如，如果

$$\begin{split}J = \begin{bmatrix}
    1.0& 1.0 \\
    1.0& 1.0000001
    \end{bmatrix}\end{split}$$

这本质上是一个秩亏矩阵，我们有

$$\begin{split}(J'J)^{-1} = \begin{bmatrix}
             2.0471e+14&  -2.0471e+14 \\
             -2.0471e+14&   2.0471e+14
             \end{bmatrix}\end{split}$$

这不是一个可用的结果。因此，默认情况下，如果遇到秩亏雅可比行列式，`Covariance::Compute()`将返回 false。如何检测不满秩取决于所使用的算法。

1. `DENSE_SVD`

$$ \frac{\sigma_{\text{min}}}{\sigma_{\text{max}}} < \sqrt{{min\_reciprocal\_condition\_number}}$$

其中$\sigma_{\text{min}}$和$\sigma_{\text{min}}$分别是 的最小和最大奇异值。

2. `SPARSE_QR`

$$\operatorname{rank}(J) < \operatorname{num\_col}(J)$$

这是SPARSE_QR分解算法返回的秩的估计。使用这种方法判断不满秩很靠谱。

> int Covariance::Options::null_space_rank

使用`DENSE_SVD`时，用户可以更好地控制处理奇异和近奇异协方差矩阵。

 如上所述，当协方差矩阵接近奇异时，不计算$J'J$的逆矩阵
 ，而应该计算摩尔-彭罗斯伪逆。

 如果$J'J$具有特征分解$(\lambda_i,
e_i)$，其中$\lambda_i$是特征值,$e_i$是相应的特征向量，则计算$J'J$的逆矩阵可以计算：

$$(J'J)^{-1} = \sum_i \frac{1}{\lambda_i} e_i e_i'$$

 计算伪逆过程需要删除与小特征值相对应的项的和。

 如何删除项由`min_reciprocal_condition_number`和 `null_space_rank`控制。

 如果`null_space_rank`为非负数，则最小的`null_space_rank`特征值/特征向量将被丢弃，而不管$\lambda_i$幅值的大小。如果截断矩阵中最小非零特征值与最大特征值的比率仍然低于   `min_reciprocal_condition_number`，则`Covariance::Compute()`将失败并返回$false$。

 设置`null_space_rank = -1`会删除所有项。 此选项对 SPARSE_QR 没有影响

> bool Covariance::Options::apply_loss_function

默认值：*true*

即使问题中的残差块可能已经包含损失函数，将`apply_loss_function`设置为$false$也会停止损失函数，进而停止其对协方差的影响。

> class Covariance

`Covariance::Options`顾名思义是用来控制协方差估计算法的。协方差估计是一个复杂且数字敏感的过程。在使用`Covariance`之前，请阅读 `Covariance::Options`的完整文档。

>bool Covariance::Compute(const vector<pair<const double*, const double*>> &covariance_blocks, Problem *problem)

计算协方差矩阵的一部分。

 向量`covariance_blocks`使用参数块pair按块索引到协方差矩阵。这允许协方差估计算法仅计算和存储这些被索引块。

 由于协方差矩阵是对称的，如果用户传递`<block1，block2>`，则可以使用`block1`，`block2`以及`block2`，`block1`来调用`GetCovarianceBlock`。

 `covariance_blocks`不能包含重复项。包含重复项会导致错误的结果。

 请注意，`covariance_blocks`列表仅用于确定计算协方差矩阵的哪些部分。完整的雅可比行列式用于进行计算，即它们不会影响雅可比行列式的哪一部分用于计算。

 返回值指示协方差计算的成功或失败。请参阅上面`Covariance::Options`文档，了解有关此函数返回 false 的条件的更多信息。

 >bool GetCovarianceBlock(const double *parameter_block1, const double *parameter_block2, double *covariance_block) const

返回`parameter_block1`和`parameter_block2`对应的互协方差矩阵的块。

 必须在第一次调用`GetCovarianceBlock`之前调用`Compute`，并且在调用`Compute`时，`<parameter_block1,parameter_block2>` 参数对或 `<parameter_block2,parameter_block1>` 必须已存在于向量 `covariance_blocks`中。否则`GetCovarianceBlock`将返回 false。

 `covariance_block`必须指向可以存储`parameter_block1_size xparameter_block2_size`矩阵的内存位置。返回的协方差将是行向量组成的矩阵。

>bool GetCovarianceBlockInTangentSpace(const double *parameter_block1, const double *parameter_block2, double *covariance_block) const

返回`parameter_block1`和`parameter_block2`对应的互协方差矩阵的块。如果任一参数块进行了Local Parameterization，则返回正交切空间中的互协方差；否则返回环境空间中的互协方差。

 必须在第一次调用`GetCovarianceBlock`之前调用`Compute`，并且在调用`Compute`时，`<parameter_block1,parameter_block2>` 参数对或 `<parameter_block2,parameter_block1>` 必须已存在于向量 `covariance_blocks`中。否则`GetCovarianceBlock`将返回 false。

## 用例

```cpp
double x[3];
double y[2];

Problem problem;
problem.AddParameterBlock(x, 3);
problem.AddParameterBlock(y, 2);
<Build Problem>
<Solve Problem>

Covariance::Options options;
Covariance covariance(options);

vector<pair<const double*, const double*> > covariance_blocks;
covariance_blocks.push_back(make_pair(x, x));
covariance_blocks.push_back(make_pair(y, y));
covariance_blocks.push_back(make_pair(x, y));

CHECK(covariance.Compute(covariance_blocks, &problem));

double covariance_xx[3 * 3];
double covariance_yy[2 * 2];
double covariance_xy[3 * 2];
covariance.GetCovarianceBlock(x, x, covariance_xx)
covariance.GetCovarianceBlock(y, y, covariance_yy)
covariance.GetCovarianceBlock(x, y, covariance_xy)

```



