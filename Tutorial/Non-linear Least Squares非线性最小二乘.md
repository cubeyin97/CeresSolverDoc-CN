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

## Hello World

首先，让我们考虑寻找下述函数的最小值

$$
\frac{1}{2}(10-x)^2
$$

这是一个非常简单的问题，函数在$x=10$处取得最小值，我们将用这个例子来展示如何利用Ceres求解问题。

第一步是用一个函数对象/仿函数（Functor）来表示（残差）函数$f(x)=10-x$

```c++
struct CostFunctor {
   template <typename T>
   bool operator()(const T* const x, T* residual) const {
     residual[0] = 10.0 - x[0];
     return true;
   }
};
```

需要注意，`operator`是一个模板方法，它所有输入和输出的数据类型都为模板类型`T`。之所以使用模板方法，是为了让Ceres在调用`CostFunctor::operator<T>()`时能够兼容不同的数据类型：当仅需要计算残差时，模板类型`T=double`；而当需要计算雅克比的时候，会使用一种特殊的数据类型`T=Jet`。在[Derivatives导数]()中我们会详细讨论在Ceres计算导数的多种方法。

当我们获得了计算残差函数的方法后，接下来就可以构建非线性最小二乘问题并使用Ceres求解。

```c++
int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);

  // 待求解变量和初始值
  double initial_x = 5.0;
  double x = initial_x;

  // 构建优化问题
  Problem problem;

  // 设置CostFunction（也就是残差），使用自动求导来计算导数（雅克比）
  CostFunction* cost_function =
      new AutoDiffCostFunction<CostFunctor, 1, 1>(new CostFunctor);
  problem.AddResidualBlock(cost_function, nullptr, &x);

  // 执行求解过程！
  Solver::Options options;
  options.linear_solver_type = ceres::DENSE_QR;
  options.minimizer_progress_to_stdout = true;
  Solver::Summary summary;
  Solve(options, &problem, &summary);

  std::cout << summary.BriefReport() << "\n";
  std::cout << "x : " << initial_x
            << " -> " << x << "\n";
  return 0;
}
```

`AutoDiffCostFunction`的输入是一个`CostFunctor`，它会对输入的函数对象自动进行求导并给出一个`CostFunction`接口。

编译并运行代码[examples/helloworld.cc](https://ceres-solver.googlesource.com/ceres-solver/+/master/examples/helloworld.cc)，结果如下

```sh
iter      cost      cost_change  |gradient|   |step|    tr_ratio  tr_radius  ls_iter  iter_time  total_time
   0  4.512500e+01    0.00e+00    9.50e+00   0.00e+00   0.00e+00  1.00e+04       0    5.33e-04    3.46e-03
   1  4.511598e-07    4.51e+01    9.50e-04   9.50e+00   1.00e+00  3.00e+04       1    5.00e-04    4.05e-03
   2  5.012552e-16    4.51e-07    3.17e-08   9.50e-04   1.00e+00  9.00e+04       1    1.60e-05    4.09e-03
Ceres Solver Report: Iterations: 2, Initial cost: 4.512500e+01, Final cost: 5.012552e-16, Termination: CONVERGENCE
x : 0.5 -> 10
```

$x$的初始值为5，两个循环后变成了10。认真的读者会发现这是一个线性问题，使用线性求解器就可以处理这个问题。默认配置里的求解器是用于求解非线性优化问题，这里为了实现简洁我们并没有修改默认选项。事实上，对于这个问题我们使用Ceres可以在一轮循环中得到答案。除此之外，我们还可以看到，求解器在第一轮优化中就让待优化值非常接近0。我们将在后续讨论关于收敛和参数设置的时候深入讨论上述现象。


注：
- 事实上求解器跑了三次迭代循环，在第三次迭代结束时，它注意到参数块的更新量非常小，因而认为结果已收敛。Ceres只会在一次迭代完全结束的时候打印结果，而会在检测到收敛的时候立即停止迭代，因此我们只能看到前两次迭代的结果（第三次迭代因收敛而终止，未打印log）。

## Derivative导数

和大多数优化库类似，Ceres通过计算目标函数任意参数、任意自变量下的函数数值和导数来求解问题。正确高效的数值计算对于求解问题至关重要。Ceres Solver提供了多种求解方式，在前面[examples/helloworld.cc](https://ceres-solver.googlesource.com/ceres-solver/+/master/examples/helloworld.cc)中已经见识到了Automatic Differentiation自动求导的表现。

现在我们来看另外两种选择，解析求导和数值求导。

### 数值求导

在某些情况下，我们无法定义一个模板函数对象，比如我们需要在计算残差时调用一个你无法查看源码的库函数。在这种情况下，可以使用数值求导。用户定义一个计算残差的函数对象，并使用它来构造一个`NumericDiffCostFunction`。举个例子，对于上面的函数$f(x)=10-x$对应的函数对象为

```c++
struct NumericDiffCostFunctor {
  bool operator()(const double* const x, double* residual) const {
    residual[0] = 10.0 - x[0];
    return true;
  }
};
```

接下来将它添加到`Problem`对象

```c++
CostFunction* cost_function = new NumericDiffCostFunction<NumericDiffCostFunctor, ceres::CENTRAL, 1, 1>(new NumericDiffCostFunctor);
problem.AddResidualBlock(cost_function, nullptr, &x);
```

对比我们在使用自动求导的时候，使用方法如下
```c++
CostFunction* cost_function =
    new AutoDiffCostFunction<CostFunctor, 1, 1>(new CostFunctor);
problem.AddResidualBlock(cost_function, nullptr, &x);
```
数值求导的Costfunctor和自动求导的看起来几乎一模一样，除了在参数类型上数值求导没有用模板参数，而是指明了用于导数计算的类型。更多细节参考`NumericDiffCostFunction`的文档。

**通常来说我们推荐使用自动求导而不是数值求导。使用模板的自动求导更为高效，而数值求导的计算代价更为昂贵，并且容易受到数值稳定性问题的影响，导致收敛更慢**

### 解析导数

在某些场景下不应该使用自动求导。比如，在一些场景中可能使用闭式解析解计算导数更加高效，这种时候不应该依赖自动求导中的链式法则。

在这些应用解析导数的场景中，可以提供自行编写的残差计算和雅克比计算的代码。为此，你需要定义一个`CostFunction`的子类，或者，如果你知道参数和残差在编译时的具体维度，你可以使用`SizedCostFunction`。这里有一个实现$f(x)=10-x$解析求导的例子`SimpleCostFunction`

```c++
class QuadraticCostFunction : public ceres::SizedCostFunction<1, 1> {
 public:
  virtual ~QuadraticCostFunction() {}
  virtual bool Evaluate(double const* const* parameters,
                        double* residuals,
                        double** jacobians) const {
    const double x = parameters[0][0];
    residuals[0] = 10 - x;

    // 如果jacobians不为空才表示需要计算雅克比
    if (jacobians != nullptr && jacobians[0] != nullptr) {
      jacobians[0][0] = -1;
    }
    return true;
  }
};
```

`SimpleCostFunction::Evaluate`的输入参数是一个叫做`parameters`的数组，输出参数包括一个表示残差的数组`residuals`和一个表示雅克比的数组`jacobians`。`jacobians`是一个可选参数，`Evaluate`函数需要检查它是否为空，如果非空，则将参数对应的导数填入其中。在这个例子中残差方程是线性的，所以雅克比是一个常数。

如你所见，实现一个`CostFunction`对象的过程有一点枯燥。我们建议除非你有不得不手动计算雅克比的需求，最好还是用`AutoDiffCostFunction`或者`NumericDiffCostFunction`来构建残差块。

### 关于导数的更多内容

计算导数是目前为止使用Ceres时最复杂的部分，在不同的场景中用户可能需要更加复杂的计算导数的方式。这一部分只是简略的说明了在Ceres中是如何使用导数的。当你可以熟练使用`NumericDiffCostFunction`和`AutoDiffCostFunction`之后，我们推荐继续了解`DynamicAutoDiffCostFunction`，`CostFunctionToFunctor`，`NumericDiffFunctor`和`ConditionedCostFunction`等计算成本函数的进阶方法。

注：
- [examples/helloworld_numeric_diff.cc.](https://ceres-solver.googlesource.com/ceres-solver/+/master/examples/helloworld_numeric_diff.cc)
- [examples/helloworld_analytic_diff.cc.](https://ceres-solver.googlesource.com/ceres-solver/+/master/examples/helloworld_analytic_diff.cc)


## Powell函数

现在来考虑一个更复杂一点的例子——最小化Powell函数的数值。另$x=[x_1, x_2, x_3, x_4]$

$$
f_1(x) = x_1 + 10 x_2 \\ f_2(x) = \sqrt{5}(x_3 - x_4) \\
f_3(x) = (x_2 - 2x_3)^2 \\ f_4(x) = \sqrt{10}(x_1 - x_4)^2
\\ F(x) = [f_1(x), f_2(x), f_3(x), f_4(x)]
$$

$F(x)$是一个拥有四个参数的函数，并有四部分残差组成。我们希望找到一个$x$使得$\frac{1}{2}\|F(x)\|^2$最小化

与之前的方法类似，第一步是定义函数对象来表示上述函数。这里给出了计算$f_4(x_1, x_4)$的代码：

```cpp
struct F4 {
  template <typename T>
  bool operator()(const T* const x1, const T* const x4, T* residual) const {
    residual[0] = sqrt(10.0) * (x1[0] - x4[0]) * (x1[0] - x4[0]);
    return true;
  }
};
```

类似的，我们可以定义类`F1`,`F2`,`F3`来计算$f_1(x_1, x_2)$ , $f_2(x_3,x_4)$以及$f_3(x_2,x_3)$。使用这些定义的类，就可以构建待求解问题


```cpp
double x1 =  3.0; double x2 = -1.0; double x3 =  0.0; double x4 = 1.0;

Problem problem;

// 构建问题时，使用自动求导来添加残差项

problem.AddResidualBlock(
  new AutoDiffCostFunction<F1, 1, 1, 1>(new F1), nullptr, &x1, &x2);
problem.AddResidualBlock(
  new AutoDiffCostFunction<F2, 1, 1, 1>(new F2), nullptr, &x3, &x4);
problem.AddResidualBlock(
  new AutoDiffCostFunction<F3, 1, 1, 1>(new F3), nullptr, &x2, &x3);
problem.AddResidualBlock(
  new AutoDiffCostFunction<F4, 1, 1, 1>(new F4), nullptr, &x1, &x4);
```

请注意每一个`ResidualBlock`只依赖两个参数而不是全部四个参数。编译程序[examples/powells.cc](https://ceres-solver.googlesource.com/ceres-solver/+/master/examples/powell.cc)可以得到输出：


```cpp
Initial x1 = 3, x2 = -1, x3 = 0, x4 = 1
iter      cost      cost_change  |gradient|   |step|    tr_ratio  tr_radius  ls_iter  iter_time  total_time
   0  1.075000e+02    0.00e+00    1.55e+02   0.00e+00   0.00e+00  1.00e+04       0    4.95e-04    2.30e-03
   1  5.036190e+00    1.02e+02    2.00e+01   2.16e+00   9.53e-01  3.00e+04       1    4.39e-05    2.40e-03
   2  3.148168e-01    4.72e+00    2.50e+00   6.23e-01   9.37e-01  9.00e+04       1    9.06e-06    2.43e-03
   3  1.967760e-02    2.95e-01    3.13e-01   3.08e-01   9.37e-01  2.70e+05       1    8.11e-06    2.45e-03
   4  1.229900e-03    1.84e-02    3.91e-02   1.54e-01   9.37e-01  8.10e+05       1    6.91e-06    2.48e-03
   5  7.687123e-05    1.15e-03    4.89e-03   7.69e-02   9.37e-01  2.43e+06       1    7.87e-06    2.50e-03
   6  4.804625e-06    7.21e-05    6.11e-04   3.85e-02   9.37e-01  7.29e+06       1    5.96e-06    2.52e-03
   7  3.003028e-07    4.50e-06    7.64e-05   1.92e-02   9.37e-01  2.19e+07       1    5.96e-06    2.55e-03
   8  1.877006e-08    2.82e-07    9.54e-06   9.62e-03   9.37e-01  6.56e+07       1    5.96e-06    2.57e-03
   9  1.173223e-09    1.76e-08    1.19e-06   4.81e-03   9.37e-01  1.97e+08       1    7.87e-06    2.60e-03
  10  7.333425e-11    1.10e-09    1.49e-07   2.40e-03   9.37e-01  5.90e+08       1    6.20e-06    2.63e-03
  11  4.584044e-12    6.88e-11    1.86e-08   1.20e-03   9.37e-01  1.77e+09       1    6.91e-06    2.65e-03
  12  2.865573e-13    4.30e-12    2.33e-09   6.02e-04   9.37e-01  5.31e+09       1    5.96e-06    2.67e-03
  13  1.791438e-14    2.69e-13    2.91e-10   3.01e-04   9.37e-01  1.59e+10       1    7.15e-06    2.69e-03

Ceres Solver v1.12.0 Solve Report
----------------------------------
                                     Original                  Reduced
Parameter blocks                            4                        4
Parameters                                  4                        4
Residual blocks                             4                        4
Residual                                    4                        4

Minimizer                        TRUST_REGION

Dense linear algebra library            EIGEN
Trust region strategy     LEVENBERG_MARQUARDT

                                        Given                     Used
Linear solver                        DENSE_QR                 DENSE_QR
Threads                                     1                        1
Linear solver threads                       1                        1

Cost:
Initial                          1.075000e+02
Final                            1.791438e-14
Change                           1.075000e+02

Minimizer iterations                       14
Successful steps                           14
Unsuccessful steps                          0

Time (in seconds):
Preprocessor                            0.002

  Residual evaluation                   0.000
  Jacobian evaluation                   0.000
  Linear solver                         0.000
Minimizer                               0.001

Postprocessor                           0.000
Total                                   0.005

Termination:                      CONVERGENCE (Gradient tolerance reached. Gradient max norm: 3.642190e-11 <= 1.000000e-10)

Final x1 = 0.000292189, x2 = -2.92189e-05, x3 = 4.79511e-05, x4 = 4.79511e-05

```

很容易可以看到该问题的解为$x_1 = 0, x_2 = 0 x_3 = 0, x_4 = 0$时目标函数取得最小值0。经过十轮迭代，ceres找到了目标函数的解为$4*10^{-12}$



## 曲线拟合

目前我们看到的优化问题都是没有数据的。最小二乘和非线性最小二乘的研究最早是为了求解曲线拟合的问题。让我们来看这样一个问题。现在有一些通过采样曲线$y = e^{0.3x+0.1}$和添加标准差为$\sigma=0.2$的高斯噪声生成的数据。现在尝试利用这些数据拟合曲线

$$y= e^{mx+c}$$

首先类定义一个模板类来计算残差。对每一个观测都会有一项残差。

```cpp
struct ExponentialResidual {
  ExponentialResidual(double x, double y) 
    :x_(x), y_(y) {}
  template<typename T>
  bool operator()(const T* const m, const T* const c, T* residual) const {
    residual[0] = y_ - exp(m[0] * x_ + c[0]);
    return true;
  }

  private:
  // 观测数据
  double x_;
  double y_;
};

```

假设观测数据叫做`data`，是一个大小为$2n$的数组，那么构建待优化问题的过程就是为每一个观测样本生成一个`CostFunction`:

```cpp
double m = 0.0;
double c = 0.0;

Problem problem;

for(int i=0; i < kNumObservations; ++i) {
  CostFunction* cost_function = 
    new AutoDiffCostFunction<ExponentialResidual, 1, 1, 1> (
      new ExponentialResidual(data[2*i], data[2*i+1]));
  problem.AddResidualBlock(cost_function, nullptr, &m, &c);
}

```

编译并运行代码[examples/curve_fitting.cc](https://ceres-solver.googlesource.com/ceres-solver/+/master/examples/curve_fitting.cc)可以得到输出:

```cpp
iter      cost      cost_change  |gradient|   |step|    tr_ratio  tr_radius  ls_iter  iter_time  total_time
   0  1.211734e+02    0.00e+00    3.61e+02   0.00e+00   0.00e+00  1.00e+04       0    5.34e-04    2.56e-03
   1  1.211734e+02   -2.21e+03    0.00e+00   7.52e-01  -1.87e+01  5.00e+03       1    4.29e-05    3.25e-03
   2  1.211734e+02   -2.21e+03    0.00e+00   7.51e-01  -1.86e+01  1.25e+03       1    1.10e-05    3.28e-03
   3  1.211734e+02   -2.19e+03    0.00e+00   7.48e-01  -1.85e+01  1.56e+02       1    1.41e-05    3.31e-03
   4  1.211734e+02   -2.02e+03    0.00e+00   7.22e-01  -1.70e+01  9.77e+00       1    1.00e-05    3.34e-03
   5  1.211734e+02   -7.34e+02    0.00e+00   5.78e-01  -6.32e+00  3.05e-01       1    1.00e-05    3.36e-03
   6  3.306595e+01    8.81e+01    4.10e+02   3.18e-01   1.37e+00  9.16e-01       1    2.79e-05    3.41e-03
   7  6.426770e+00    2.66e+01    1.81e+02   1.29e-01   1.10e+00  2.75e+00       1    2.10e-05    3.45e-03
   8  3.344546e+00    3.08e+00    5.51e+01   3.05e-02   1.03e+00  8.24e+00       1    2.10e-05    3.48e-03
   9  1.987485e+00    1.36e+00    2.33e+01   8.87e-02   9.94e-01  2.47e+01       1    2.10e-05    3.52e-03
  10  1.211585e+00    7.76e-01    8.22e+00   1.05e-01   9.89e-01  7.42e+01       1    2.10e-05    3.56e-03
  11  1.063265e+00    1.48e-01    1.44e+00   6.06e-02   9.97e-01  2.22e+02       1    2.60e-05    3.61e-03
  12  1.056795e+00    6.47e-03    1.18e-01   1.47e-02   1.00e+00  6.67e+02       1    2.10e-05    3.64e-03
  13  1.056751e+00    4.39e-05    3.79e-03   1.28e-03   1.00e+00  2.00e+03       1    2.10e-05    3.68e-03
Ceres Solver Report: Iterations: 13, Initial cost: 1.211734e+02, Final cost: 1.056751e+00, Termination: CONVERGENCE
Initial m: 0 c: 0
```

初始参数数值$m=0$,$c=0$对应目标函数的数值为121.173，ceres最终得到的参数为$m=0.291861$,$c=0.131439$，对应目标函数的数值为1.05675。这些数值和原模型中的$m=0.3$,$c=0.1$有些许差别，这是可以预见的。当利用含有噪声的数据来拟合曲线的时候，我们已预料到了这种情况。事实上，如果用$m=0.3,c=0.1$代入模型来计算目标函数，我们反而会得到更差的结果1.082425。

下图展示了这一结果。

![最小二乘拟合曲线](./pics/least_squares_fit.png)

## 鲁棒曲线拟合

现在假设数据中有一些离群点，比如一些点不符合噪声模型。如果使用上面的代码来拟合函数，会获得下图中的拟合结果

![最小二乘拟合曲线](./pics/non_robust_least_squares_fit.png)

为了处理这些离群点，一种标准做法是使用`LossFunction`。LossFunction能够减小数值巨大的残差带来的影响，这些数值巨大的残差块通常对应离群点。为了将LossFunction和残差块结合在一起，我们将

```cpp
problem.AddResidualBlock(cost_function, nullptr, &m, &c);
```

转换为


```cpp
problem.AddResidualBlock(cost_function, new CauchyLoss(0.5), &m, &c);
```

`CauchyLoss`是CeresSolver自带的LossFunction之一。参数0.5指定了它的尺度。使用LossFunction之后，我们得到了如下的结果。可以看到拟合后的曲线和曲线真值更为接近了。

![鲁棒曲线拟合](./pics/robust_least_squares_fit.png)


## 集束调整

设计Ceres的最主要的一个原因是求解大规模的集束调整问题。

给定一系列的图像特征的位置和对应关系，集束调整的目标是寻找3D点的坐标和相机参数来最小化重投影误差。该优化问题通常建模为最小二乘问题，残差为观测值与对应3D点在相机平面投影值的差的L2范数的平方。Ceres非常适合求解集束调整问题。

让我们解决一个来自[BAL](http://grail.cs.washington.edu/projects/bal/)数据集的优化问题。

同往常一样，第一步是定义一个模板类来计算重投影误差/残差。这个函数对象的结构与`ExponentialResidual`相似，每一个该类别的实例对象对应一个图像观测。

BAL问题中的每一个残差的计算都需要一个三维点和相机的九个参数。定义相机的九个参数是：用于表示旋转的Rodrigues轴角向量的三个分量，用于表示平移的三个参数，一个焦距，以及两个径向畸变参数。相机模型的详细信息可以在[Bundler homepage](http://phototour.cs.washington.edu/bundler/)以及[BAL homePage](http://grail.cs.washington.edu/projects/bal/)中找到。


```cpp
struct SnavelyReprojectionError {
  SnavelyReprojectionError(double observed_x, double observed_y)
      : observed_x(observed_x), observed_y(observed_y) {}

  template <typename T>
  bool operator()(const T* const camera,
                  const T* const point,
                  T* residuals) const {
    // camera[0,1,2]是旋转轴角
    T p[3];
    ceres::AngleAxisRotatePoint(camera, point, p);
    // camera[3,4,5]代表平移参数
    p[0] += camera[3]; p[1] += camera[4]; p[2] += camera[5];

    // 计算畸变中心，符号来自Noah Snavely的集束假设，相机坐标系里的z
    // 轴指向外侧，成倒像
    T xp = - p[0] / p[2];
    T yp = - p[1] / p[2];

    // 计算径向畸变的二次项和四次项
    const T& l1 = camera[7];
    const T& l2 = camera[8];
    T r2 = xp*xp + yp*yp;
    T distortion = 1.0 + r2  * (l1 + l2  * r2);

    // 计算最终的投影点坐标
    const T& focal = camera[6];
    T predicted_x = focal * distortion * xp;
    T predicted_y = focal * distortion * yp;

    // 重投影误差是观测值和预测值间的坐标差
    residuals[0] = predicted_x - T(observed_x);
    residuals[1] = predicted_y - T(observed_y);
    return true;
  }

   // 使用工厂模式，用户可以直接使用该静态函数构造CostFunction
   static ceres::CostFunction* Create(const double observed_x,
                                      const double observed_y) {
     // 2: 代价函数的残差维度，分别是x和y方向的重投影误差
     // 9和3: 优化问题的参数块的维度，第一个参数块的维度是9，第二个参数块的维度是3
     return (new ceres::AutoDiffCostFunction<SnavelyReprojectionError, 2, 9, 3>(
                 new SnavelyReprojectionError(observed_x, observed_y)));
   }

  double observed_x;
  double observed_y;
};

```

与之前的例子不同，这个函数比较复杂，计算雅克比的解析形式会令人痛苦。自动求导让这个过程简单了许多。函数中的`AngleAxisRotatePoint()`以及其他完成旋转过程的函数可以在`include/ceres/rotation.h`中找到。

有了上述的函数对象之后，就可以构造集束优化问题了

```cpp
ceres::Problem problem;
for (int i = 0; i < bal_problem.num_observations(); ++i) {
  ceres::CostFunction* cost_function =
      SnavelyReprojectionError::Create(
           bal_problem.observations()[2 * i + 0],
           bal_problem.observations()[2 * i + 1]);
  problem.AddResidualBlock(cost_function,
                           nullptr /* squared loss */,
                           bal_problem.mutable_camera_for_observation(i),
                           bal_problem.mutable_point_for_observation(i));
}
```

BA问题的构造过程与曲线拟合问题的构造过程十分相似，每一个观测都会有一项残差被加入到问题中。

由于这是一个大型的稀疏问题（对`DENSE_QR`来说已经非常大了），求解这个问题的一种方案是使用`SPARSE_NORMAL_CHOLESKY`作为`Solver::Options::linear_solver_type`的值，然后调用`Solve()`函数。BA问题有一个特殊的系数结构可以被利用，使得求解过程更为高效。Ceres为BA提供了三种特殊的求解器（被称作基于舒尔的求解器）。实例代码中使用了三种求解器中最简单的一种`DENSE_SCHUR`。

```cpp
ceres::Solver::Options options;
options.linear_solver_type = ceres::DENSE_SCHUR;
options.minimizer_progress_to_stdout = true;
ceres::Solver::Summary summary;
ceres::Solve(options. &problem, &summary);
std::cout << summary.FullReport() << "\n";
```

对于更复杂的BA问题，使用Ceres中的高级特性例如多种线性求解器，鲁棒LossFunction和流型来求解的例子可以参照[examples/bundle_adjuster.cc](https://ceres-solver.googlesource.com/ceres-solver/+/master/examples/bundle_adjuster.cc)


## 其他例子

除了本章展示的例子之外，[example](https://ceres-solver.googlesource.com/ceres-solver/+/master/examples/)文件夹中也包含了许多别的例子。

1. [bundle_adjuster.cc](https://ceres-solver.googlesource.com/ceres-solver/+/master/examples/bundle_adjuster.cc)展示了如何使用多种Ceres的特性来求解BA问题。
2. [circle_fit.cc](https://ceres-solver.googlesource.com/ceres-solver/+/master/examples/circle_fit.cc)如何用数据来拟合原型
3. [ellipse_approximation.cc](https://ceres-solver.googlesource.com/ceres-solver/+/master/examples/ellipse_approximation.cc)使用在椭圆上随机采样的数据来拟合一个合适的有分段线段组成的包络线。该优化过程同时优化包络线的控制点和数据在原图像上的坐标。这个例子的目的是展示如何使用`Solver::Options::dynamic_sparsity`，以及这个求解器为何善于处理在数值上稠密但是动态过程中具有稀疏结构的问题。
4. [denoising.cc](https://ceres-solver.googlesource.com/ceres-solver/+/master/examples/denoising.cc)使用[Field of Experts](http://www.gris.informatik.tu-darmstadt.de/~sroth/research/foe/index.html)模型实现图像去噪。
5. [nist.cc](https://ceres-solver.googlesource.com/ceres-solver/+/master/examples/nist.cc)构造并求解了[NIST](http://www.itl.nist.gov/div898/strd/nls/nls_main.shtml)非线性回归问题。
6. [more_garbow_hillstrom.cc](https://ceres-solver.googlesource.com/ceres-solver/+/master/examples/more_garbow_hillstrom.cc)是下述论文中测试问题的子集：
    - Testing Unconstrained Optimization Software Jorge J. More, Burton S. Garbow and Kenneth E. Hillstrom ACM Transactions on Mathematical Software, 7(1), pp. 17-41, 1981
    - A Trust Region Approach to Linearly Constrained Optimization David M. Gay Numerical Analysis (Griffiths, D.F., ed.), pp. 72-105 Lecture Notes in Mathematics 1066, Springer Verlag, 1984.这篇文章在前者的基础上增加了数据增强和有界约束。

7. [libmv_bundle_adjuster.cc](https://ceres-solver.googlesource.com/ceres-solver/+/master/examples/libmv_bundle_adjuster.cc)包含了Blender/libmv中使用的BA算法
8. [libmv_homography.cc](https://ceres-solver.googlesource.com/ceres-solver/+/master/examples/libmv_homography.cc)演示了如何求解两组点之间的homograph变换，使用了检查图像空间的回调函数作为设定的优化退出条件。
9. [robot_pose_mle.cc](https://ceres-solver.googlesource.com/ceres-solver/+/master/examples/robot_pose_mle.cc)展示了`DynamicAutoDiffCostFunction`的使用方法。`DynamicAutoDiffCostFunction`适用于参数块的数量或者尺寸在编译器无法确定的问题。在这个例子中仿真了一个机器人的运动轨迹，机器人在一维走廊中运动，输入数据是有噪声的里程计和有噪声的测距传感器数据。通过融合带噪声的里程计和传感器读数，这个例子展示了如何利用最大似然估计MLE来计算机器人在每一个时刻的姿态。
10. [slam/pose_graph_2d/pose_graph_2d.cc](https://ceres-solver.googlesource.com/ceres-solver/+/master/examples/slam/pose_graph_2d/pose_graph_2d.cc)同时定位和建图问题包括为未知环境构造地图以及同步估计相对地图的位姿两个部分。该问题的主要难点在于缺乏额外的辅助信息例如GPS。SLAM被认为是机器人领域的基础挑战之一。有一些可供参考的SLAM资料(Giorgio Grisetti, Rainer Kummerle, Cyrill Stachniss, Wolfram Burgard. A Tutorial on Graph-Based SLAM. IEEE Intelligent Transportation Systems Magazine, 52(3):199-222, 2010.) 位姿图优化问题是SLAM中的问题之一。下面的例子展示了如何使用相对位姿约束来构造2维空间的位姿图优化问题。

考虑一个机器人运动在2D空间之中。机器人配备有一系列的传感器，例如轮速里程计或者一个激光测距传感器。从这些原始数据中，我们想估计机器人的轨迹同时构建环境的地图。为了减少这个问题的计算复杂度，这个位姿图优化方法在原始测量值上进行了一些处理。它创建了一个包含若干节点的图，每一个节点代表机器人的的姿态，节点之间的边代表两个节点位姿之间的相对变换。这些边是通过传感器原始数据计算得到的，比如对轮速里程计积分以及通过对齐激光扫描线等方法。这个构造出的图的可视化结果如下：

![slam problem 2d](pics/slam2d.png)

这个图里面三角形的是机器人的位姿，连接线代表测量值，虚线代表回环检测得到的测量值。回环检测是非连续序列的节点之前的测量值，它可以减少随时间累计的误差。下面将使用数学公式来描述位姿图优化问题。

机器人在t时刻的状态$x_t= [p^T, \psi]^T$, 其中p是一个2D向量表示平面坐标系中的位置，`\psi`代表以弧度为单位的朝向。不同时刻a和b之间的相对变换测量值为$z_{ab} = [\hat{p}_{ab}^T, \hat{\psi}_{ab}]$Ceres的CostFunction中计算了观测值和估计值之间的残差：

$$
\begin{split}r_{ab} =
\left[
\begin{array}{c}
  R_a^T\left(p_b - p_a\right) - \hat{p}_{ab} \\
  \mathrm{Normalize}\left(\psi_b - \psi_a - \hat{\psi}_{ab}\right)
\end{array}
\right]\end{split}
$$

里面的函数`Normalize()`对角度归一化至$[-\pi,\pi)$, R代表旋转矩阵

$$
\begin{split}R_a =
\left[
\begin{array}{cc}
  \cos \psi_a & -\sin \psi_a \\
  \sin \psi_a & \cos \psi_a \\
\end{array}
\right]\end{split}
$$

为了完成CostFunction。我们需要给每一个残差一个权重来表示测量的不确定性。因此，我们给残差项提前乘了测量值协方差矩阵方根值的逆。即$\Sigma_{ab}^{-\frac{1}{2}} r_{ab}$,其中$\Sigma_{ab}$是协方差矩阵。

最后，我们使用流型来归一化范围为$[-\pi,\pi)$的朝向角。特别的，我们定义了`AngleManifold::Plus()`函数代表$\mathrm{Normalize}(\psi + \Delta)$以及`::member::AngleManifold::Minus() `代表$\mathrm{Normalize}(y) - \mathrm{Normalize}(x)$

这个代码包里包括了一个可执行程序`pose_graoh_2d`来读取问题描述文件。这个程序可以处理任何以g2o格式定义的2D问题。实现一个新的读取类就可以读取不同类型的数据了。`pose_graph_2d`会打印出来ceres solver全部的结果然后将原始位姿和优化后的位姿保存到硬盘上。保存文件的格式为`pose_id x y yaw_radians`

`pose_id`是对应的ID序号。保存的文件使用pose_id的升序排序。

执行的命令是

```sh
/path/to/bin/pose_graph_2d /path/to/dataset/dataset.g2o
```

用python脚本可以可视化结果

```sh
/path/to/repo/examples/slam/pose_graph_2d/plot_result.py --optimized_poses ./poses_optimized.txt --initial_poses ./poses_original.txt
```

作为示例，一个Edwin Olson制作的标准合成数据集基准包括3500个节点和总计5598条边。使用脚本可以可视化优化结果：

![](./pics/manhattan_3500_result.png)

原始位姿为绿色，优化后的位姿为蓝色。可以看到，优化后的位姿更接近网格坐标系。可以注意到，图的左半边有一些轻微的角度误差，这是由于在这边缺乏足够的信息来构建约束。

11. [slam/pose_graph_3d/pose_graph_3d.cc](https://ceres-solver.googlesource.com/ceres-solver/+/master/examples/slam/pose_graph_3d/pose_graph_3d.cc)这个示例展示了如何用相对位姿约束构建3D的位姿图优化问题。这个例子也说明了如何使用Eigen的几何模块结合Ceres的自动求导功能。


机器人在t时刻的状态$x_t= [p^T, q^T]^T$, 其中p是一个3D向量表示坐标系中的位置, q是用Eigen四元数表示的姿态。。不同时刻a和b之间的相对变换测量值为$z_{ab} = [\hat{p}_{ab}^T, \hat{q}_{ab}^T]$Ceres的CostFunction中计算了观测值和估计值之间的残差：

$$
\begin{split}r_{ab} =
\left[
\begin{array}{c}
   R(q_a)^{T} (p_b - p_a) - \hat{p}_{ab} \\
   2.0 \mathrm{vec}\left((q_a^{-1} q_b) \hat{q}_{ab}^{-1}\right)
\end{array}
\right]\end{split}
$$

式中函数$vec()$返回四元数的向量部分（虚部），$R(q)$这是四元数对应的旋转矩阵。

为了完成CostFunction。我们需要给每一个残差一个权重来表示测量的不确定性。因此，我们给残差项提前乘了测量值协方差矩阵方根值的逆。即$\Sigma_{ab}^{-\frac{1}{2}} r_{ab}$,其中$\Sigma_{ab}$是协方差矩阵。

考虑到我们使用四元数来表示姿态，我们需要一个流型（`EigenQuaternionManifold`）来为四元数的4维向量沿正交方向更新数值。Eigen的四元数使用了不同的内存布局，Eigen按照$[x,y,z,w]$的顺序存储，而一般的做法是把实部放在前面。需要注意，在构造Eigen的四元数时，变量需要按照$w,x,y,z$的顺序提供。由于Ceres直接使用原生的double指针来访问参数块，这部分尤为需要注意。

这个代码包里包括了一个可执行程序`pose_graoh_3d`来读取问题描述文件。这个程序可以处理任何以g2o格式定义的2D问题。实现一个新的读取类就可以读取不同类型的数据了。`pose_graph_3d`会打印出来ceres solver全部的结果然后将原始位姿和优化后的位姿保存到硬盘上。保存文件的格式为`pose_id x y z q_x q_y q_z q_w`

`pose_id`是对应的ID序号。保存的文件使用pose_id的升序排序。

执行的命令是

```sh
/path/to/bin/pose_graph_3d /path/to/dataset/dataset.g2o
```

用python脚本可以可视化结果.使用`--axes_equal`选项可以启动等轴距坐标系。

```sh
/path/to/repo/examples/slam/pose_graph_3d/plot_result.py --optimized_poses ./poses_optimized.txt --initial_poses ./poses_original.txt
```

下面给出的示例使用了一个合成数据集，机器人在一个球的表面运动，包含2500个位姿节点以及4949条边。使用脚本可以可视化结果：

![](pics/pose_graph_3d_ex.png)

