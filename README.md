# Statistical Learning Models Implementations

> 从零实现经典统计学习模型 —— 手撕《统计学习方法》（李航）

本项目使用 `numpy` / `matplotlib` / `rich`，纯手工实现了《统计学习方法》中一系列经典机器学习模型。每个算法都从数学原理出发亲自实现，力求把「造轮子」的过程和推导讲清楚。

每个模型文件底部都带有 `if __name__ == "__main__":` 的可运行示例，直接运行即可看到训练过程、可视化图像与预测结果表格，非常适合对照书本学习。

---

## 目录

- [模型一览](#模型一览)
- [环境依赖](#环境依赖)
- [快速开始](#快速开始)
- [项目结构](#项目结构)
- [算法与实现对照](#算法与实现对照)
- [说明](#说明)

---

## 模型一览

|   章节   | 模型                             | 文件                                                       | 核心方法                                            |
| :------: | :------------------------------- | :--------------------------------------------------------- | :-------------------------------------------------- |
| 第 2 章  | 感知机 Perceptron                | [`perceptron.py`](perceptron.py)                           | 随机梯度下降 / 误分类驱动更新                       |
| 第 3 章  | K 近邻 KNN（暴力）               | [`knn.py`](knn.py)                                         | 线性扫描 + 距离度量                                 |
| 第 3 章  | K 近邻 KNN（KD 树）              | [`knn_kdtree.py`](knn_kdtree.py)                           | KD-Tree 构建与最近邻回溯搜索                        |
| 第 4 章  | 朴素贝叶斯 Naive Bayes           | [`naive_bayes.py`](naive_bayes.py)                         | 极大似然估计 (MLE) 与贝叶斯估计 (MAP，拉普拉斯平滑) |
| 第 5 章  | 决策树 ID3 / C4.5                | [`desition_tree_ID3_C45.py`](desition_tree_ID3_C45.py)     | 信息增益 / 信息增益比 + 剪枝                        |
| 第 5 章  | 决策树 CART                      | [`decision_tree_CART.py`](decision_tree_CART.py)           | 分类树 (Gini) / 回归树 (平方误差) + 代价复杂度剪枝  |
| 第 6 章  | 逻辑斯谛回归 Logistic Regression | [`logistic_regression.py`](logistic_regression.py)         | Sigmoid + 对数似然梯度上升                          |
| 第 6 章  | 最大熵模型 Max Entropy           | [`max_entropy.py`](max_entropy.py)                         | BFGS 拟牛顿法 + 线搜索                              |
| 第 7 章  | 支持向量机 SVM                   | [`support_vector_machine.py`](support_vector_machine.py)   | SMO 算法 + 核函数（线性核 / 高斯核）                |
| 第 8 章  | 提升方法 AdaBoost                | [`adaboost.py`](adaboost.py)                               | 决策桩 (Decision Stump) + 加权提升                  |
| 第 8 章  | 梯度提升决策树 GBDT              | [`GBDT.py`](GBDT.py)                                       | 回归 CART 拟合负梯度 + 线搜索叶子取值               |
| 第 9 章  | EM · 高斯混合模型 GMM            | [`em_GaussianMixtureModel.py`](em_GaussianMixtureModel.py) | EM 迭代（对角 / 全协方差）                          |
| 第 10 章 | 隐马尔可夫模型 HMM               | [`Hidden_Markov_Model_HMM.py`](Hidden_Markov_Model_HMM.py) | 前向后向 + Viterbi + Baum-Welch                     |

> HMM 拆分为 4 个文件，分别对应不同的子算法，详见 [项目结构](#项目结构)。

---

## 环境依赖

- Python >= 3.8（开发环境为 Python 3.8.20）
- 第三方库：`text` `numpy` `matplotlib` `rich`
- 安装依赖：`pip install numpy matplotlib rich`

---

## 项目结构

```text
statistical_learning_models_inplementaions/ 
├── perceptron.py # 感知机 
├── knn.py # KNN（暴力搜索） 
├── knn_kdtree.py # KNN（KD 树加速） 
├── naive_bayes.py # 朴素贝叶斯（MLE / MAP） 
├── desition_tree_ID3_C45.py # 决策树 ID3 / C4.5 + 剪枝 
├── decision_tree_CART.py # CART（分类树 / 回归树 / 剪枝树） 
├── logistic_regression.py # 逻辑斯谛回归 
├── max_entropy.py # 最大熵模型（BFGS） 
├── support_vector_machine.py # 支持向量机（SMO） 
├── adaboost.py # AdaBoost（决策桩） 
├── GBDT.py # 梯度提升决策树 
├── em_GaussianMixtureModel.py # EM 高斯混合模型 
├── Hidden_Markov_Model_HMM.py # HMM 主类（训练 + 预测封装） 
├── Hidden_Markov_Model_forward_backward.py # HMM 前向 / 后向算法 
├── Hidden_Markov_Model_Viterbi.py # HMM Viterbi 解码 
├── Hidden_Markov_Model_Baum_Welch.py # HMM Baum-Welch 训练（EM） 
├── utils.py # 公共工具函数 
├── test.py # KD 树单元测试 
└── README.md
```

#### 公共工具 `utils.py`

集中实现了各模型复用的基础组件：

- **损失 / 度量**：`entropy`（熵）、`gini`（基尼指数）、`information_gain` / `information_gain_ratio`（信息增益 / 增益比）、`euc_dis`（欧氏距离）、`binary_cross_entropy`（二元交叉熵）
- **激活 / 概率**：`sigmoid`、`softmax`
- **数值优化**：`line_search`（黄金分割线搜索，供 BFGS、GBDT 使用）
- **数据结构**：`Heap`（带最大长度限制的堆，供 KD 树最近邻搜索使用）
- **可视化**：`wbline` / `kbline`（根据权重或斜率截距绘制分类超平面）
- **通用**：`argmax` / `argmin`

---

## 算法与实现对照

| 算法       | 训练方法               | 关键实现细节                                                |
| :--------- | :--------------------- | :---------------------------------------------------------- |
| 感知机     | 误分类点驱动的参数更新 | 演示了无法解决线性不可分问题（XOR）                         |
| KNN-KD 树  | 空间划分建树           | 中位数切分 + 回溯剪枝，等价于书 3.3 节                      |
| 朴素贝叶斯 | 概率统计               | MLE 与带拉普拉斯平滑的 MAP 两种估计对比                     |
| 决策树     | 递归特征选择           | ID3（信息增益）与 C4.5（增益比）可切换，支持后剪枝          |
| CART       | 递归二叉划分           | 分类用 Gini、回归用平方误差，含代价复杂度剪枝与验证集选阈值 |
| 逻辑回归   | 梯度上升最大化对数似然 | 演示非严格线性可分场景                                      |
| 最大熵     | BFGS 拟牛顿法          | 特征函数期望约束 + 黄金分割线搜索                           |
| SVM        | SMO 序列最小优化       | KKT 违背选点、双变量更新、可插拔核函数                      |
| AdaBoost   | 加权集成               | 决策桩基分类器 + 样本权重与模型权重更新                     |
| GBDT       | 拟合负梯度             | 基于回归 CART，叶子节点用线搜索求最优取值                   |
| EM-GMM     | E 步 / M 步迭代        | 支持对角协方差与全协方差，含对数似然早停                    |
| HMM        | Baum-Welch 无监督训练  | 前向后向计算状态概率，Viterbi 解码最优状态序列              |

---

## 说明

- 本项目以**学习和教学**为目的，代码力求清晰易懂，未针对大规模数据和工程性能做优化，实际生产请使用 `scikit-learn` 等成熟库。
- 模型实现主要参考 **李航《统计学习方法》**，示例数据多取自书中例题与习题（例如决策树使用了「青年/老年、是/否」等中文示例）。
- 部分随机初始化的模型（如 SVM、GMM、逻辑回归）每次运行结果可能不同，属于正常现象。