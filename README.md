# HW1: Fashion-MNIST NumPy MLP

这是一个按作业要求实现的三层神经网络分类器项目，使用 `NumPy` 从零完成前向传播、自动微分、反向传播、`SGD` 优化、学习率衰减、交叉熵损失和 `L2` 正则化，并在 `Fashion-MNIST` 上训练与评估。

## 功能概览

- 三层 `MLP`：`784 -> hidden -> hidden -> 10`
- 支持 `relu`、`sigmoid`、`tanh` 三种激活函数
- 支持训练集 / 验证集 / 测试集完整流程
- 支持自动保存验证集表现最好的模型
- 支持超参数网格搜索
- 支持输出混淆矩阵、第一层权重可视化、错分样本可视化

## 环境依赖

```bash
python -m pip install numpy matplotlib
```

## 文件说明

- `autograd.py`：轻量级自动微分引擎
- `data_utils.py`：Fashion-MNIST 下载、解析、划分与批处理
- `mlp_model.py`：三层 MLP 定义
- `optim.py`：SGD 优化器
- `train.py`：训练脚本
- `evaluate.py`：测试与评估脚本
- `search.py`：超参数网格搜索
- `visualize.py`：曲线、权重、混淆矩阵、错分样本可视化

## 训练

```bash
python train.py --output-dir outputs/baseline --hidden-dim 256 --activation relu --epochs 20 --lr 0.05 --weight-decay 1e-4
```

训练完成后会在 `outputs/baseline` 下生成：

- `best_model.npz`
- `train_config.json`
- `loss_curve.png`
- `val_accuracy_curve.png`
- `first_layer_weights.png`

## 测试

```bash
python evaluate.py --checkpoint outputs/baseline/best_model.npz --output-dir outputs/eval
```

测试完成后会生成：

- `metrics.json`
- `confusion_matrix.png`
- `misclassified_samples.png`

## 超参数搜索

```bash
python search.py --output-root outputs/search --hidden-dims 128 256 512 --activations relu tanh --learning-rates 0.05 0.01 --weight-decays 0.0 1e-4
```

搜索结果会保存在 `outputs/search/search_results.json`。

## 报告建议

实验报告里建议包含以下内容：

- 模型结构与自动微分实现思路
- 数据预处理和训练设置
- 训练集 / 验证集 `Loss` 曲线
- 验证集 `Accuracy` 曲线
- 不同超参数组合的结果对比
- 第一层权重可视化分析
- 测试集混淆矩阵分析
- 错分样本分析

## 说明

首次运行时脚本会自动下载 `Fashion-MNIST` 原始 `gz` 文件到 `data/` 目录。

## 链接

- GitHub Repo: `https://github.com/jiab666/cvhw`
- Best Model Weight: `https://github.com/jiab666/cvhw/blob/main/outputs/final_run/best_model.npz`
