# CT Motion Artifact Classification

![Python](https://img.shields.io/badge/python-3.10-blue)
![PyTorch Lightning](https://img.shields.io/badge/pytorch--lightning-2.0-blue)
![License](https://img.shields.io/badge/license-MIT-green)

---

## 项目简介

本项目旨在通过深度学习对 CT 图像进行伪影（Motion Artifact）分类，判断图像质量为 **良好 (good)** 或 **伪影 (bad)**。

* 使用 **PyTorch Lightning** 构建训练流程
* 支持 **单次训练** 与 **k-fold 交叉验证**
* 支持灰度 CT 图像（单通道）扩展至三通道以使用预训练卷积神经网络
* SwanLab 可视化训练日志和模型性能

---

## 项目结构

```
CT-MAR/
├── main.py                  # 训练与测试入口
├── data_interface.py        # LightningDataModule 封装数据集
├── model/
│   ├── model_interface.py   # LightningModule 封装模型
│   ├── classifier.py        # 分类器网络定义
│   └── backbone/            # Backbone 网络（ResNet 等）
├── data/
│   ├── labels.csv      # CSV 文件：图像路径 + 标签
│   └── images/         # DICOM 图像存放目录
├── CT_MotionArtifact/             # 模型权重保存目录
├── requirements.txt         # Python 依赖
└── README.md
```

> data/需要自己创建，CT_MotionArtifact在运行训练代码后会自己生成。

---

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 数据准备

在项目根目录创建一个data文件夹，将数据集放入./data/。（数据集包括labels.csv和images/）

CSV 文件格式：

```
data/exam_images/00001.dcm,good
data/exam_images/00002.dcm,good
...
```

图片存放目录与 CSV 中路径对应。

### 3. 训练模型

```bash
python main.py --mode train --csv_file ./data/labels.csv --batch_size 16 --epochs 100
```

支持 **k-fold 交叉验证**：

```bash
python main.py --mode train --csv_file ./data/labels.csv --k_fold 5 --epochs 100
```

### 4. 数据增强

支持增强策略 `--augment none|light|strong`

```bash
python main.py --mode train --csv_file ./data/labels.csv
```

---

## 模型与训练

* 默认 Backbone: `resnet18`（灰度图自动扩展为三通道）
* 分类器输出 2 类（good / bad）
* Lightning 内置 **EarlyStopping** 与 **ModelCheckpoint**
* 使用 **SwanLab** 可视化训练日志

---

## 输出

* 最佳模型保存在 `CT_MotionArtifact/`
* 测试集准确率及交叉验证结果自动打印
* SwanLab 提供在线训练记录和指标分析

---

## License

MIT License
