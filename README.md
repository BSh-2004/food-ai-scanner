# 🚦 AI + 食品健康：食品配料表“红绿灯”智能体检器

《人工智能基础与应用》课程大作业项目。本项目结合 **国标 GB 7718（配料顺位原则）** 与 **GB 28050（营养素参考值 NRV%）**，采用“**PyTorch 离线训练推理 + 轻量级交互 Web 前端**”双层架构，实现预包装食品配料健康分级、吃/喝同类健康替代推荐、营养成分表负荷换算、两款食品横向 PK 及本地历史记录统计。

---

## 🌐 一、在线访问（GitHub Pages）

- **在线演示地址**：`https://<你的用户名>.github.io/<你的仓库名>/`
- **源代码仓库地址**：`https://github.com/<你的用户名>/<你的仓库名>`

> 打开在线网页即可直接使用预设样本（自动加载 PyTorch 离线推理结果 `predictions.json`）与自定义配料/营养成分测算功能，无需安装任何本地依赖。

---

## 📁 二、项目目录结构

```text
food-ai-scanner/
├── index.html               # Web 前端主页面（部署至 GitHub Pages 的入口文件）
├── predictions.json         # 由 PyTorch 模型批量推理导出的预设食品预测结果
├── model_utils.py           # 配料分词清洗、6 维领域特征提取与 FoodRiskNet 网络定义
├── train_model.py           # 合成数据集构建、PyTorch 模型训练与损失曲线绘制脚本
├── predict.py               # 加载权重进行单品推理与命令行交互测试脚本
├── export_predictions.py    # 批量运行 PyTorch 推理并更新 predictions.json 的脚本
├── README.md                # 项目简介与环境配置、运行说明
└── artifacts/
    ├── food_risk_net.pth    # 训练保存的 PyTorch 模型权重文件
    └── training_loss.png    # 训练过程损失收敛曲线图
```

---

## ⚙️ 三、本地 Python / PyTorch 环境配置

### 1. 创建并激活 Anaconda 虚拟环境

确保电脑已安装 **Anaconda** 或 **Miniconda**，打开终端（Anaconda Prompt 或 PyCharm Terminal）执行：

```bash
conda create -n food-ai python=3.11 -y
conda activate food-ai
```

### 2. 安装依赖包

安装 PyTorch（CPU 版即可快速完成训练与推理）及科学计算、绘图库：

```bash
pip install torch numpy matplotlib
```

---

## 🚀 四、完整运行步骤

在项目根目录（确保已激活 `food-ai` 环境）按顺序执行以下命令：

### 步骤 1：训练 PyTorch 风险分类模型

```bash
python train_model.py
```

- **执行说明**：脚本将基于三档配料词库与 GB 7718 顺位规则构建合成训练集与验证集，训练三分类多层感知机 `FoodRiskNet`（输入 6 维特征，输出绿灯/黄灯/红灯概率）。
- **输出结果**：在 `artifacts/` 目录下生成模型权重 `food_risk_net.pth` 与收敛曲线 `training_loss.png`。

### 步骤 2：命令行推理与交互测试

```bash
python predict.py
```

- **执行说明**：加载 `artifacts/food_risk_net.pth`（已启用 `weights_only=True` 安全加载），输出多款典型食品的预测等级、三类概率分布及配料顺位拆解标签，并支持在终端手动输入任意配料表进行实时测试。

### 步骤 3：导出推理结果供网页调用

```bash
python export_predictions.py
```

- **执行说明**：对预置的 10 款代表性食品批量执行 PyTorch 推理，生成结构化数据文件 `predictions.json`。

### 步骤 4：本地预览 Web 网页

由于浏览器安全策略限制，直接双击 `index.html`（`file://` 协议）可能无法通过 `fetch` 读取同级目录下的 `predictions.json`。建议在项目根目录启动 Python 内置轻量级 HTTP 服务器：

```bash
python -m http.server 8000
```

随后在浏览器访问 **`http://localhost:8000`** 即可完整体验所有功能（按 `Ctrl + C` 可停止本地服务器）。

---

## 🧩 五、核心功能说明

1. **单品配料体检**：支持预设样本的 PyTorch 推理结果展示与自定义配料表实时评估，逐项标注 `#顺位` 与 `[绿/黄/红]` 风险等级。
2. **吃/喝同类健康替代推荐**：自动判别固态点心/零食与液态饮品形态，匹配对应场景下的清爽替代方案。
3. **营养成分表 NRV% 换算**：依据 GB 28050 标准折算千卡热量与五大营养素每日参考值占比，对能量/脂肪/碳水/钠偏高及蛋白质偏少（`< 5% NRV`）给出预警提示。
4. **两款食品对比 & 本地历史统计**：支持双品配料横向 PK，并利用浏览器 `localStorage` 自动累计绿/黄/红检测次数与最近 20 条检测记录。

---

## ⚠️ 六、免责声明

本项目仅用于《人工智能基础与应用》课程实验教学与食品标签科普展示，模型训练数据包含基于国标规则构建的合成样本，输出结果仅供日常膳食参考，不构成任何医学或临床营养诊疗建议。
