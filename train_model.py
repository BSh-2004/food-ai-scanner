import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

# 固定随机种子，保证结果可复现
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# 常见配料词库（按风险特征分组）
RED_TOKENS = [
    "白砂糖", "果葡糖浆", "葡萄糖浆", "植脂末", "氢化植物油",
    "人造奶油", "起酥油", "代可可脂", "阿斯巴甜", "三氯蔗糖", "亚硝酸钠"
]
YELLOW_TOKENS = [
    "植物油", "食用盐", "麦芽糊精", "食用香精", "谷氨酸钠",
    "山梨酸钾", "脱氢乙酸钠", "丙酸钙", "焦糖色", "柠檬酸", "乳化剂"
]
GREEN_TOKENS = [
    "生牛乳", "水", "乌龙茶", "茶叶", "小麦粉", "马铃薯",
    "全蛋液", "全脂乳粉", "酵母", "维生素C", "燕麦", "荞麦"
]


def parse_ingredients(text: str):
    """清洗括号并按常见分隔符切分配料列表"""
    cleaned = text.replace("（", "、").replace("）", "、").replace("(", "、").replace(")", "、")
    tokens = [t.strip() for t in cleaned.replace("，", "、").replace(",", "、").replace("；", "、").split("、")]
    return [t for t in tokens if t]


def extract_features(ingredients_text: str) -> np.ndarray:
    """
    从配料表中提取 6 维工程特征（体现 GB 7718 顺位递减原则）：
    1. red_ratio: 红灯配料占比
    2. yellow_ratio: 黄灯配料占比
    3. top3_red_count: 前 3 位配料中红灯成分个数（归一化到 [0, 1]）
    4. top3_green_ratio: 前 3 位配料中绿灯成分占比
    5. norm_length: 配料表长度（归一化，反映深加工复杂度）
    6. sugar_fat_flag: 是否含高风险糖/精炼油脂特征（0 或 1）
    """
    tokens = parse_ingredients(ingredients_text)
    total = max(len(tokens), 1)

    levels = []
    for tok in tokens:
        if any(k in tok for k in RED_TOKENS):
            levels.append(2)
        elif any(k in tok for k in YELLOW_TOKENS):
            levels.append(1)
        else:
            levels.append(0)

    red_cnt = sum(1 for lv in levels if lv == 2)
    yellow_cnt = sum(1 for lv in levels if lv == 1)
    top3 = levels[:3]
    top3_red = sum(1 for lv in top3 if lv == 2) / 3.0
    top3_green = sum(1 for lv in top3 if lv == 0) / max(len(top3), 1)
    norm_len = min(total / 15.0, 1.0)
    sugar_fat_flag = 1.0 if any(
        any(w in tok for w in ["糖", "浆", "植脂末", "氢化", "奶油", "起酥油", "代可可脂"])
        for tok in tokens[:5]
    ) else 0.0

    return np.array(
        [red_cnt / total, yellow_cnt / total, top3_red, top3_green, norm_len, sugar_fat_flag],
        dtype=np.float32
    )


def generate_synthetic_dataset(num_samples: int = 1500):
    """构造带标签的食品配料训练样本（0: 绿灯, 1: 黄灯, 2: 红灯）"""
    X_list, y_list = [], []
    for _ in range(num_samples):
        n_items = random.randint(1, 14)
        # 随机决定样本偏向哪一类风格
        style = random.choice(["clean", "moderate", "heavy"])
        chosen = []
        for idx in range(n_items):
            r = random.random()
            if style == "clean":
                if r < 0.75:
                    chosen.append(random.choice(GREEN_TOKENS))
                elif r < 0.95:
                    chosen.append(random.choice(YELLOW_TOKENS))
                else:
                    chosen.append(random.choice(RED_TOKENS))
            elif style == "moderate":
                if idx < 2 and r < 0.55:
                    chosen.append(random.choice(GREEN_TOKENS))
                elif r < 0.55:
                    chosen.append(random.choice(YELLOW_TOKENS))
                elif r < 0.82:
                    chosen.append(random.choice(GREEN_TOKENS))
                else:
                    chosen.append(random.choice(RED_TOKENS))
            else:  # heavy
                if idx < 4 and r < 0.55:
                    chosen.append(random.choice(RED_TOKENS))
                elif r < 0.45:
                    chosen.append(random.choice(RED_TOKENS))
                elif r < 0.80:
                    chosen.append(random.choice(YELLOW_TOKENS))
                else:
                    chosen.append(random.choice(GREEN_TOKENS))

        text = "、".join(chosen)
        feat = extract_features(text)

        # 基于特征给出监督标签（加入轻微噪声模拟真实标注差异）
        risk_score = feat[2] * 2.2 + feat[0] * 2.0 + feat[5] * 1.1 + feat[1] * 0.7 + feat[4] * 0.4 - feat[3] * 1.2
        risk_score += np.random.normal(0, 0.15)

        if risk_score >= 1.35:
            label = 2  # 红灯
        elif risk_score >= 0.45:
            label = 1  # 黄灯
        else:
            label = 0  # 绿灯

        X_list.append(feat)
        y_list.append(label)

    return torch.tensor(np.array(X_list)), torch.tensor(np.array(y_list), dtype=torch.long)


class FoodRiskNet(nn.Module):
    """轻量级食品配料风险分类 MLP 网络"""
    def __init__(self, in_dim: int = 6, hidden_dim: int = 32, num_classes: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        return self.net(x)


def main():
    os.makedirs("artifacts", exist_ok=True)
    X, y = generate_synthetic_dataset(num_samples=1800)
    dataset = TensorDataset(X, y)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FoodRiskNet(in_dim=6, hidden_dim=32, num_classes=3).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-4)

    epochs = 40
    train_losses, val_losses, val_accs = [], [], []

    for epoch in range(1, epochs + 1):
        model.train()
        batch_losses = []
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())

        train_losses.append(np.mean(batch_losses))

        # 验证
        model.eval()
        v_loss, correct, total = 0.0, 0, 0
        all_preds, all_targets = [], []
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                out = model(bx)
                v_loss += criterion(out, by).item() * by.size(0)
                preds = out.argmax(dim=1)
                correct += (preds == by).sum().item()
                total += by.size(0)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(by.cpu().numpy())

        val_losses.append(v_loss / total)
        val_accs.append(correct / total)

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch [{epoch:02d}/{epochs}] | Train Loss: {train_losses[-1]:.4f} | Val Loss: {val_losses[-1]:.4f} | Val Acc: {val_accs[-1]*100:.2f}%")

    # 保存模型权重
    weight_path = os.path.join("artifacts", "food_risk_net.pth")
    torch.save(model.state_dict(), weight_path)
    print(f"\n模型权重已保存至: {weight_path}")

    # 打印测试集分类指标
    print("\n--- 验证集分类报告 ---")
    print(classification_report(all_targets, all_preds, target_names=["绿灯(推荐)", "黄灯(适量)", "红灯(慎选)"]))
    print("混淆矩阵:\n", confusion_matrix(all_targets, all_preds))

    # 绘制 Loss 曲线图，方便直接放进实验报告第 4 章
    plt.rcParams["font.sans-serif"] = ["SimHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.figure(figsize=(8, 4.5))
    plt.plot(range(1, epochs + 1), train_losses, label="Train Loss")
    plt.plot(range(1, epochs + 1), val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("CrossEntropy Loss")
    plt.title("PyTorch 食品配料风险模型训练曲线")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    curve_path = os.path.join("artifacts", "training_loss.png")
    plt.savefig(curve_path, dpi=150)
    print(f"训练曲线图已保存至: {curve_path}")


if __name__ == "__main__":
    main()