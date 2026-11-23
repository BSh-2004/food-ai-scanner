import numpy as np
import torch
import torch.nn as nn

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

LABEL_NAMES = {
    0: "🟢 绿灯（推荐日常选择）",
    1: "🟡 黄灯（适量关注/注意频次）",
    2: "🔴 红灯（高糖高脂/精加工慎选）"
}


def parse_ingredients(text: str):
    """清洗括号并按常见分隔符切分配料列表"""
    cleaned = (
        text.replace("（", "、")
        .replace("）", "、")
        .replace("(", "、")
        .replace(")", "、")
    )
    for sep in ["，", ",", "；", ";", "\n"]:
        cleaned = cleaned.replace(sep, "、")
    return [t.strip() for t in cleaned.split("、") if t.strip()]


def classify_token_level(token: str) -> int:
    """单条配料风险等级：2=红灯，1=黄灯，0=绿灯"""
    if any(k in token for k in RED_TOKENS):
        return 2
    if any(k in token for k in YELLOW_TOKENS):
        return 1
    return 0


def extract_features(ingredients_text: str) -> np.ndarray:
    """提取与训练阶段一致的 6 维特征向量"""
    tokens = parse_ingredients(ingredients_text)
    total = max(len(tokens), 1)
    levels = [classify_token_level(tok) for tok in tokens]

    red_cnt = sum(1 for lv in levels if lv == 2)
    yellow_cnt = sum(1 for lv in levels if lv == 1)
    top3 = levels[:3]
    top3_red = sum(1 for lv in top3 if lv == 2) / 3.0
    top3_green = sum(1 for lv in top3 if lv == 0) / max(len(top3), 1)
    norm_len = min(total / 15.0, 1.0)
    sugar_fat_flag = (
        1.0
        if any(
            any(w in tok for w in ["糖", "浆", "植脂末", "氢化", "奶油", "起酥油", "代可可脂"])
            for tok in tokens[:5]
        )
        else 0.0
    )

    return np.array(
        [red_cnt / total, yellow_cnt / total, top3_red, top3_green, norm_len, sugar_fat_flag],
        dtype=np.float32,
    )


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
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.net(x)