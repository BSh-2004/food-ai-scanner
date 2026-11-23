import os
import torch
from model_utils import (
    FoodRiskNet,
    LABEL_NAMES,
    classify_token_level,
    extract_features,
    parse_ingredients,
)


def load_trained_model(weight_path: str = "artifacts/food_risk_net.pth", device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FoodRiskNet(in_dim=6, hidden_dim=32, num_classes=3).to(device)
    state_dict = torch.load(weight_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, device


def predict_food(food_name: str, ingredients_text: str, model, device):
    tokens = parse_ingredients(ingredients_text)
    feat = extract_features(ingredients_text)
    tensor_x = torch.tensor(feat, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor_x)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
        pred_cls = int(probs.argmax())

    level_badges = {0: "[绿]", 1: "[黄]", 2: "[红]"}
    tagged_tokens = [f"#{i+1}{level_badges[classify_token_level(t)]}{t}" for i, t in enumerate(tokens)]

    print("=" * 60)
    print(f"食品名称：{food_name}")
    print(f"配料总数：{len(tokens)} 项 | 特征向量：{[round(float(v), 3) for v in feat]}")
    print(f"预测结果：{LABEL_NAMES[pred_cls]}")
    print(
        f"各类概率：绿灯 {probs[0]*100:.1f}% | "
        f"黄灯 {probs[1]*100:.1f}% | "
        f"红灯 {probs[2]*100:.1f}%"
    )
    print("配料拆解：", " -> ".join(tagged_tokens) if tagged_tokens else "无有效配料")

    if pred_cls == 2:
        print("健康建议：高风险或精加工配料占比较高/排位靠前，建议减少食用频次，注意控糖控脂。")
    elif pred_cls == 1:
        print("健康建议：存在一定调味成分或合规添加剂，日常可适量食用，注意搭配天然食材。")
    else:
        print("健康建议：配方相对简洁或基础原料占主导，整体负担较低，适合作为日常选择。")


def main():
    weight_path = os.path.join("artifacts", "food_risk_net.pth")
    if not os.path.exists(weight_path):
        print(f"未找到权重文件 {weight_path}，请先运行 train_model.py 完成训练！")
        return

    model, device = load_trained_model(weight_path)

    samples = [
        ("某品牌纯牛奶", "生牛乳"),
        ("无糖乌龙茶饮料", "水、乌龙茶茶叶、维生素C"),
        ("某瓶装风味奶茶", "水、白砂糖、植脂末（葡萄糖浆、氢化植物油）、全脂乳粉、食用香精"),
        ("某网红烧烤味薯片", "马铃薯、植物油、白砂糖、食用盐、谷氨酸钠、麦芽糊精、阿斯巴甜"),
    ]

    for name, ing in samples:
        predict_food(name, ing, model, device)

    # 支持在 PyCharm 控制台交互输入自定义配料表测试
    while True:
        user_ing = input("\n请输入要检测的新配料表（直接回车退出）：").strip()
        if not user_ing:
            break
        user_name = input("请输入食品名称（可选，直接回车默认“自定义食品”）：").strip() or "自定义食品"
        predict_food(user_name, user_ing, model, device)


if __name__ == "__main__":
    main()