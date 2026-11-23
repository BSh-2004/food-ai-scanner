import json
import os
import torch
from model_utils import (
    classify_token_level,
    extract_features,
    parse_ingredients,
)
from predict import load_trained_model

SAMPLE_FOODS = [
    {
        "id": "milk",
        "name": "某品牌纯牛奶",
        "ingredients": "生牛乳",
    },
    {
        "id": "tea",
        "name": "无糖乌龙茶饮料",
        "ingredients": "水、乌龙茶茶叶、维生素C",
    },
    {
        "id": "milktea",
        "name": "某瓶装风味奶茶",
        "ingredients": "水、白砂糖、植脂末（葡萄糖浆、氢化植物油）、全脂乳粉、食用香精",
    },
    {
        "id": "chips",
        "name": "某网红烧烤味薯片",
        "ingredients": "马铃薯、植物油、白砂糖、食用盐、谷氨酸钠、麦芽糊精、阿斯巴甜",
    },
{
        "id": "soda",
        "name": "某原味无糖气泡水",
        "ingredients": "水、二氧化碳",
    },
    {
        "id": "wholewheat",
        "name": "某全麦夹心粗粮饼干",
        "ingredients": "小麦粉、白砂糖、植物油、全麦粉、起酥油、食用盐、碳酸氢钠、食用香精",
    },
    {
        "id": "lactic",
        "name": "某风味乳酸菌饮品",
        "ingredients": "水、白砂糖、全脂乳粉、果葡糖浆、柠檬酸、果胶、食用香精、嗜酸乳杆菌",
    },
    {
        "id": "chicken",
        "name": "某香辣即食鸡胸肉",
        "ingredients": "鸡胸肉、饮用水、酿造酱油、食用盐、白砂糖、香辛料、谷氨酸钠、呈味核苷酸二钠、山梨酸钾",
    },
    {
        "id": "strawberry",
        "name": "某冻干草莓脆片",
        "ingredients": "草莓",
    },
    {
        "id": "chocopie",
        "name": "某巧克力涂层夹心派",
        "ingredients": "小麦粉、白砂糖、植物油、鸡蛋、葡萄糖浆、代可可脂、麦芽糊精、甘油、单硬脂酸甘油酯、大豆磷脂、食用香精",
    },
    {
        "id": "bread",
        "name": "某夹心吐司面包",
        "ingredients": "小麦粉、水、白砂糖、人造奶油、全蛋液、果葡糖浆、食用盐、酵母、丙酸钙、脱氢乙酸钠",
    },
]


def main():
    weight_path = os.path.join("artifacts", "food_risk_net.pth")
    model, device = load_trained_model(weight_path)

    exported = []
    for item in SAMPLE_FOODS:
        tokens = parse_ingredients(item["ingredients"])
        feat = extract_features(item["ingredients"])
        tensor_x = torch.tensor(feat, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(tensor_x)
            probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
            pred_cls = int(probs.argmax())

        token_details = [
            {"name": tok, "rank": idx + 1, "level": classify_token_level(tok)}
            for idx, tok in enumerate(tokens)
        ]

        exported.append({
            "id": item["id"],
            "name": item["name"],
            "ingredients": item["ingredients"],
            "total_tokens": len(tokens),
            "features": [round(float(v), 3) for v in feat],
            "pred_class": pred_cls,
            "probs": {
                "green": round(float(probs[0]) * 100, 1),
                "yellow": round(float(probs[1]) * 100, 1),
                "red": round(float(probs[2]) * 100, 1),
            },
            "tokens": token_details,
        })

    with open("predictions.json", "w", encoding="utf-8") as f:
        json.dump(exported, f, ensure_ascii=False, indent=2)

    print("已成功导出 PyTorch 推理结果至 predictions.json")


if __name__ == "__main__":
    main()