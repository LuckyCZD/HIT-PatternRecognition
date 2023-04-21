import torch
import clip
import numpy as np
from sklearn.metrics import accuracy_score, recall_score
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
from PIL import Image
import pickle

def test_model():
    # 加载 CLIP 模型
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    model.eval().to(device)

    # 准备数据集
    transform = transforms.Compose([
        transforms.Resize((224, 224), antialias=True),
        preprocess
    ])
    test_dataset = ImageFolder("./data/test", transform=transform)
    print(test_dataset.class_to_idx)

    with open('model.pkl', 'rb') as f:
        clf = pickle.load(f)

    # 获取标签
    y_true = np.array([label for _, label in test_dataset])
    y_pred = []

    # 测试分类器
    for image, _ in test_dataset:
        image_tensor = image.unsqueeze(0).to(device)
        test_feature = model.encode_image(image_tensor)
        test_feature = test_feature.detach().cpu().numpy()[0]
        test_prediction = clf.predict([test_feature])[0]
        y_pred.append(test_prediction)

    # 计算准确率
    accuracy = accuracy_score(y_true, y_pred)
    print("准确率：{:.2%}".format(accuracy))

    # 计算召回率
    recall = recall_score(y_true, y_pred)
    print("召回率：{:.2%}".format(recall))

