import torch
import numpy as np
import clip
from sklearn.svm import LinearSVC
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
import pickle

def train_model():
    # 加载 CLIP 模型
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    model.eval().to(device)

    # 准备数据集
    transform = transforms.Compose([
        transforms.Resize((224, 224), antialias=True),
        preprocess
    ])
    train_dataset = ImageFolder("./data/train", transform=transform)
    print(train_dataset.class_to_idx)

    # 提取特征向量
    features = []
    for img, _ in train_dataset:
        with torch.no_grad():
            img = img.unsqueeze(0).to(device)
            feature = model.encode_image(img)
            features.append(feature.cpu().numpy()[0])


    # 训练分类器
    X_train = np.array(features)
    y_train = np.array([label for _, label in train_dataset])
    clf = LinearSVC(C=0.1)
    clf.fit(X_train, y_train)
    print("训练完成")

    with open('model.pkl', 'wb') as f:
        pickle.dump(clf, f)
