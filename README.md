# HIT-PatternRecognition
#### 本实验选择基于CLIP模型来识别是否该将社交媒体上的某张图片标记为“敏感的”
本实验使用公开可用的数据集UCF-Crime(标记了犯罪行为与正常生活的监控录像)、秘鲁新闻中暴动抗议图片以及在微博平台利用爬虫爬取的正常生活照片。

在数据集中，将每个图片与它的敏感度标签相关联。

train.py和test.py分别为训练模型与测试模型的代码，已经训练好的模型保存为model.pkl，运行test.py时读取该模型并测试，无需再次训练。

这两个模考分别集成train_model()与test_model()两个接口以便调用。

如果想重新训练，则将main.py中的train.train_model()取消注释运行即可。
