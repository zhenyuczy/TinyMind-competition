This is my first time to attend a competition related to DL, I have spent nearly a week now. The current accuracy is about 98.5, and I think I can achieve an accuracy of more than 99.

这是我第一次参加与深度学习有关的比赛，现在已经做了大约一星期。目前准确度为98.5左右，我想自己努努力可以提到99+。

# Introduction
[TinyMind第一届汉字书法识别挑战赛](https://www.tinymind.cn/competitions/41)

# How to use it
- First you need to use the [split_dataset.ipynb](https://github.com/finalacm/Chinese-character-recognition/blob/master/split_dataset.ipynb) to get your training set and validation set
- Here I give an example [VGG16.ipynb](https://github.com/finalacm/Chinese-character-recognition/blob/master/VGG16.ipynb), you can try to train your model in a notebook.
- Finally, if you are not satisfied with the performance of a single model, use [ensemble.ipynb](https://github.com/finalacm/Chinese-character-recognition/blob/master/ensemble.ipynb) you can achieve better results.

# 如何使用
- 首先你需要使用[split_dataset.ipynb](https://github.com/finalacm/Chinese-character-recognition/blob/master/split_dataset.ipynb)去获取你的训练集和验证集
- 这里我给出一个训练例子[VGG16.ipynb](https://github.com/finalacm/Chinese-character-recognition/blob/master/VGG16.ipynb)的例子，你可以尝试训练你自己的网络
- 最后，如果你对单一模型的表现不满意，使用[ensemble.ipynb](https://github.com/finalacm/Chinese-character-recognition/blob/master/ensemble.ipynb)你可以得到更好的结果


# Architecture
- After the competition end, I will give my network architecture here.

# 架构
- 比赛结束后，我会给出我的网络架构

# Note
- Data agumentation and shuffle are very important
- You can use BatchNormalization or Dropout to improve your model
- Add regularization in some (not all) Fully-connected layer may be achieve better results 
- Decay learning rate when overfitting
- Anyway, it is important that try more and learn from the results

# 提示
- 数据增强以及抖动非常重要
- 你可以使用BN和Dropout来提升你的模型
- 在一些全连接层（不是所有）增加正则化可以获得更好的结果
- 在过拟合的时候记得衰减学习率
- 不管怎么样，多尝试并从结果中学习是非常重要的


