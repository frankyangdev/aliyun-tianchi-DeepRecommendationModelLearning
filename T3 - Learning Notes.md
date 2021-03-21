#### [深度推荐模型之DeepFM](https://zhuanlan.zhihu.com/p/57873613)

#### [速览 DeepFM: 使用 FM 取代 Wide & Deep 中的 LR](https://zhuanlan.zhihu.com/p/57158486)

#### [CTR论文精读(七)--DeepFM](https://zhuanlan.zhihu.com/p/54776945)

#### CNN/RNN/FNN/PNN

* CNN模型的缺点是：偏向于学习相邻特征的组合特征。 

* RNN模型的缺点是：比较适用于有序列(时序)关系的数据。
* FNN (Factorization-machine supported Neural Network) 的提出，应该算是一次非常不错的尝试：先使用预先训练好的FM，得到隐向量，然后作为DNN的输入来训练模型。缺点在于：受限于FM预训练的效果。
* PNN (Product-based Neural Network)，PNN为了捕获高阶组合特征，在embedding layer和first hidden layer之间增加了一个product layer。根据product layer使用内积、外积、混合分别衍生出IPNN, OPNN, PNN*三种类型。

无论是FNN还是PNN，他们都有一个绕不过去的缺点：对于低阶的组合特征，学习到的比较少。而前面我们说过，低阶特征对于CTR也是非常重要的。

Google意识到了这个问题，为了同时学习低阶和高阶组合特征，提出了Wide&Deep模型。它混合了一个线性模型（Wide part）和Deep模型(Deep part)。这两部分模型需要不同的输入，而Wide part部分的输入，依旧依赖人工特征工程。

但是，这些模型普遍都存在两个问题：

偏向于提取低阶或者高阶的组合特征。不能同时提取这两种类型的特征。
需要专业的领域知识来做特征工程。
DeepFM在Wide&Deep的基础上进行改进，成功解决了这两个问题，并做了一些改进，其优势/优点如下：

不需要预训练FM得到隐向量
不需要人工特征工程
能同时学习低阶和高阶的组合特征
FM模块和Deep模块共享Feature Embedding部分，可以更快的训练，以及更精确的训练学习


#### 模型演进历史

DeepFM借鉴了Google的wide & deep的做法，其本质是
* 将Wide & Deep 部分的wide部分由 人工特征工程+LR 转换为FM模型，避开了人工特征工程；
* FM模型与deep part共享feature embedding。
* DeepFM模型包含FM和DNN两部分，FM模型可以抽取low-order特征，DNN可以抽取high-order特征。无需Wide&Deep模型人工特征工程。
* 由于输入仅为原始特征，而且FM和DNN共享输入向量特征，DeepFM模型训练速度很快。
* 在Benchmark数据集和商业数据集上，DeepFM效果超过目前所有模型。

![image](https://user-images.githubusercontent.com/39177230/111905420-7c43c900-8a86-11eb-823c-a6f6cede4f41.png)


#### Advantage of DeepFM

![image](https://user-images.githubusercontent.com/39177230/111905512-fe33f200-8a86-11eb-9e79-1d7182a5427a.png)


#### Code example analysis

[T3 - Code Analysis - DeepFM_Model.ipynb](https://github.com/frankyangdev/aliyun-tianchi-DeepRecommendationModelLearning/blob/main/T3%20-%20%20Code%20Analysis%20-%20DeepFM_Model.ipynb)





