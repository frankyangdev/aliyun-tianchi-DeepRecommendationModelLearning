#### 推荐系统

![image](https://user-images.githubusercontent.com/39177230/111759984-2bdb3880-88d9-11eb-8bd7-c07a759f0b23.png)


#### [CNN、RNN、DNN区别](https://blog.csdn.net/lff1208/article/details/77717149)

![image](https://user-images.githubusercontent.com/39177230/111754741-4e6a5300-88d3-11eb-83aa-c7015afc501c.png)

#### [神经网络DNN--详解](https://blog.csdn.net/qq_33472146/article/details/91351181)

![image](https://user-images.githubusercontent.com/39177230/111755165-c5075080-88d3-11eb-96ba-b5b1a353c835.png)

![image](https://user-images.githubusercontent.com/39177230/111757653-83c47000-88d6-11eb-8393-01529848a8eb.png)


#### Wide部分是一个广义的线性模型，输入的特征主要有两部分组成，一部分是原始的部分特征，另一部分是原始特征的交叉特征(cross-product transformation)，对于交互特征可以定义为： $$ \phi_{k}(x)=\prod_{i=1}^d x_i^{c_{ki}}, c_{ki}\in {0,1} $$ $c_{ki}$是一个布尔变量，当第i个特征属于第k个特征组合时，$c_{ki}$的值为1，否则为0，$x_i$是第i个特征的值，大体意思就是两个特征都同时为1这个新的特征才能为1，否则就是0，说白了就是一个特征组合。用原论文的例子举例：

AND(user_installed_app=QQ, impression_app=WeChat)，当特征user_installed_app=QQ,和特征impression_app=WeChat取值都为1的时候，组合特征AND(user_installed_app=QQ, impression_app=WeChat)的取值才为1，否则为0。

对于wide部分训练时候使用的优化器是带$L_1$正则的FTRL算法(Follow-the-regularized-leader)，而L1 FTLR是非常注重模型稀疏性质的，也就是说W&D模型采用L1 FTRL是想让Wide部分变得更加的稀疏，即Wide部分的大部分参数都为0，这就大大压缩了模型权重及特征向量的维度。**Wide部分模型训练完之后留下来的特征都是非常重要的，那么模型的“记忆能力”就可以理解为发现"直接的"，“暴力的”，“显然的”关联规则的能力。**例如Google W&D期望wide部分发现这样的规则：用户安装了应用A，此时曝光应用B，用户安装应用B的概率大。

Wide侧记住的是历史数据中那些常见、高频的模式，是推荐系统中的“红海”。实际上，Wide侧没有发现新的模式，只是学习到这些模式之间的权重，做一些模式的筛选。正因为Wide侧不能发现新模式，因此我们需要根据人工经验、业务背景，将我们认为有价值的、显而易见的特征及特征组合，喂入Wide侧

#### Deep部分是一个DNN模型，输入的特征主要分为两大类，一类是数值特征(可直接输入DNN)，一类是类别特征(需要经过Embedding之后才能输入到DNN中)，Deep部分的数学形式如下： $$ a^{(l+1)} = f(W^{l}a^{(l)} + b^{l}) $$ **我们知道DNN模型随着层数的增加，中间的特征就越抽象，也就提高了模型的泛化能力。**对于Deep部分的DNN模型作者使用了深度学习常用的优化器AdaGrad，这也是为了使得模型可以得到更精确的解。

Deep侧就是DNN，通过embedding的方式将categorical/id特征映射成稠密向量，让DNN学习到这些特征之间的深层交叉，以增强扩展能力。

模型的实现与模型结构类似由deep和wide两部分组成，这两部分结构所需要的特征在上面已经说过了，针对当前数据集实现，我们在wide部分加入了所有可能的一阶特征，包括数值特征和类别特征的onehot都加进去了，其实也可以加入一些与wide&deep原论文中类似交叉特征。只要能够发现高频、常见模式的特征都可以放在wide侧，对于Deep部分，在本数据中放入了数值特征和类别特征的embedding特征，实际应用也需要根据需求进行选择

#### [AdaGrad](https://blog.csdn.net/u010089444/article/details/76725843?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522161614564616780262511289%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=161614564616780262511289&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~baidu_landing_v2~default-6-76725843.first_rank_v2_pc_rank_v29&utm_term=Adagrad)
![image](https://user-images.githubusercontent.com/39177230/111758613-ac009e80-88d7-11eb-9e5b-dcbc596376be.png)


#### Questions:

#### 在你的应用场景中，哪些特征适合放在Wide侧，哪些特征适合放在Deep侧，为什么呢？
Wide侧记住的是历史数据中那些常见、高频的模式,Wide侧就是普通LR，一般根据人工先验知识，将一些简单、明显的特征交叉，喂入Wide侧，让Wide侧能够记住这些规则。
Deep侧数值特征和类别特征的embedding特征


#### 为什么Wide部分要用L1 FTRL训练？
L1 FTLR是非常注重模型稀疏性质的，也就是说W&D模型采用L1 FTRL是想让Wide部分变得更加的稀疏，即Wide部分的大部分参数都为0，这就大大压缩了模型权重及特征向量的维度

#### 为什么Deep部分不特别考虑稀疏性的问题？
Deep部分是传统的前馈神经网络，对于定类特征，会先对其进行嵌入操作，即对每个类别特征嵌入到低维的稠密向量。
Deep侧就是DNN，通过embedding的方式将categorical/id特征映射成稠密向量，让DNN学习到这些特征之间的深层交叉，以增强扩展能力。

关键在于Deep侧与Wide侧共享一个embedding矩阵来映射categorical/id特征到稠密向量
Deep侧将embedding结果喂入DNN，来学习深层交互的权重，着重“扩展”
Wide侧将embedding结果喂入FM，来学习二次交互的权重，着重“记忆”


[Wide&Deep versus DeepFM](https://blog.csdn.net/sinat_29819401/article/details/91359217?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522161614607416780266229481%2522%252C%2522scm%2522%253A%252220140713.130102334.pc%255Fall.%2522%257D&request_id=161614607416780266229481&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~first_rank_v2~rank_v29-11-91359217.first_rank_v2_pc_rank_v29&utm_term=wide+deep%E7%89%B9%E5%BE%81%E5%B7%A5%E7%A8%8B)

