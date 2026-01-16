# optical-property_machine-learning
This is a model that transforms 2D images into performance predictions
These code belonging to pjy991211@sjtu.edu.cn; zhongzhiyan@sjtu.edu.cn;zhouxiao113@sjtu.edu.cn
1.Phase0表示对整体进行相关性分析，得到每一个像素点与R相关性程度；
2.Phase1表示用简单CNN方法进行性能预测；
3.Phase2是采用Geninimi3中的FFN进行性能预测；
4.Phase3a是在phase2的基础上增加了傅里叶变换的代码；
5.Phase3b是Phase3a的基础上进行交叉验证；
6.Phase3c是基于训练好的模型判断哪些像素点对R值影响大；
7.Phase3e是判断1000个结构里面哪些性能最差，然后分析为什么在这个模型下性能会最差；
8.Phase0_3c_3e表示代码对应的结果；
9.checkpoint_300表示phase3a训练好模型的权重；
