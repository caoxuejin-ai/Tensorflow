# 模块导入
import numpy as np
import tensorflow as tf
from sklearn import datasets

# 导入数据集，分别为输入特征和标签
x_train = datasets.load_iris().data
y_train = datasets.load_iris().target

# 随机打乱数据（因为原始数据是顺序的，顺序不打乱会影响准确率）
# seed：随机种子，是一个整数，当设置之后，每次生成的随机数都一样
np.random.seed(116)     # 使用相同的seed，保证输入特征和标签一一对应
np.random.shuffle(x_train)
np.random.seed(116)
np.random.shuffle(y_train)
tf.random.set_seed(116)

# 搭建网络结构
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(3, activation="softmax", kernel_regularizer=tf.keras.regularizers.l2())
])

# 配置训练方法
model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.1),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=["sparse_categorical_accuracy"])

# 执行训练过程
model.fit(x_train,
          y_train,
          batch_size=32,
          epochs=500,
          validation_split=0.2,
          validation_freq=20
          )

# 打印出网络结构和参数统计
model.summary()
