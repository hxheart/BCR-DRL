import tensorflow as tf
from tensorflow.keras.regularizers import l1_l2

l1_reg = 1e-4 # 稀疏
l2_reg = 2e-4 # 缩小

def func_nn_ppo(observation_dimensions, num_actions):
    
    inputs = tf.keras.Input(shape=(observation_dimensions,), dtype="float32")
    
#     x = inputs

#     -- actor --
    x = tf.keras.layers.Reshape((12, 8, 1))(inputs)
    x = tf.keras.layers.Conv2D(25, (5, 5), strides=1, activation="tanh", padding='same', kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg))(x)
    x = tf.keras.layers.Conv2D(25, (3, 3), strides=1, activation="tanh", padding='same', kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg))(x)
    x = tf.keras.layers.Conv2D(25, (3, 3), strides=1, activation="tanh", padding='same', kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg))(x)
    x = tf.keras.layers.Flatten()(x)
    
    x = tf.keras.layers.Dense(64, activation="tanh", kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg))(x)
    x = tf.keras.layers.Dense(64, activation="tanh", kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg))(x)
    action_logits = tf.keras.layers.Dense(num_actions, kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg))(x)  # 通常策略网络的输出不需要激活函数
    actor = tf.keras.Model(inputs=inputs, outputs=action_logits, name='actor')


    # -- critic --
    x = tf.keras.layers.Reshape((12, 8, 1))(inputs)
    x = tf.keras.layers.Conv2D(25, (5, 5), strides=1, activation="tanh", padding='same', kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg))(x)
    x = tf.keras.layers.Conv2D(25, (3, 3), strides=1, activation="tanh", padding='same', kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg))(x)
    x = tf.keras.layers.Conv2D(25, (3, 3), strides=1, activation="tanh", padding='same', kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg))(x)
    x = tf.keras.layers.Flatten()(x)    
    
    x = tf.keras.layers.Dense(64, activation="tanh", kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg))(inputs)
    x = tf.keras.layers.Dense(64, activation="tanh", kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg))(x)
    value = tf.keras.layers.Dense(1, kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg))(x)  #  Dense 后直接接 squeeze，无需再在 Dense 层中使用 relu
    value = tf.squeeze(value, axis=1)
    critic = tf.keras.Model(inputs=inputs, outputs=value, name='critic')
    
    
    return actor, critic


