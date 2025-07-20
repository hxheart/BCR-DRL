import tensorflow as tf

def func_nn_ppo(observation_dimensions, num_actions):
    
    inputs = tf.keras.Input(shape=(observation_dimensions,), dtype="float32")
    
    x = inputs

    if observation_dimensions == 96:
        x = tf.keras.layers.Reshape((12, 8, 1))(inputs)
    elif observation_dimensions == 4:
        x = tf.keras.layers.Reshape((2, 2, 1))(inputs)

    x = tf.keras.layers.Conv2D(25, (5, 5), strides=1, activation="relu", padding='same')(x)
    x = tf.keras.layers.Conv2D(25, (3, 3), strides=1, activation="relu", padding='same')(x)
#     x = tf.keras.layers.Conv2D(25, (3, 3), strides=1, activation="relu", padding='same')(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(64, activation="tanh")(x)
    x = tf.keras.layers.Dense(64, activation="tanh")(x)
#     x = tf.keras.layers.Dense(32, activation="relu")(x)

    action_logits = tf.keras.layers.Dense(num_actions)(x)  # 通常策略网络的输出不需要激活函数
    actor = tf.keras.Model(inputs=inputs, outputs=action_logits, name='actor')


    # --
    x = tf.keras.layers.Dense(64, activation="tanh")(inputs)
    x = tf.keras.layers.Dense(64, activation="tanh")(x)
    value = tf.keras.layers.Dense(1)(x)  # 直接使用 Dense 层后接 squeeze，无需再在 Dense 层中使用 relu
    value = tf.squeeze(value, axis=1)
    critic = tf.keras.Model(inputs=inputs, outputs=value, name='critic')
    
    
    return actor, critic


