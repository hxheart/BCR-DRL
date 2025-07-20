import tensorflow as tf

def func_nn_attention(observation_dimensions):
    
    inputs = tf.keras.Input(shape=(observation_dimensions,), dtype="float32")

    # --
    x = tf.keras.layers.Dense(64, activation="tanh")(inputs)
    x = tf.keras.layers.Dense(64, activation="tanh")(x)
    value = tf.keras.layers.Dense(1)(x)  # 直接使用 Dense 层后接 squeeze，无需再在 Dense 层中使用 relu
    value = tf.squeeze(value, axis=1)
    nn_attention = tf.keras.Model(inputs=inputs, outputs=value, name='nn attention reward')
    
    
    return nn_attention


