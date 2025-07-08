import tensorflow as tf
from tensorflow.keras.layers import Layer

class MyDropout(Layer):
    """
    Custom Dropout layer that uses stateless random ops for reproducibility.
    """
    def __init__(self, rate, seed, **kwargs):
        super(MyDropout, self).__init__(**kwargs)
        self.rate = rate
        self.seed = seed

    def call(self, x, training=None):
        if not training or self.rate == 0:
            return x
        
        # Create a stateless seed for reproducibility
        seed_tensor = tf.constant([self.seed, self.seed + 1], dtype=tf.int32)
        
        keep_prob = 1 - self.rate
        scale = 1 / keep_prob
        x_scale = x * scale
        
        random_tensor = tf.random.stateless_uniform(
            tf.shape(x), seed=seed_tensor, dtype=x.dtype
        )
        
        keep_mask = tf.cast(random_tensor >= self.rate, dtype=x.dtype)
        return x_scale * keep_mask

    def get_config(self):
        config = super(MyDropout, self).get_config()
        config.update({
            'rate': self.rate,
            'seed': self.seed,
        })
        return config 