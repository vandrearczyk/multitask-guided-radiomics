import tensorflow as tf
from src.models.layers import ResidualLayer


class UpBlock(tf.keras.layers.Layer):
    def __init__(self,
                 filters,
                 *args,
                 upsampling_factor=1,
                 filters_output=24,
                 n_conv=2,
                 **kwargs):
        super().__init__()
        self.upsampling_factor = upsampling_factor
        self.conv = tf.keras.Sequential()
        for k in range(n_conv):
            self.conv.add(
                tf.keras.layers.Conv3D(filters,
                                       3,
                                       padding='SAME',
                                       activation='relu'), )
        self.trans_conv = tf.keras.layers.Conv3DTranspose(filters,
                                                          3,
                                                          strides=(2, 2, 2),
                                                          padding='SAME',
                                                          activation='relu')
        self.concat = tf.keras.layers.Concatenate()
        if upsampling_factor != 1:
            self.upsampling = tf.keras.Sequential([
                tf.keras.layers.Conv3D(filters_output,
                                       1,
                                       padding='SAME',
                                       activation='relu'),
                tf.keras.layers.UpSampling3D(size=(upsampling_factor,
                                                   upsampling_factor,
                                                   upsampling_factor)),
            ])
        else:
            self.upsampling = None

    def call(self, inputs):
        x, skip = inputs
        x = self.trans_conv(x)
        x = self.concat([x, skip])
        x = self.conv(x)
        if self.upsampling:
            return x, self.upsampling(x)
        else:
            return x

class UnetRadiomics(tf.keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.down_stack = [
            self.get_first_block(8),
            self.get_down_block(16),
            self.get_down_block(32),
            self.get_down_block(64),
            self.get_down_block(128),
        ]

        self.up_stack = [
            UpBlock(64, n_conv=1, upsampling_factor=8, filters_output=8),
            UpBlock(32, n_conv=1, upsampling_factor=4, filters_output=8),
            UpBlock(16, n_conv=1, upsampling_factor=2, filters_output=8),
            UpBlock(8, n_conv=1),
        ]
        self.last = tf.keras.Sequential([
            tf.keras.layers.Conv3D(8, 3, activation='relu', padding='SAME'),
            tf.keras.layers.Conv3D(
                1, 1, activation='sigmoid', padding='SAME', name="segmentation_output"),
        ])
        self.radiomics = tf.keras.Sequential([
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(1, activation=None, name="radiomics_output")
        ])

    def get_first_block(self, filters):
        return tf.keras.Sequential([
            ResidualLayer(filters, 7, padding='SAME'),
            ResidualLayer(filters, 3, padding='SAME'),
        ])

    def get_down_block(self, filters):
        return tf.keras.Sequential([
            tf.keras.layers.MaxPool3D(pool_size=(2, 2, 2), padding='SAME'),
            ResidualLayer(filters, 3, padding='SAME'),
            ResidualLayer(filters, 3, padding='SAME'),
        ])

    def call(self, inputs, training=None):
        x = tf.keras.layers.InputLayer((144, 144, 144, 2))(inputs)
        skips = []
        xs_downsampled = []
        for block in self.down_stack:
            x = block(x)
            skips.append(x)
            xs_downsampled.append(tf.reduce_mean(x, axis=[1, 2, 3]))
        x_middle = tf.concat(xs_downsampled, axis=-1)
        skips = reversed(skips[:-1])
        xs_upsampled = []

        for block, skip in zip(self.up_stack, skips):
            x = block((x, skip))
            if type(x) is tuple:
                x, x_upsampled = x
                xs_upsampled.append(x_upsampled)

        x += tf.add_n(xs_upsampled)
        return self.last(x), self.radiomics(x_middle)