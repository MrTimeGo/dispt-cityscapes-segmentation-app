import keras

from keras.layers import Conv2D, Conv2DTranspose, InputLayer, Layer, Input, Dropout, MaxPool2D, concatenate
from tensorflow.keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint


# Encoder Layer Block
class EncoderLayerBlock(Layer):
  def __init__(self, filters, rate, pooling=True):
    super(EncoderLayerBlock, self).__init__()
    self.filters = filters
    self.rate = rate
    self.pooling = pooling

    self.c1 = Conv2D(self.filters, kernel_size=3, padding='same', activation='relu', kernel_initializer='he_normal')
    self.drop = Dropout(self.rate)
    self.c2 = Conv2D(self.filters, kernel_size=3, padding='same', activation='relu', kernel_initializer='he_normal')
    self.pool = MaxPool2D(pool_size=(2,2))

  def call(self, X):
    x = self.c1(X)
    x = self.drop(x)
    x = self.c2(x)
    if self.pooling:
      y = self.pool(x)
      return y, x
    else:
      return x

  def get_config(self):
    base_estimator = super().get_config()
    return {
        **base_estimator,
        "filters":self.filters,
        "rate":self.rate,
        "pooling":self.pooling
    }

#  Decoder Layer
class DecoderLayerBlock(Layer):
  def __init__(self, filters, rate, padding='same'):
    super(DecoderLayerBlock, self).__init__()
    self.filters = filters
    self.rate = rate
    self.cT = Conv2DTranspose(self.filters, kernel_size=3, strides=2, padding=padding)
    self.next = EncoderLayerBlock(self.filters, self.rate, pooling=False)

  def call(self, X):
    X, skip_X = X
    x = self.cT(X)
    c1 = concatenate([x, skip_X])
    y = self.next(c1)
    return y

  def get_config(self):
    base_estimator = super().get_config()
    return {
        **base_estimator,
        "filters":self.filters,
        "rate":self.rate,
    }


def create_model_and_compile(input_width, input_height, n_classes):
    # Input Layer
    input_layer = Input(shape=(input_width, input_height, 3), dtype='float32')

    # Encoder
    p1, c1 = EncoderLayerBlock(16, 0.1)(input_layer)
    p2, c2 = EncoderLayerBlock(32, 0.1)(p1)
    p3, c3 = EncoderLayerBlock(64, 0.2)(p2)
    p4, c4 = EncoderLayerBlock(128, 0.2)(p3)

    # Encoding Layer
    c5 = EncoderLayerBlock(256, 0.3, pooling=False)(p4)

    # Decoder
    d1 = DecoderLayerBlock(128, 0.2)([c5, c4])
    d2 = DecoderLayerBlock(64, 0.2)([d1, c3])
    d3 = DecoderLayerBlock(32, 0.2)([d2, c2])
    d4 = DecoderLayerBlock(16, 0.2)([d3, c1])

    # Output layer
    output = Conv2D(n_classes, kernel_size=1, strides=1, padding='same', activation='sigmoid')(d4)

    # U-Net Model
    model = keras.models.Model(
        inputs=[input_layer],
        outputs=[output],
    )

    model.compile(
        loss='binary_crossentropy',
        optimizer=Adam(learning_rate=0.001),
        metrics=['accuracy', keras.metrics.MeanIoU(num_classes=2)]
    )

    return model
