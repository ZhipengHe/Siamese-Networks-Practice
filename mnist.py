import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import tensorflow_datasets as tfds

BATCH_SIZE = 128
NUM_CLASSES = 10
NUM_EPOCHS = 10

def normalize_img(image, label):
  """
  Normalizes images: `uint8` -> `float32`.
  """
  return tf.cast(image, tf.float32) / 255., label


def main():
    """
    """
    (ds_train, ds_test), ds_info = tfds.load('mnist', 
        split=['train','test'], 
        as_supervised=True, 
        with_info=True)
    


if __name__ == "__main__":
    main()
