# tensorflow INFO and WARNING messages are not printed
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Input, backend, Model
import tensorflow_datasets as tfds





def load_dataset(dataset):
    print('Start loading dataset - ' + dataset + '...\n')

    (ds_train, ds_test), ds_info = tfds.load(dataset, split=['train','test'], with_info=True, shuffle_files=True)

    print('Dataset  - ' + dataset + ' loaded into train and test splits successfully.\n')

    print("The list of all available labels for this dataset:")
    print(list(ds_info.features.keys()))
    print()

    print("The input shape of the provided image in the dataset:")
    print(ds_info.features['image'].shape)
    print()

    print("The number of images in the training set: " + str(ds_info.splits['train'].num_examples))
    print("The number of images in the test set: " + str(ds_info.splits['test'].num_examples))
    print()

    return ds_train, ds_test, ds_info


def euclidean_distance(vec):
    """Find the Euclidean distance between two vectors.

    Arguments:
        vec: List containing two tensors of same length.

    Returns:
        Tensor containing euclidean distance
        (as floating point value) between vectors.
    """

    x, y = vec
    distance = tf.math.sqrt(tf.math.reduce_sum(tf.math.square(x - y), axis=1, keepdims=True))
    return distance


def siamese_networks(input_shape, loss_name):

    def base_model():

        input_layer = Input(shape=input_shape)
        x = layers.Conv2D(filters=64, kernel_size=(10,10), activation='relu')(input_layer)
        x = layers.MaxPool2D(pool_size=(2,2))(x)
        x = layers.Conv2D(filters=128, kernel_size=(7,7), activation='relu')(x)
        x = layers.MaxPool2D(pool_size=(2,2))(x)
        x = layers.Conv2D(filters=128, kernel_size=(4,4), activation='relu')(x)
        x = layers.MaxPool2D(pool_size=(2,2))(x)
        x = layers.Conv2D(filters=256, kernel_size=(4,4), activation='relu')(x)
        x = layers.Flatten()(x)
        x = layers.Dense(4096, activation='sigmoid')(x)

        model = keras.Model(inputs=input_layer, outputs=x)

        return model

    def contrastive_loss_network():

        input1 = Input(input_shape)
        input2 = Input(input_shape)

        left_model = base_model(input1)
        right_model = base_model(input2)
        L1_distance = layers.Lambda(euclidean_distance)([left_model, right_model])

        prediction = layers.Dense(1,activation='sigmoid')(L1_distance)

        siamese_model = keras.Model(inputs=[input1, input2], outputs=prediction, name="siamese_networks")

        return siamese_model

    def triplet_loss_network():

        input1 = Input(input_shape)
        input2 = Input(input_shape)
        input3 = Input(input_shape)


    if loss_name == 'contrastive_loss':
        return base_model, contrastive_loss_network
    
    if loss_name == 'triplet_loss':
        return base_model, triplet_loss_network


def split_dataset_for_contrastive_networks(ds_train, ds_test, ds_info):

    def pairing(left, right):
        """Normalizes images: `uint8` -> `float32`."""
        # left, right = case
        if left["alphabet"] == right["alphabet"]:
            flag = 0
        else:
            flag = 1
        return tf.cast(left["image"], tf.float32) / 255., tf.cast(right["image"], tf.float32) / 255., tf.cast(flag, tf.float32)

    def split_to_array(dataset, dataset_name):
        num = int(ds_info.splits[dataset_name].num_examples / 2 * 0.8)
        left_train = dataset.take(num)
        right_train = dataset.skip(num).take(num)
        ds_train = tf.data.Dataset.zip((left_train, right_train))

        x_left = []
        x_right = []
        y = []

        for left, right in ds_train:
            left_x, right_x, flag = pairing(left, right)

            x_left.append(left_x)
            x_right.append(right_x)
            y.append(flag)

        x_left = np.array(x_left)
        x_right = np.array(x_right)
        y = np.array(y)

        return x_left, x_right, y


    train_x_left, train_x_right, train_y = split_to_array(ds_train, 'train')
    test_x_left, test_x_right, test_y = split_to_array(ds_test, 'test')

    return train_x_left, train_x_right, train_y, test_x_left, test_x_right, test_y


def loss(loss_name):



    def contrastive_loss(y_true, y_pred, margin = 1):
        """Implementation of the triplet loss function

        Inspired by https://stackoverflow.com/questions/38260113/implementing-contrastive-loss-and-triplet-loss-in-tensorflow

        Args:
            y_true (int): true label, positive pair (same class) -> 0, 
                        negative pair (different class) -> 1
            
            y_pred (list): python list containing two objects in a pair of tensors:
                left : the encodings for one image data in a pair
                right : the encodings for the other image data in a pair
            
            margin (float, optional): m > 0 determines how far the embeddings of 
                        a negative pair should be pushed apart. Defaults to 1.

        Returns:
            loss (float): real number, value of the loss
        """

        left = y_pred[0]
        right = y_pred[1]

        # squared distance between the left image and the right image
        dist = tf.math.reduce_sum(tf.math.square(left - right), 0)

        loss_positive = dist
        loss_negative = tf.math.square(tf.maximum(0., margin - tf.math.sqrt(dist)))
        
        loss = y_true * loss_negative + (1 - y_true) * loss_positive
        loss = 0.5 * tf.math.reduce_mean(loss)

        return loss

    def triplet_loss(y_true, y_pred, margin = 1):
        """Implementation of the triplet loss function

        Arguments:
            y_true : true labels, required when you define a loss in Keras, 
                    not applied in this function.

            y_pred (list): python list containing three objects:
                anchor : the encodings for the anchor data
                positive : the encodings for the positive data (similar to anchor)
                negative : the encodings for the negative data (different from anchor)
            
            margin (float, optional): m > 0 determines how far the embeddings of 
                        a negative data should be pushed apart. Defaults to 1.

        Returns:
            loss (float): real number, value of the loss
        """

        anchor = y_pred[0]
        positive = y_pred[1]
        negative = y_pred[2]

        # squared distance between the anchor and the positive
        pos_dist = tf.math.reduce_sum(tf.math.square(anchor - positive), axis=1)

        # squared distance between the anchor and the negative
        neg_dist = tf.math.reduce_sum(tf.math.square(anchor - negative), axis=1)

        # compute loss
        basic_loss = margin + pos_dist - neg_dist
        loss = tf.math.maximum(basic_loss,0.0)
        loss = tf.math.reduce_mean(loss)
        return loss

    
    if loss_name == 'contrastive_loss':
        return contrastive_loss
    
    if loss_name == 'triplet_loss':
        return triplet_loss




if __name__ == "__main__":

    ds_train, ds_test, ds_info = load_dataset("omniglot")