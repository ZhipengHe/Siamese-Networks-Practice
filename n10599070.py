"""
------------------------------------------------------
IFN680 - ARTIFICIAL INTELLIGENCE AND MACHINE LEARNING 
Assignment 2 - Siamese network
------------------------------------------------------
Created by Zhipeng He [n10599070], zhipenghe@connect.qut.edu.au
Created on 29 Sep 2021

Last updated on 31 Oct 2021
------------------------------------------------------

Tasks:
1. Implement and test the contrastive loss function and the triplet loss function
    - Implement in function `loss()`
    - Test in function `loss_test()`
2. Build a Siamese network
    - Build network in function `siamese_networks()`
3. Successful training of the Siamese networks
4. Evaluate the generalization capability of the Siamese networks

"""


# tensorflow INFO and WARNING messages are not printed
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Input, backend, Model
import tensorflow_datasets as tfds


def load_dataset(dataset):
    """ - load dataset from tensorflow dataset
        - print related information to console.
        - split into training and test sets
    Args:
        dataset (str): name of the tensorflow dataset.

    Returns:
        ds_train (tf.data.Dataset): training set
        ds_test (tf.data.Dataset): test set
        ds_info (tfds.core.DatasetInfo): info about the dataset
    """
    print('Start loading dataset - ' + dataset + '...\n')
    # split into training and test sets
    (ds_train, ds_test), ds_info = tfds.load(dataset, split=['train','test'], with_info=True, shuffle_files=True)

    print('Dataset  - ' + dataset + ' loaded into train and test splits successfully.\n')

    print("The list of all available labels for this dataset:")
    print(list(ds_info.features.keys())) # extract available labels from ds_info 
    print()

    print("The input shape of the provided image in the dataset:")
    print(ds_info.features['image'].shape) # extract image shape from ds_info
    print()

    # print the size of training and test sets to console
    print("The number of images in the training set: " + str(ds_info.splits['train'].num_examples))
    print("The number of images in the test set: " + str(ds_info.splits['test'].num_examples))
    print()

    return ds_train, ds_test, ds_info


def split_dataset_for_contrastive_networks(ds_train, ds_test, ds_info, label):
    """Split the training and test sets into subsets for inputing into contrastive networks

    Steps: 
        1. split the training or test sets into two groups, left and right
        2. make pairing of the left image and right image, and
            check if two images belonging to same alphabet class by flag
        3. store left images, right images and flag to three numpy array

    Args:
        dataset (dict): a dictionary store all the images and labels in the dataset
    """

    def pairing(left, right):
        """ - check if two images belonging to same alphabet class by label 
                same class: 0, not same: 1
            - normalizes images: `uint8` -> `float32`.

        Returns:
            the left image, right image and label in `float32`
        """
        # same class: 0
        if left[label] == right[label]:
            flag = 0
        # not same: 1
        else:
            flag = 1
        return tf.cast(left["image"], tf.float32) / 255., tf.cast(right["image"], tf.float32) / 255., tf.cast(flag, tf.float32)

    # initialize a dictionary to store all dataset
    dataset = dict()

    # calculate the number of pairs in training, test and validation sset
    num_train = int(ds_info.splits['train'].num_examples / 2 * 0.8)
    num_val = int(ds_info.splits['train'].num_examples / 2 * 0.2)
    num_test = int(ds_info.splits['test'].num_examples / 2)

    # split train set into train and validation
    left_train = ds_train.take(num_train)
    right_train = ds_train.skip(num_train).take(num_train)
    train = tf.data.Dataset.zip((left_train, right_train))

    left_val = ds_train.skip(num_train * 2).take(num_val)
    right_val = ds_train.skip(num_train * 2 + num_val).take(num_val)
    val = tf.data.Dataset.zip((left_val, right_val))

    left_test = ds_test.take(num_test)
    right_test = ds_test.skip(num_test).take(num_test)
    test = tf.data.Dataset.zip((left_test, right_test))

    # store left images, right images and flag to three numpy array
    # training set
    train_x_left = []
    train_x_right = []
    train_y = []

    for left, right in train:
        left_x, right_x, flag = pairing(left, right)

        train_x_left.append(left_x)
        train_x_right.append(right_x)
        train_y.append(flag)

    train_x_left = np.array(train_x_left)
    train_x_right = np.array(train_x_right)
    train_y = np.array(train_y)

    # store to dictionary
    dataset["train_x_left"] = train_x_left
    dataset["train_x_right"] = train_x_right
    dataset["train_y"] = train_y

    # validation set 
    val_x_left = []
    val_x_right = []
    val_y = []

    for left, right in val:
        left_x, right_x, flag = pairing(left, right)

        val_x_left.append(left_x)
        val_x_right.append(right_x)
        val_y.append(flag)

    val_x_left = np.array(val_x_left)
    val_x_right = np.array(val_x_right)
    val_y = np.array(val_y)

    # store to dictionary
    dataset["val_x_left"] = val_x_left
    dataset["val_x_right"] = val_x_right
    dataset["val_y"] = val_y


    # test set 
    test_x_left = []
    test_x_right = []
    test_y = []

    for left, right in test:
        left_x, right_x, flag = pairing(left, right)

        test_x_left.append(left_x)
        test_x_right.append(right_x)
        test_y.append(flag)

    test_x_left = np.array(test_x_left)
    test_x_right = np.array(test_x_right)
    test_y = np.array(test_y)

    # store to dictionary
    dataset["test_x_left"] = test_x_left
    dataset["test_x_right"] = test_x_right
    dataset["test_y"] = test_y

    return dataset


def split_dataset_for_triplet_networks(ds_train, ds_test, ds_info):
    """Split the training and test sets into subsets for inputing into triplet networks

    Steps: 
        1. split the training or test sets into two groups, anchor and positive
        2. make pairing of the anchor image and positive image, and
            check if two images belonging to same alphabet class by flag
        3. store anchor images, positive images and negative images to three numpy array

    Args:
        dataset (dict): a dictionary store all the images and labels in the dataset
    """

    def pairing(anchor, positive, negative):
        """- normalizes images: `uint8` -> `float32`.

        Returns:
            the left image, right image and label in `float32`
        """
        return tf.cast(anchor["image"], tf.float32) / 255., tf.cast(positive["image"], tf.float32) / 255., tf.cast(negative["image"], tf.float32)

    # initialize a dictionary to store all dataset
    dataset = dict()

    # calculate the number of pairs in training, test and validation sset
    num_train = int(ds_info.splits['train'].num_examples / 3 * 0.8)
    num_val = int(ds_info.splits['train'].num_examples / 3 * 0.2)
    num_test = int(ds_info.splits['test'].num_examples / 3)

    # split train set into train and validation
    anchor_train = ds_train.take(num_train)
    positive_train = ds_train.skip(num_train).take(num_train)
    negative_train = ds_train.skip(num_train*2).take(num_train)
    train = tf.data.Dataset.zip((anchor_train, positive_train, negative_train))

    anchor_val = ds_train.skip(num_train*3).take(num_val)
    positive_val = ds_train.skip(num_train*3 + num_val).take(num_val)
    negative_val = ds_train.skip(num_train*3 + num_val*2).take(num_val)
    val = tf.data.Dataset.zip((anchor_val, positive_val, negative_val))

    anchor_test = ds_test.take(num_test)
    positive_test = ds_test.skip(num_test).take(num_test)
    negative_test = ds_test.skip(num_test*2).take(num_test)
    train = tf.data.Dataset.zip((anchor_test, positive_test, negative_test))

    # store anchor images, positive images and negative images to three numpy array
    # training set
    train_x_anchor = []
    train_x_positive = []
    train_x_negative = []

    for anchor, positive, negative in train:
        anchor_x, positive_x, negative_x = pairing(anchor, positive, negative)

        train_x_anchor.append(anchor_x)
        train_x_positive.append(positive_x)
        train_x_negative.append(negative_x)

    train_x_anchor = np.array(train_x_anchor)
    train_x_positive = np.array(train_x_positive)
    train_x_negative = np.array(train_x_negative)

    # store to dictionary
    dataset["train_x_anchor"] = train_x_anchor
    dataset["train_x_positive"] = train_x_positive
    dataset["train_x_negative"] = train_x_negative

    # val set
    val_x_anchor = []
    val_x_positive = []
    val_x_negative = []

    for anchor, positive, negative in val:
        anchor_x, positive_x, negative_x = pairing(anchor, positive, negative)

        val_x_anchor.append(anchor_x)
        val_x_positive.append(positive_x)
        val_x_negative.append(negative_x)

    val_x_anchor = np.array(val_x_anchor)
    val_x_positive = np.array(val_x_positive)
    val_x_negative = np.array(val_x_negative)

    # store to dictionary
    dataset["val_x_anchor"] = val_x_anchor
    dataset["val_x_positive"] = val_x_positive
    dataset["val_x_negative"] = val_x_negative


    # test  set
    test_x_anchor = []
    test_x_positive = []
    test_x_negative = []

    for anchor, positive, negative in val:
        anchor_x, positive_x, negative_x = pairing(anchor, positive, negative)

        test_x_anchor.append(anchor_x)
        test_x_positive.append(positive_x)
        test_x_negative.append(negative_x)

    test_x_anchor = np.array(test_x_anchor)
    test_x_positive = np.array(test_x_positive)
    test_x_negative = np.array(test_x_negative)

    # store to dictionary
    dataset["test_x_anchor"] = test_x_anchor
    dataset["test_x_positive"] = test_x_positive
    dataset["test_x_negative"] = test_x_negative

    return dataset


def base_model(input_shape):
    """The base branch for creating siamese networks. 
    A full siamese network is a combination of several base models."""

    # Layer 0: input layer
    input_layer = Input(shape=input_shape)
    # layer 1: Conv2D layer
    x = layers.Conv2D(filters=64, kernel_size=(10,10), activation='relu')(input_layer)
    # layer 2: MaxPool2D layer
    x = layers.MaxPool2D(pool_size=(2,2))(x)
    # layer 3: Conv2D layer
    x = layers.Conv2D(filters=128, kernel_size=(7,7), activation='relu')(x)
    # layer 4: MaxPool2D layer
    x = layers.MaxPool2D(pool_size=(2,2))(x)
    # layer 5: Conv2D layer
    x = layers.Conv2D(filters=128, kernel_size=(4,4), activation='relu')(x)
    # layer 6: MaxPool2D layer
    x = layers.MaxPool2D(pool_size=(2,2))(x)
    # layer 7: Conv2D layer
    x = layers.Conv2D(filters=256, kernel_size=(4,4), activation='relu')(x)
    # Layer 8: flatten layer
    x = layers.Flatten()(x)
    # layer 9: Fully connected layer
    x = layers.Dense(4096, activation='sigmoid')(x)

    base_model = keras.Model(inputs=input_layer, outputs=x)
    return base_model


def contrastive_loss_network(input_shape):
    """ Nested internal function
        Based on two branches of base_model(), create siamese networks for usage of contrastive loss.

    Returns:
        contrastive_loss_model (tf.keras.Model): siamese networks for usage of contrastive loss
    """
    model = base_model(input_shape)

    # input layer for left and right branches
    input_left = Input(input_shape)
    input_right = Input(input_shape)

    # init the left model and right model (two base_model())
    left_model = model(input_left)
    right_model = model(input_right)

    contrastive_loss_model = keras.Model(inputs=[input_left, input_right], outputs=[left_model, right_model], name="contrastive_loss_network")

    return contrastive_loss_model



def triplet_loss_network(input_shape):
    """ Nested internal function
        Based on three branches of base_model(), create siamese networks for usage of triplet loss.

    Returns:
        triplet_loss_model (tf.keras.Model): siamese networks for usage of triplet loss
    """
    model = base_model(input_shape)

    # input layer for anchor, positive and negative branches
    input_anchor = Input(input_shape)
    input_positive = Input(input_shape)
    input_negative = Input(input_shape)

    # init the anchor model, positive model and negative model (three base_model())
    anchor_model = model(input_anchor)
    positive_model = model(input_positive)
    negative_model = model(input_negative)

    triplet_loss_model = keras.Model(inputs=[input_anchor, input_positive, input_negative], outputs=[anchor_model, positive_model, negative_model], name="triplet_loss_network")

    return triplet_loss_model


def loss(loss_name):
    """Implementation my own loss function for tensorflow.
    Two loss functions are implemented here, "contrastive_loss" and "triplet_loss".

    Args:
        loss_name (str): the name of loss function. In this function, 
                        the name should be "contrastive_loss" or "triplet_loss".
        margin (float, optional): m > 0 determines how far the embeddings of 
                        a negative pair should be pushed apart. Defaults to 1.
    """
 
    def contrastive_loss(y_true, y_pred, margin = 1):
        """Implementation of the triplet loss function


        Contrastive loss = 0.5 * mean( (1-true_value) * square(distance) + true_value * square( max(margin-distance, 0) ))

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

        distance = tf.math.sqrt(tf.math.reduce_sum(tf.math.square(left - right), axis=-1))

        loss_positive = tf.math.square(distance)
        loss_negative = tf.math.square(tf.maximum(0., margin - distance))
        
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
        pos_dist = tf.math.reduce_sum(tf.math.square(anchor - positive), axis=-1)

        # squared distance between the anchor and the negative
        neg_dist = tf.math.reduce_sum(tf.math.square(anchor - negative), axis=-1)

        # compute loss
        basic_loss = margin + pos_dist - neg_dist
        loss = tf.math.maximum(basic_loss,0.0)
        loss = tf.math.reduce_mean(loss)
        return loss

    
    if loss_name == 'contrastive_loss':
        return contrastive_loss
    
    if loss_name == 'triplet_loss':
        return triplet_loss


def contrastive_loss_test():
    """Test implementation of contrastive loss function 
    code from week9 practice"""
    num_data = 10
    feat_dim = 6
    margin = 0.2

    embeddings = [np.random.rand(num_data, feat_dim).astype(np.float32),
                np.random.rand(num_data, feat_dim).astype(np.float32)]
    labels = np.random.randint(0, 1, size=(num_data)).astype(np.float32)

    #Compute loss with numpy
    loss_np = 0.
    left = embeddings[0]
    right = embeddings[1]

    for i in range(num_data):
        dist = np.sqrt(np.sum(np.square(left[i] - right[i])))
        loss_pos = np.square(dist)
        loss_neg = np.square(max(0. ,(margin - dist)))
        loss_np += labels[i] * loss_neg + (1 - labels[i]) * loss_pos
    loss_np /= num_data
    loss_np *= 0.5
    print('Contrastive loss computed with numpy', loss_np)

    loss_tf = loss('contrastive_loss')

    loss_tf_val = loss_tf(labels, embeddings, margin)
    print('Contrastive loss computed with tensorflow', loss_tf_val)
    assert np.allclose(loss_np, loss_tf_val)


def triplet_loss_test():
    """Test if the triplet loss function works correctly
    """

    #Test implementation of triplet loss function 
    # code from week9 practice
    num_data = 10
    feat_dim = 6
    margin = 0.2

    embeddings = [np.random.rand(num_data, feat_dim).astype(np.float32),
                np.random.rand(num_data, feat_dim).astype(np.float32),
                np.random.rand(num_data, feat_dim).astype(np.float32)]
    labels = np.random.randint(0, 1, size=(num_data)).astype(np.float32)

    #Compute loss with numpy
    loss_np = 0.
    anchor = embeddings[0]
    positive = embeddings[1]
    negative = embeddings[2]

    for i in range(num_data):
        pos_dist = np.sum(np.square(anchor[i] - positive[i]))
        neg_dist = np.sum(np.square(anchor[i] - negative[i]))
        loss_np += max(0. ,(margin + pos_dist - neg_dist))
    loss_np /= num_data
    print('Triplet loss computed with numpy', loss_np)

    loss_tf = loss('triplet_loss')

    loss_tf_val = loss_tf(labels, embeddings, margin)
    print('Triplet loss computed with tensorflow', loss_tf_val)
    assert np.allclose(loss_np, loss_tf_val)

def euclidean_distance(vec):
    """Find the Euclidean distance between two vectors.

    Arguments:
        vec: List containing two tensors of same length.

    Returns:
        Tensor containing euclidean distance
        (as floating point value) between vectors.
    """

    x, y = vec
    distance = tf.math.sqrt(tf.math.reduce_sum(tf.math.square(x - y), axis=-1, keepdims=True))
    return distance

def sigmoid(X):
    """Apply sigmoid function
    """
    return 1/(1+np.exp(-X))

def prediction(score, test_y, lossname):
    """Based on the outcome of the models predict if provided pairs belonging to the same alphabet
    """
    if lossname == 'contrastive_loss':
        prediction = sigmoid(euclidean_distance(score))

        threshold = 0.5
        correction = 0
        error = 0

        for i in range(len(prediction)):
            if prediction[i] >= threshold:
                pred = 1
            else:
                pred = 0
            
            if pred == test_y[i]:
                correction += 1
            else:
                error += 1
        
        accuracy = 1.0 * correction / (correction + error)
    
    return correction, error, accuracy

def plot_loss(history, name):
    """plot loss vs time figure"""

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    # plt.show()
    plt.savefig(name, format="png")

if __name__ == "__main__":

    # task: load and split omniglot dataset
    print('===============================')
    ds_train, ds_test, ds_info = load_dataset("omniglot")
    print("Preprocessing dataset for contrastive loss-based networks:")
    print("It takes about 60 seconds...")
    dataset_contrastive_networks = split_dataset_for_contrastive_networks(ds_train, ds_test, ds_info, 'alphabet')
    dataset_triplet_networks = split_dataset_for_triplet_networks(ds_train, ds_test, ds_info)
    print('Function split_dataset_for_contrastive_networks finished')

    print('===============================')

    # task: implement and test loss functions
    print('===============================')
    print('Implement and test loss functions')
    triplet_loss_test()
    contrastive_loss_test()
    print('===============================')

    # get the shape of the image
    input_shape = ds_info.features['image'].shape

    # task: build siamese network
    print('===============================')
    
    print('\nBuild a base network for Siamese network\n\n')
    base = base_model(input_shape)
    base.summary()

    print('\nBuild a Siamese network with contrastive loss\n\n')
    contrastive_loss_model = contrastive_loss_network(input_shape)
    contrastive_loss_model.summary()

    print('\nBuild a Siamese network with triplet loss\n\n')
    triplet_loss_model = triplet_loss_network(input_shape)
    triplet_loss_model.summary()
    print('===============================')

    # task: train Siamese network
    print('===============================')

    print('\nTrain the Siamese network with contrastive loss\n\n')
    contrastive_loss_model.compile(
        loss=loss('contrastive_loss'),
        optimizer=keras.optimizers.SGD(),
    )
    contrastive_loss_history = contrastive_loss_model.fit(
        x=[dataset_contrastive_networks['train_x_left'], dataset_contrastive_networks['train_x_right']],
        y=dataset_contrastive_networks['train_y'],
        epochs=20,
        batch_size=256,
        validation_data=([dataset_contrastive_networks['val_x_left'], dataset_contrastive_networks['val_x_right']], dataset_contrastive_networks['val_y']),
    )

    plot_loss(contrastive_loss_history, "Siamese-network-with-contrastive-loss.png")

    # print('\nTrain the Siamese network with triplet loss\n\n')
    # triplet_loss_model.compile(
    #     loss=loss('triplet_loss'),
    #     optimizer=keras.optimizers.SGD(),
    # )
    # triplet_loss_history = triplet_loss_model.fit(
    #     x=[dataset_triplet_networks['train_x_anchor'], dataset_triplet_networks['train_x_positive'], dataset_triplet_networks['train_x_negative']],
    #     epochs=20,
    #     batch_size=256,
    #     validation_data=([dataset_triplet_networks['val_x_anchor'], dataset_triplet_networks['val_x_positive'], dataset_triplet_networks['val_x_negative']]),
    # )


    print("Predict on training split with contrastive_networks")
    test_x_1 = [ np.concatenate((dataset_contrastive_networks['train_x_left'], dataset_contrastive_networks['val_x_left']), axis=0),
         np.concatenate((dataset_contrastive_networks['train_x_right'], dataset_contrastive_networks['val_x_right']), axis=0)]
    test_y_1 = np.concatenate((dataset_contrastive_networks['train_y'], dataset_contrastive_networks['val_y']), axis=0)
    contrastive_score_1 = contrastive_loss_model.predict(x=test_x_1, verbose=1)
    correction_1, error_1, accuracy_1 = prediction(contrastive_score_1, test_y_1, 'contrastive_loss')
    print('Test accuracy:', accuracy_1)
    print("True:", correction_1)
    print("False:", error_1)

    print("predict on training and test splits with contrastive_networks")
    test_x_2 = [ np.concatenate((dataset_contrastive_networks['train_x_left'], dataset_contrastive_networks['val_x_left'], dataset_contrastive_networks['test_x_left']),axis=0),
        np.concatenate((dataset_contrastive_networks['train_x_right'], dataset_contrastive_networks['val_x_right'], dataset_contrastive_networks['test_x_right']),axis=0)]
    test_y_2 = np.concatenate((dataset_contrastive_networks['train_y'] , dataset_contrastive_networks['val_y'] , dataset_contrastive_networks['test_y']), axis=0)
    contrastive_score_2 = contrastive_loss_model.predict(x=test_x_2, verbose=1)
    correction_2, error_2, accuracy_2 = prediction(contrastive_score_2, test_y_2, 'contrastive_loss')
    print('Test accuracy:', accuracy_2)
    print("True:", correction_2)
    print("False:", error_2)

    print("predict on test split with contrastive_networks")
    test_x_3 = [dataset_contrastive_networks['test_x_left'] , dataset_contrastive_networks['test_x_right']]
    test_y_3 = dataset_contrastive_networks['test_y']
    contrastive_score_3 = contrastive_loss_model.predict(x=test_x_3, verbose=1)
    correction_3, error_3, accuracy_3 = prediction(contrastive_score_3, test_y_3, 'contrastive_loss')
    print('Test accuracy:', accuracy_3)
    print("True:", correction_3)
    print("False:", error_3)




    

