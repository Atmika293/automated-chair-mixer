# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 13:44:25 2020

@author: JANAKI
modified by Samuel Hatin
"""

import numpy as np
import tensorflow as tf
import chairs_dataset_final

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

#tf.logging.set_verbosity(tf.logging.INFO)


def cnn_model_fn(features, labels, mode):
    # Input layer, change 56 to whatever the dimensions of the input images are
    input_layer = tf.reshape(features['x'], [-1, 224, 224, 1])

    # Conv Layer #1
    conv1 = tf.layers.conv2d(inputs=input_layer, filters=32, kernel_size=[5, 5], padding='same', activation=tf.nn.relu)

    # Pooling Layer #1
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    # Conv Layer #2
    conv2 = tf.layers.conv2d(inputs=pool1, filters=64, kernel_size=[5, 5], padding='same', activation=tf.nn.relu)

    # Pooling Layer #2
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    # Dense Layer
    pool2_flat = tf.reshape(pool2, [-1, 56 * 56 * 64])
    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
    dropout = tf.layers.dropout(inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

    # Logits Layer
    logits = tf.layers.dense(inputs=dropout, units=2)

    predictions = {"classes": tf.argmax(input=logits, axis=1),
                   "probabilities": tf.nn.softmax(logits, name="softmax_tensor")}

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    eval_metrics_ops = {"accuracy": tf.metrics.accuracy(labels=labels, predictions=predictions["classes"])}

    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metrics_ops)


def main(*argv):
    #how many samples will be part of the train slice, the rest will be test data
    train_slice = 0.8

    #load chairs dataset
    imagesTop, imagesFront, imagesLeftSide, imagesRightSide, imagesLeftFront, imagesRightFront, imagesLeftBack, labelsTop, labelsFront, labelsLeftSide, labelsRightSide, labelsLeftFront, labelsRightFront, labelsLeftBack, ids = chairs_dataset_final.load(224)
    
    #first we calculate the ID that will split the dataset between training samples and test samples
    dataset_length = imagesTop.shape[0]
    sliceId = int(dataset_length * train_slice)

    images = []
    
    images.append(imagesTop)
    images.append(imagesFront)
    images.append(imagesLeftSide)
    images.append(imagesRightSide)
    images.append(imagesLeftFront)
    images.append(imagesRightFront)
    images.append(imagesLeftBack)

    
    probabilities = [0] * len(imagesTop)
    
    for view in ["Left Front", "Front", "Right Front", "Right Side", "Left Back", "Left Side", "Top"]:
        id = ["Left Front", "Front", "Right Front", "Right Side", "Left Back", "Left Side", "Top"].index(view)

        classifier = tf.estimator.Estimator(model_fn=cnn_model_fn, model_dir="checkpoint/"+view+"/")

        tensors_to_log = {"probabilities": "softmax_tensor"}

        eval_input_fn = tf.estimator.inputs.numpy_input_fn( x={"x": images[id]},
                                                            num_epochs=1,
                                                            shuffle=False)
        
        predictions = classifier.predict(input_fn=eval_input_fn)

        print("Predictions calculated!")
        
        pred_i = 0
        for pred_dict in predictions:
            probability = pred_dict['probabilities'][0]
            probabilities[pred_i] += probability
            
            #print('Prediction is ({:.1f}%'.format(probability))
            pred_i += 1
    
    for i in range(0, len(probabilities)):
        probability = probabilities[i] / 7.0 #average it over all sides
        chair_id = ids[i]
        print('Prediction for ' + chair_id  + " " + str(probability) )
    
    

if __name__ == "__main__":
    # Add ops to save and restore all the variables.
    #saver = tf.train.Saver()
    tf.compat.v1.app.run()