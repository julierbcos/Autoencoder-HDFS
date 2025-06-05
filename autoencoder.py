#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Updated on 2025-06-02

Originally by @hananhindy
Updated for TensorFlow 2.x compatibility
"""

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import RMSprop, Adam, Nadam
from tensorflow.keras.regularizers import l2, l1, l1_l2

class autoencoder:
    def __init__(self, num_features, verbose=True, mse_threshold=0.5, archi="U15,D,U9,D,U6,D,U9,D,U15",
                 reg='l2', l1_value=0.1, l2_value=0.001, dropout=0.05, loss='mse'):
        self.mse_threshold = mse_threshold

        regularisation = l2(l2_value)
        if reg == 'l1':
            regularisation = l1(l1_value)
        elif reg == 'l1l2':
            regularisation = l1_l2(l1=l1_value, l2=l2_value)

        layers = archi.split(',')

        input_ = Input((num_features,))
        previous = input_

        for l in layers:
            if l[0] == 'U':
                layer_value = int(l[1:])
                current = Dense(units=layer_value, use_bias=True,
                                kernel_regularizer=regularisation,
                                kernel_initializer='uniform')(previous)
                previous = current
            elif l[0] == 'D':
                current = Dropout(dropout)(previous)
                previous = current

        output_ = Dense(units=num_features)(previous)

        self.model = Model(input_, output_)

        if loss == 'mae':
            self.model.compile(loss=self.loss, optimizer='rmsprop', metrics=[self.accuracy])
        else:
            self.model.compile(loss='mean_squared_error', optimizer='adadelta', metrics=[self.accuracy])

        if verbose:
            self.model.summary()

    def accuracy(self, y_true, y_pred):
        mse = tf.reduce_mean(tf.square(y_true - y_pred), axis=1)
        temp = tf.ones_like(mse)
        return tf.reduce_mean(tf.cast(tf.equal(temp, tf.cast(mse < self.mse_threshold, temp.dtype)), tf.float32))

    def loss(self, y_true, y_pred):
        return tf.reduce_mean(tf.abs(y_pred - y_true), axis=1)