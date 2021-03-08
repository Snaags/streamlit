
"""
Module containing methods to build useful network architectures
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.regularizers import l1_l2


def build_sequential(n_inputs, n_classes, n_neurons_l1, n_neurons_l2, layers, learning_rate,
                     input_dropout, l1_dropout, l2_dropout, l1_reg, l2_reg):
    model = Sequential()
    model.add(Dropout(input_dropout))
    
    # Add the input layer and the first hidden layer
    model.add(Dense(units=n_neurons_l1, 
                    activity_regularizer=l1_l2(l1_reg, l2_reg), 
                    activation='relu', 
                    input_dim=n_inputs))

    model.add(Dropout(l1_dropout))

    # Hidden layers
    for layer in range(layers - 1):
        model.add(Dense(units=n_neurons_l2, 
                        activity_regularizer=l1_l2(l1_reg, l2_reg),
                        activation='relu'))
        model.add(Dropout(l2_dropout))

    # Output layer
    if n_classes <= 2:
        model.add(Dense(units=1, activation='sigmoid'))
    else:
        model.add(Dense(units=n_classes, activation='softmax'))

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    loss = 'binary_crossentropy' if n_classes <= 2 else 'sparse_categorical_crossentropy'
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

    return model


def build_sequential_from_dict(n_inputs, n_classes, params):
    n_neurons_l1 = params['n_neurons_l1']
    n_neurons_l2 = params['n_neurons_l2']
    layers = params['layers']
    learning_rate = params['learning_rate']
    l1_dropout = params['l1_dropout']
    l2_dropout = params['l2_dropout']
    l1_reg = params['l1_reg']
    l2_reg = params['l2_reg']
    return build_sequential(n_inputs, n_classes, n_neurons_l1, n_neurons_l2, layers, learning_rate, 
                            0.0, l1_dropout, l2_dropout, l1_reg, l2_reg)
    
