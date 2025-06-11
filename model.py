import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2, l1, l1_l2


class Autoencoder:
    def __init__(self, num_features, verbose=True, mse_threshold=0.5,
                 archi="U128,D,U64,D,U32,D,U64,D,U128",
                 reg='l2', l1_value=0.1, l2_value=0.001, dropout=0.2, loss='mse'):

        self.mse_threshold = mse_threshold

        # Regularization
        regularisation = l2(l2_value)
        if reg == 'l1':
            regularisation = l1(l1_value)
        elif reg == 'l1l2':
            regularisation = l1_l2(l1=l1_value, l2=l2_value)

        # Build model
        layers = archi.split(',')
        input_ = Input((num_features,))
        previous = input_

        for l in layers:
            if l[0] == 'U':
                layer_value = int(l[1:])
                current = Dense(units=layer_value,
                                activation='relu',
                                use_bias=True,
                                kernel_regularizer=regularisation,
                                kernel_initializer='he_normal')(previous)
                previous = current
            elif l[0] == 'D':
                current = Dropout(dropout)(previous)
                previous = current

        output_ = Dense(units=num_features)(previous)
        self.model = Model(input_, output_)
        self.model.compile(loss='mean_squared_error', optimizer='adam')

        if verbose:
            self.model.summary()
