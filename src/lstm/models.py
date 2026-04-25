from typing import Callable

import keras_tuner as kt
from keras.layers import LSTM, Bidirectional, Dense, Dropout, Input, LayerNormalization
from keras.metrics import AUC
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l2

def make_model_bilstm(hp: kt.HyperParameters, input_shape):
    inputs = Input(shape=input_shape)

    lstm_units    = get_hp(hp, 16,    lambda h: h.Choice("lstm_units", [4, 8, 16, 32]))
    dropout_lstm  = get_hp(hp, 0.3, lambda h: h.Float("lstm_dropout", 0.1, 0.5, step=0.1))
    rec_dropout   = get_hp(hp, 0.3, lambda h: h.Float("lstm_rec_dropout", 0.1, 0.4, step=0.1))
    dropout_mid   = get_hp(hp, 0.5, lambda h: h.Float("lstm_dropout_mid", 0.3, 0.6, step=0.1))
    dense_units   = get_hp(hp, 16,  lambda h: h.Choice("lstm_dense_units", [16, 32, 64]))
    dropout_clf   = get_hp(hp, 0.4, lambda h: h.Float("lstm_dropout_clf", 0.2, 0.5, step=0.1))
    l2_rate       = get_hp(hp, 1e-3,         lambda h: h.Float("lstm_l2", 1e-5, 1e-2, sampling="log"))
    lr            = get_hp(hp, 5e-4,          lambda h: h.Float("lstm_lr", 1e-4, 1e-2, sampling="log"))

    x = Bidirectional(LSTM(
        lstm_units,
        return_sequences=False,
        dropout=dropout_lstm,
        recurrent_dropout=rec_dropout
    ))(inputs)
    x = LayerNormalization()(x)

    x = Dropout(dropout_mid)(x)
    x = Dense(dense_units, activation="relu", kernel_regularizer=l2(l2_rate))(x)
    x = Dropout(dropout_clf)(x)
    outputs = Dense(1, activation="sigmoid")(x)

    model = Model(inputs, outputs)
    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=[AUC(name="pr_auc", curve="PR")]
    )
    return model

def get_hp(hp: kt.HyperParameters, default, hp_fn: Callable):
    if hp is None:
        return default  
    return hp_fn(hp)