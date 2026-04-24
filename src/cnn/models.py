from typing import Callable

import keras_tuner as kt
import tensorflow as tf
from keras.layers import (
    Add,
    Concatenate,
    Conv1D,
    Dense,
    Dropout,
    GlobalAveragePooling1D,
    Input,
    Lambda,
    LayerNormalization,
    Multiply,
    ReLU,
    Softmax,
)
from keras.metrics import AUC
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l2
from tcn import TCN


def make_model_res_net1D(hp: kt.HyperParameters, input_shape):
    inputs = Input(shape=input_shape)

    # Hiperparámetros — con restricciones anotadas
    filters_b1    = get_hp(hp, 32,   lambda h: h.Choice("resnet_filters_b1", [16, 32, 64]))
    filters_b2    = get_hp(hp, 64,   lambda h: h.Choice("resnet_filters_b2", [32, 64, 128]))
    # filters_b3 DEBE ser igual a filters_b2 — el shortcut2 no proyecta
    filters_b3    = filters_b2
    kernel_b1     = get_hp(hp, 5,     lambda h: h.Choice("resnet_kernel_b1", [3, 5, 7]))
    l2_rate       = get_hp(hp, 1e-4,         lambda h: h.Float("resnet_l2", 1e-5, 1e-2, sampling="log"))
    dropout_b1    = get_hp(hp, 0.2,  lambda h: h.Float("resnet_dropout_b1", 0.1, 0.4, step=0.1))
    dropout_b2b3  = get_hp(hp, 0.3,lambda h: h.Float("resnet_dropout_b2b3", 0.1, 0.4, step=0.1))
    dropout_pool  = get_hp(hp, 0.5,lambda h: h.Float("resnet_dropout_pool", 0.3, 0.6, step=0.1))
    dropout_clf   = get_hp(hp, 0.4, lambda h: h.Float("resnet_dropout_clf", 0.2, 0.5, step=0.1))
    dense_units   = get_hp(hp, 32,  lambda h: h.Choice("resnet_dense_units", [16, 32, 64]))
    lr            = get_hp(hp, 5e-4,         lambda h: h.Float("resnet_lr", 1e-4, 1e-2, sampling="log"))

    # ── Bloque 1 ─────────────────────────────
    x = Conv1D(filters_b1, kernel_b1, padding="same", kernel_regularizer=l2(l2_rate))(inputs)
    x = LayerNormalization()(x)
    x = ReLU()(x)
    x = Dropout(dropout_b1)(x)

    # ── Bloque 2 (residual) ─────────────────
    # shortcut proyecta de filters_b1 → filters_b2
    shortcut = Conv1D(filters_b2, 1, padding="same")(x)
    x = Conv1D(filters_b2, kernel_b1, padding="same", kernel_regularizer=l2(l2_rate))(x)
    x = LayerNormalization()(x)
    x = ReLU()(x)
    x = Dropout(dropout_b2b3)(x)
    x = Conv1D(filters_b2, 3, padding="same", kernel_regularizer=l2(l2_rate))(x)
    x = LayerNormalization()(x)
    x = Add()([x, shortcut])
    x = ReLU()(x)

    # ── Bloque 3 (residual) ─────────────────
    # filters_b3 == filters_b2, shortcut2 no necesita proyección
    shortcut2 = x
    x = Conv1D(filters_b3, 3, padding="same", kernel_regularizer=l2(l2_rate))(x)
    x = LayerNormalization()(x)
    x = ReLU()(x)
    x = Dropout(dropout_b2b3)(x)
    x = Conv1D(filters_b3, 3, padding="same", kernel_regularizer=l2(l2_rate))(x)
    x = LayerNormalization()(x)
    x = Add()([x, shortcut2])
    x = ReLU()(x)

    # ── Attention Pooling ────────────────────
    score = Dense(1, use_bias=False)(x)
    weight = Softmax(axis=1)(score)
    x_att = Multiply()([x, weight])
    x_att = Lambda(lambda t: tf.reduce_sum(t, axis=1))(x_att)
    x_avg = GlobalAveragePooling1D()(x)
    x = Concatenate()([x_att, x_avg])
    x = Dropout(dropout_pool)(x)

    # ── Clasificador ────────────────────────
    x = Dense(dense_units, activation="relu")(x)
    x = Dropout(dropout_clf)(x)
    outputs = Dense(1, activation="sigmoid")(x)

    model = Model(inputs, outputs)
    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=[AUC(name="pr_auc", curve="PR")]
    )
    return model


def make_model_tcn(hp, input_shape):
    inputs = Input(shape=input_shape)

    # Hiperparámetros
    nb_filters   = get_hp(hp, "tcn_filters", 32,    lambda h: h.Choice("tcn_filters", [16, 32, 64]))
    kernel_size  = get_hp(hp, "tcn_kernel", 5,      lambda h: h.Choice("tcn_kernel", [3, 5, 7]))
    dropout_tcn  = get_hp(hp, "tcn_dropout", 0.4,   lambda h: h.Float("tcn_dropout", 0.2, 0.5, step=0.1))
    dropout_pool = get_hp(hp, "tcn_dropout_pool", 0.5, lambda h: h.Float("tcn_dropout_pool", 0.3, 0.6, step=0.1))
    dropout_clf  = get_hp(hp, "tcn_dropout_clf", 0.4,  lambda h: h.Float("tcn_dropout_clf", 0.2, 0.5, step=0.1))
    dense_units  = get_hp(hp, "tcn_dense_units", 16,   lambda h: h.Choice("tcn_dense_units", [16, 32, 64]))
    l2_rate      = get_hp(hp, "tcn_l2", 1e-3,          lambda h: h.Float("tcn_l2", 1e-5, 1e-2, sampling="log"))
    lr           = get_hp(hp, "tcn_lr", 5e-4,           lambda h: h.Float("tcn_lr", 1e-4, 1e-2, sampling="log"))

    x = TCN(
        nb_filters=nb_filters,
        kernel_size=kernel_size,
        dilations=[1, 2, 4, 8, 16],
        padding="causal",
        use_skip_connections=True,
        use_layer_norm=True,
        dropout_rate=dropout_tcn,
        return_sequences=True,
        name="tcn",
    )(inputs)

    score = Dense(1, use_bias=False)(x)
    weight = Softmax(axis=1)(score)
    x_att = Multiply()([x, weight])
    x_att = Lambda(lambda t: tf.reduce_sum(t, axis=1))(x_att)
    x_avg = GlobalAveragePooling1D()(x)
    x = Concatenate()([x_att, x_avg])
    x = Dropout(dropout_pool)(x)

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