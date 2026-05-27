from typing import Any, Callable

import optuna
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


def make_model_res_net1D(trial: optuna.Trial, input_shape):
    def hp(default, fn: Callable[[optuna.Trial], Any]):
        return fn(trial) if trial is not None else default

    inputs = Input(shape=input_shape)

    filters_b1   = hp(64,   lambda t: t.suggest_categorical("resnet_filters_b1",   [16, 32, 64]))
    filters_b2   = hp(64,   lambda t: t.suggest_categorical("resnet_filters_b2",   [32, 64, 128]))
    kernel_b1    = hp(5,    lambda t: t.suggest_categorical("resnet_kernel_b1",    [3, 5, 7]))
    l2_rate      = hp(1e-4, lambda t: t.suggest_float("resnet_l2",        1e-5, 1e-2, log=True))
    dropout_b1   = hp(0.2,  lambda t: t.suggest_float("resnet_dropout_b1",  0.1, 0.4, step=0.1))
    dropout_b2 = hp(0.3,  lambda t: t.suggest_float("resnet_dropout_b2b3",0.1, 0.4, step=0.1))
    dropout_pool = hp(0.5,  lambda t: t.suggest_float("resnet_dropout_pool",0.3, 0.6, step=0.1))
    dropout_clf  = hp(0.4,  lambda t: t.suggest_float("resnet_dropout_clf", 0.2, 0.5, step=0.1))
    dense_units  = hp(32,   lambda t: t.suggest_categorical("resnet_dense_units",  [16, 32, 64]))
    lr           = hp(5e-4, lambda t: t.suggest_float("resnet_lr",        1e-4, 1e-2, log=True))
    attention = hp(False, lambda t: t.suggest_categorical("resnet_attention",    [True, False]))

    # ── Bloque 1 ──────────────────────────────────────────────
    x = Conv1D(filters_b1, kernel_b1, padding="same", kernel_regularizer=l2(l2_rate))(inputs)
    x = LayerNormalization()(x)
    x = ReLU()(x)
    x = Dropout(dropout_b1)(x)

    # ── Bloque 2 (residual) ───────────────────────────────────
    shortcut = Conv1D(filters_b2, 1, padding="same", kernel_regularizer=l2(l2_rate))(x)
    x = Conv1D(filters_b2, kernel_b1, padding="same", kernel_regularizer=l2(l2_rate))(x)
    x = LayerNormalization()(x)
    x = ReLU()(x)
    x = Conv1D(filters_b2, 3, padding="same", kernel_regularizer=l2(l2_rate))(x)
    x = LayerNormalization()(x)
    x = Add()([x, shortcut])
    x = ReLU()(x)
    x = Dropout(dropout_b2)(x)

    # ── Attention Pooling ─────────────────────────────────────
    if attention:
        score  = Dense(1, use_bias=False)(x)
        weight = Softmax(axis=1)(score)
        x_att  = Multiply()([x, weight])
        x_att  = Lambda(lambda t: tf.reduce_sum(t, axis=1))(x_att) 
        x_avg  = GlobalAveragePooling1D()(x)
        x      = Concatenate()([x_att, x_avg])
        x      = Dropout(dropout_pool)(x)
    else:
        x  = GlobalAveragePooling1D()(x)
        x      = Dropout(dropout_pool)(x)

    # ── Clasificador ──────────────────────────────────────────
    x       = Dense(dense_units, activation="relu", kernel_regularizer=l2(l2_rate))(x)
    x       = Dropout(dropout_clf)(x)
    x       = Dense(max(4,dense_units//4), activation="relu", kernel_regularizer=l2(l2_rate))(x)
    x       = Dropout(dropout_clf)(x)
    outputs = Dense(1, activation="sigmoid")(x)

    model = Model(inputs, outputs)
    model.compile(
        optimizer=Adam(learning_rate=lr, clipnorm=0.6),
        loss="binary_crossentropy",
        metrics=[AUC(name="pr_auc", curve="PR")],
    )
    return model


def make_model_tcn(trial: optuna.Trial, input_shape):
    def hp(default, fn):
        return fn(trial) if trial is not None else default

    DILATIONS_MAP = {
        "small":  [1, 2, 4],
        "medium": [1, 2, 4, 8],
    }

    inputs = Input(shape=input_shape)

    nb_filters    = hp(32,      lambda t: t.suggest_categorical("tcn_filters",      [16, 32, 64]))
    kernel_size   = hp(5,       lambda t: t.suggest_categorical("tcn_kernel",       [3, 5, 7]))
    dropout_tcn   = hp(0.4,     lambda t: t.suggest_float("tcn_dropout",      0.2, 0.5, step=0.1))
    dropout_pool  = hp(0.5,     lambda t: t.suggest_float("tcn_dropout_pool",  0.3, 0.6, step=0.1))
    dropout_clf   = hp(0.4,     lambda t: t.suggest_float("tcn_dropout_clf",   0.2, 0.5, step=0.1))
    dense_units   = hp(16,      lambda t: t.suggest_categorical("tcn_dense_units",  [16, 32, 64]))
    l2_rate       = hp(1e-3,    lambda t: t.suggest_float("tcn_l2",           1e-5, 1e-2, log=True))
    lr            = hp(5e-4,    lambda t: t.suggest_float("tcn_lr",           1e-4, 3e-3, log=True))
    dilations_key = hp("medium", lambda t: t.suggest_categorical("tcn_dilations",    ["small", "medium"]))
    attention = hp(False, lambda t: t.suggest_categorical("resnet_attention",    [True, False]))
    
    dilations     = DILATIONS_MAP[dilations_key]

    x = TCN(
        nb_filters=nb_filters,
        kernel_size=kernel_size,
        dilations=dilations,
        padding="causal",
        use_skip_connections=True,
        use_layer_norm=True,
        dropout_rate=dropout_tcn,
        return_sequences=True,
        name="tcn",
    )(inputs)

    if attention:
        score  = Dense(1, use_bias=False)(x)
        weight = Softmax(axis=1)(score)
        x_att  = Multiply()([x, weight])
        x_att  = Lambda(lambda t: tf.reduce_sum(t, axis=1))(x_att) 
        x_avg  = GlobalAveragePooling1D()(x)
        x      = Concatenate()([x_att, x_avg])
        x      = Dropout(dropout_pool)(x)
    else:
        x  = GlobalAveragePooling1D()(x)
        x      = Dropout(dropout_pool)(x)

    x       = Dense(dense_units, activation="relu", kernel_regularizer=l2(l2_rate))(x)
    x       = Dropout(dropout_clf)(x)
    x       = Dense(max(4,dense_units//4), activation="relu", kernel_regularizer=l2(l2_rate))(x)
    x       = Dropout(dropout_clf)(x)
    outputs = Dense(1, activation="sigmoid")(x)

    model = Model(inputs, outputs)
    model.compile(
        optimizer=Adam(learning_rate=lr, clipnorm=0.6),
        loss="binary_crossentropy",
        metrics=[AUC(name="pr_auc", curve="PR")],
    )
    return model    