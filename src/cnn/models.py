import keras_tuner as kt
import tensorflow as tf
from keras.layers import (
    Add,
    BatchNormalization,
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


def make_model_res_net1D(input_shape):
    inputs = Input(shape=input_shape)

    # ── Bloque 1 ─────────────────────────────
    x = Conv1D(32, 5, padding="same", kernel_regularizer=l2(1e-4))(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Dropout(0.2)(x)

    # ── Bloque 2 (residual) ─────────────────
    shortcut = Conv1D(64, 1, padding="same")(x)

    x = Conv1D(64, 5, padding="same", kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Dropout(0.3)(x)

    x = Conv1D(64, 3, padding="same", kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = Add()([x, shortcut])
    x = ReLU()(x)
    x = LayerNormalization()(x)

    # ── Bloque 3 (residual) ─────────────────
    shortcut2 = x

    x = Conv1D(64, 3, padding="same", kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Dropout(0.3)(x)

    x = Conv1D(64, 3, padding="same", kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = Add()([x, shortcut2])
    x = ReLU()(x)
    x = LayerNormalization()(x)

    # ── Attention Pooling ────────────────────────────────────────────────────
    score = Dense(1, use_bias=False)(x) 
    weight = Softmax(axis=1)(score)
    x_att = Multiply()([x, weight])  # pondera cada frame
    x_att = Lambda(lambda t: tf.reduce_sum(t, axis=1))(x_att)

    x_avg = GlobalAveragePooling1D()(x)

    x = Concatenate()([x_att, x_avg])
    x = Dropout(0.5)(x)

    # ── Clasificador ────────────────────────
    x = Dense(32, activation="relu")(x)
    x = Dropout(0.4)(x)

    outputs = Dense(1, activation="sigmoid")(x)

    model = Model(inputs, outputs)
    model.compile(
        optimizer=Adam(learning_rate=5e-4),
        loss="binary_crossentropy",
        metrics=[AUC(name="pr_auc", curve="PR")],
    )

    return model


def make_model_tcn(hp,input_shape):
    """
    TCN para clasificación binaria de secuencias temporales.
    """
    inputs = Input(shape=input_shape)

    x = TCN(
        nb_filters=32,
        kernel_size=5,
        dilations=[1, 2, 4, 8, 16],
        padding="causal",
        use_skip_connections=True,
        use_batch_norm=True,
        dropout_rate=0.4,
        return_sequences=True,  
        name="tcn",
    )(inputs)

    score = Dense(1, use_bias=False)(x)  
    weight = Softmax(axis=1)(score)  
    x_att = Multiply()([x, weight])  # pondera cada frame
    x_att = Lambda(lambda t: tf.reduce_sum(t, axis=1))(x_att)

    x_avg = GlobalAveragePooling1D()(x)

    x = Concatenate()([x_att, x_avg])
    x = Dropout(0.5)(x)

    # ── Clasificador ──
    x = Dense(16, activation="relu", kernel_regularizer=l2(1e-3))(x)
    x = Dropout(0.4)(x)
    outputs = Dense(1, activation="sigmoid")(x)

    model = Model(inputs, outputs)
    model.compile(
        optimizer=Adam(learning_rate=5e-4),
        loss="binary_crossentropy",
        metrics=[AUC(name="pr_auc", curve="PR")],
    )
    return model
