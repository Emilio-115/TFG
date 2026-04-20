from keras.layers import Bidirectional, LSTM, Dense, Dropout, Input
from keras.models import Model
from keras.metrics import AUC
from keras.optimizers import Adam

def make_bilstm(input_shape):
    inputs = Input(shape=input_shape)
    
    x = Bidirectional(LSTM(16, return_sequences=False))(inputs)
    
    x = Dropout(0.5)(x)
    x = Dense(16, activation='relu')(x)
    outputs = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs, outputs)
    model.compile(
        optimizer=Adam(learning_rate=5e-4),
        loss="binary_crossentropy",
        metrics=[AUC(name='pr_auc', curve='PR')]
    )

    return model