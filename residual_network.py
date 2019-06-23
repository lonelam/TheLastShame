from keras.layers import LSTM, Lambda, Input, Dense
from keras.layers.merge import add
from keras import Model

def get_compiled_residual_model(input_shape, width=256):
    input = Input(input_shape)
    x = LSTM(width, activation="tanh", return_sequences=True, dropout=0.2)(input)
    residual_from = x
    x = LSTM(width, activation='tanh', return_sequences=True, dropout=0.2)(x)
    x = LSTM(width, activation="tanh", return_sequences=True, dropout=0.2)(x)
    x = add([residual_from, x])
    # residual_from = x
    x = LSTM(width, activation="tanh", return_sequences=True, dropout=0.2)(x)
    x = LSTM(width, activation="tanh", return_sequences=True, dropout=0.2)(x)
    x = add([residual_from, x])
    x = LSTM(width, activation="tanh", return_sequences=True, dropout=0.2)(x)

    x = LSTM(width, activation="tanh", return_sequences=False)(x)
    output = Dense(input_shape[1])(x)
    model = Model(inputs=input, outputs=output)
    model.compile(optimizer="rmsprop", loss='mse', metrics=['mae'])
    model.summary()
    return model

if __name__ == '__main__':
    model = get_compiled_residual_model((2, 128))
