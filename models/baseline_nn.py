from util import load_merged_data
from tensorflow.keras import models, layers, preprocessing as kprocessing
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
#%%


class BaselineNN:

    def __init__(self):
        self.epochs = 100
        self.model = None
        self.time_steps = None

    def train(self, X, y, epochs=None):
        epochs = self.epochs if epochs is None else epochs
        self._construct_model(X.shape)
        self.model.fit(x=X, y=y, epochs=epochs, validation_split=0.2)

    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, **kwargs):
        return self.model.evaluate(**kwargs)

    def _construct_model(self, input_shape, output_size=1):
        #%% Input
        model = Sequential()

        model.add(layers.Dense(256, input_dim=input_shape[1], activation='relu'))

        model.add(layers.Dense(1, activation='sigmoid', input_dim=input_shape[1]))

        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        #%% Save Model
        self.model = model


    # def _construct_model(self, input_size, output_size=1):
    #     # Input
    #     x_in = layers.Input(shape=(input_size,))
    #
    #     # Embedding TODO: This makes no sense
    #     x = layers.Embedding(input_size,
    #                          input_size,
    #                          trainable=False)(x_in)
    #
    #     # 2 layers bidirectional LSTM
    #     x = layers.Bidirectional(layers.LSTM(units=input_size,
    #                                          dropout=0.2,
    #                                          return_sequences=True))(x)
    #     x = layers.Bidirectional(layers.LSTM(units=input_size,
    #                                          dropout=0.2))(x)
    #
    #     # Final Dense Layer
    #     x = layers.Dense(64, activation='relu')(x)
    #
    #     # Output
    #     y_out = layers.Dense(1, activation='softmax')(x)
    #
    #     # Compile
    #     model = models.Model(x_in, y_out)
    #     model.compile(loss='sparse_categorical_crossentropy',
    #                   optimizer='adam', metrics=['accuracy'])
    #
    #     self.model = model
