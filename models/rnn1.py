from tensorflow.keras import models, layers, preprocessing as kprocessing
from tensorflow.keras.models import Sequential

from models.baseline_rnn import BaselineRNN


class RNN1(BaselineRNN):

    def _construct_model(self, input_shape, output_size=1):
        self.time_steps = input_shape[1]
        model = Sequential()
        model.add(layers.LSTM(50, input_shape=input_shape, return_sequences=True))
        model.add(layers.Dropout(0.2))
        model.add(layers.LSTM(50, return_sequences=True))
        model.add(layers.Dropout(0.2))
        model.add(layers.LSTM(50))
        model.add(layers.Dropout(0.2))
        model.add(layers.Dense(1, activation='sigmoid', input_dim=input_shape[1]))

        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.model = model
