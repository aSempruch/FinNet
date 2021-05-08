from tensorflow.keras import models, layers, preprocessing as kprocessing
from tensorflow.keras.models import Sequential

from models.baseline_nn import BaselineNN
from util import TEST


class BaselineRNN(BaselineNN):

    def train_sequences(self, x, y, input_shape, epochs=None):
        self._construct_model(input_shape)

        if TEST: epochs = 1

        self.model.fit(x, y, epochs=epochs)
        # self.model.train_on_batch(
        #     sequence_list
        # )
        # for sequence in sequence_list:
        #     self.model.fit(sequence, epochs=epochs)

    def _construct_model(self, input_shape, output_size=1):
        # %%
        self.time_steps = input_shape[1]
        model = Sequential()
        model.add(layers.LSTM(1, input_shape=input_shape))
        model.add(layers.Dense(1, activation='sigmoid', input_dim=input_shape[1]))

        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.model = model
