from tensorflow.keras import models, layers, preprocessing as kprocessing
from tensorflow.keras.models import Sequential

from models.baseline_nn import BaselineNN


class DNN1(BaselineNN):

    def _construct_model(self, input_shape, output_size=1):
        model = Sequential()
        # model.add(layers.experimental.preprocessing.Normalization())
        # model.add(layers.Dense(1, input_dim=input_size, activation='relu'))

        model.add(layers.Dense(256, input_dim=input_shape[1], activation='relu'))
        model.add(layers.Dropout(0.3))
        model.add(layers.Dense(256, activation='relu'))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(80, activation='relu'))
        # model.add(layers.Dense(1, activation='softmax'))

        model.add(layers.Dense(1, activation='sigmoid', input_dim=input_shape[1]))

        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        # %% Save Model
        self.model = model
