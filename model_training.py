import os

import numpy as np
from sklearn.model_selection import train_test_split

from tensorflow.python.keras.utils.np_utils import to_categorical

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import LSTM, Dense

actions = np.array(['hello', 'thanks', 'iloveyou'])
data_path = os.path.join('mp data')
no_sequences = 30
sequence_length = 30

label_map = {label: num for num, label in enumerate(actions)}

sequences = []
labels = []

for action in actions:
    for sequence in range(no_sequences):
        window = []
        for frame_num in range(sequence_length):
            res = np.load(os.path.join(data_path, action, str(sequence), "{}.npy".format(frame_num)))
            window.append(res)
        sequences.append(window)
        labels.append(label_map[action])

x = np.array(sequences)

y = to_categorical(labels).astype(int)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.05)
print(y_test.shape)

model = Sequential()

model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30, 1662)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
model.fit(x_train, y_train, epochs=1500)
model.save(os.path.join('action_detection.h5'))
