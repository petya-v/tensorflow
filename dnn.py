import numpy as np
from tensorflow import keras


def add_dense_layers(seq_model):
    seq_model.add(keras.layers.Dense(5, activation='softmax'))
    seq_model.add(keras.layers.Dense(10, activation='relu'))
    # seq_model.add(keras.layers.Dense(32, activation='relu'))

    # 0 = false     1 = true
    seq_model.add(keras.layers.Dense(1, activation='sigmoid'))
    return seq_model


def add_lstm_layers(seq_model, embed_dims, lstm_out):
    seq_model.add(keras.layers.Embedding(256, embed_dims, input_shape=[1]))
    seq_model.add(keras.layers.Dropout(0.2))
    seq_model.add(keras.layers.LSTM(lstm_out))
    seq_model.add(keras.layers.Dropout(0.2))

    seq_model.add(keras.layers.Dense(1, activation='sigmoid'))

    return seq_model


def cnn_rnn(seq_model, embed_dims, lstm_out):
    seq_model.add(keras.layers.Embedding(256, embed_dims, input_shape=[1]))
    seq_model.add(keras.layers.Conv1D(filters=2, kernel_size=3, padding='same', activation='relu'))
    seq_model.add(keras.layers.MaxPooling1D(pool_size=1))
    seq_model.add(keras.layers.LSTM(lstm_out))
    seq_model.add(keras.layers.Dense(1, activation='sigmoid'))

    return seq_model


tensorboard = keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0,
                          write_graph=True, write_images=False)


trainDatasetPath = "./data/emg-data-total.csv"
testDatasetPath = "./data/emg-data-total-train.csv"


# Prepare train emg-rand.csv
trainDataArray = np.loadtxt(trainDatasetPath, delimiter=',', skiprows=1, usecols=10, unpack=True)
trainLabel = np.loadtxt(trainDatasetPath, dtype=str, delimiter=',', skiprows=1, usecols=9, unpack=True)

trainDataArray = trainDataArray.reshape(-1, 1)
trainLabel = trainLabel.reshape(-1, 1)

trainLabelArray = np.empty(shape=[0, 1])

for i, value in enumerate(trainLabel):
    if value == "TRUE":
        trainLabelArray = np.append(trainLabelArray, [[1]], axis=0)
    else:
        trainLabelArray = np.append(trainLabelArray, [[0]], axis=0)


# Prepare test emg-rand.csv
testData = np.loadtxt(trainDatasetPath, delimiter=',', skiprows=1, usecols=10, unpack=True)
testData = testData.reshape(-1, 1)


training_data_size = (int(testData.size*0.7))
trainData = testData[:training_data_size]
trainLabel = trainLabelArray[:training_data_size]
testData = testData[training_data_size:]

print(trainLabel[0])


model = keras.models.Sequential()

embed_dimensions = 2
LSTM_out = 32
batch_size = 2

model = add_dense_layers(model)
# model = add_lstm_layers(model, embed_dimensions, LSTM_out)
# model = cnn_rnn(model, embed_dimensions, 5)


model.compile(
  optimizer='adam',
  loss='binary_crossentropy',
  metrics=['accuracy']
)

model.fit(
  trainData,
  trainLabel,
  epochs=1,
  batch_size=2,
  verbose=1,
  callbacks=[tensorboard]
)

output = model.predict_classes(testData)

if 1 in output:
    print('TRUE')
else:
    print('FALSE')

# model.summary()
