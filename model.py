import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv1D, Dense, Dropout,
    BatchNormalization, GlobalAveragePooling1D
)
from sklearn.preprocessing import StandardScaler

train = pd.read_csv('train.csv')
test  = pd.read_csv('test.csv')

sensors = [f'sensor_{i}' for i in range(25)]

def prepare_data(df):
    sequences = []
    for seq_id, group in df.groupby('sequence_id'):
        group = group.sort_values('timestep')
        sequences.append(group[sensors].values)
    return np.array(sequences)

X_all     = prepare_data(train)
X_test_3d = prepare_data(test)
y_all     = train.groupby('sequence_id')['level'].first().astype(int).values

scaler        = StandardScaler()
X_all_scaled  = scaler.fit_transform(X_all.reshape(-1, 25)).reshape(-1, 10, 25)
X_test_scaled = scaler.transform(X_test_3d.reshape(-1, 25)).reshape(-1, 10, 25)

model = Sequential([
    # Block 1
    Conv1D(64, kernel_size=2, activation='relu', padding='same', input_shape=(10, 25)),
    BatchNormalization(),
    Dropout(0.2),

    # Block 2
    Conv1D(128, kernel_size=2, activation='relu', padding='same'),
    BatchNormalization(),
    Dropout(0.2),

    # Block 3 — stacked!
    Conv1D(256, kernel_size=2, activation='relu', padding='same'),
    BatchNormalization(),
    Dropout(0.2),

    Conv1D(256, kernel_size=2, activation='relu', padding='same'),
    BatchNormalization(),
    Dropout(0.2),

    # Block 4 — new! stacked 512
    Conv1D(512, kernel_size=2, activation='relu', padding='same'),
    BatchNormalization(),
    Dropout(0.2),

    Conv1D(512, kernel_size=2, activation='relu', padding='same'),
    BatchNormalization(),
    Dropout(0.2),

    GlobalAveragePooling1D(),

    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),

    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.2),

    Dense(64, activation='relu'),
    Dropout(0.2),

    Dense(4, activation='softmax')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

history = model.fit(
    X_all_scaled, y_all,
    epochs=80,
    batch_size=32,
    verbose=1
)

train_acc = model.evaluate(X_all_scaled, y_all, verbose=0)[1]
model.save("model.h5")