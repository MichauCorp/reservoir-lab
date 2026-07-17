import numpy as np

from reservoir_lab.reservoir import ESN


DATA_PATH = "data/UCI HAR Dataset"


# -----------------------------
# Load sensor data
# -----------------------------

def load_signals(split):

    folder = f"{DATA_PATH}/{split}/Inertial Signals"

    x = np.loadtxt(
        f"{folder}/body_acc_x_{split}.txt"
    )

    y = np.loadtxt(
        f"{folder}/body_acc_y_{split}.txt"
    )

    z = np.loadtxt(
        f"{folder}/body_acc_z_{split}.txt"
    )

    data = np.stack(
        [x, y, z],
        axis=2
    )

    return data


# -----------------------------
# Load activity labels
# -----------------------------

def load_labels(split):

    return np.loadtxt(
        f"{DATA_PATH}/{split}/y_{split}.txt"
    ).astype(int)


# -----------------------------
# One-hot encoding
# -----------------------------

def one_hot(labels):

    encoded = np.zeros(
        (len(labels), 6)
    )

    for i, label in enumerate(labels):
        encoded[i, label - 1] = 1

    return encoded


# -----------------------------
# Load dataset
# -----------------------------

X_train = load_signals("train")
y_train = load_labels("train")

X_test = load_signals("test")
y_test = load_labels("test")


print("Train:", X_train.shape)
print("Test :", X_test.shape)


# -----------------------------
# Encode labels
# -----------------------------

y_train_encoded = one_hot(y_train)


# -----------------------------
# Create reservoir
# -----------------------------

esn = ESN(
    n_inputs=3,
    n_reservoir=500,
    spectral_radius=1.2,
    sparsity=0.1
)


# -----------------------------
# Train readout
# -----------------------------

states = esn.fit_sequences(
    X_train,
    y_train_encoded
)


# -----------------------------
# Predict
# -----------------------------

pred = esn.predict_sequences(
    X_test
)


pred_labels = (
    np.argmax(pred, axis=1)
    + 1
)


# -----------------------------
# Evaluate
# -----------------------------

accuracy = np.mean(
    pred_labels == y_test
)


print("======================")
print("Human Activity Results")
print("======================")
print(
    f"Accuracy: {accuracy:.4f}"
)