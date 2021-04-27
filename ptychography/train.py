#!/usr/bin/env python

import os
import argparse
import sys
import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras import Model
import matplotlib.pyplot as plt
import numpy as np
import glob
import matplotlib
from keras_helper import Conv_Pool_block, Conv2D, Conv_Up_block
from skimage.transform import resize
from tqdm import tqdm
from sklearn.utils import shuffle

path = "/home/beams/XYU/ptychoNN/PtychoNN/TF2/"

parser = argparse.ArgumentParser()
parser.add_argument("--save", action="store_true")
parser.add_argument("--nepochs", type=int, default=75)
args = parser.parse_args()
nepochs = args.nepochs

gpus = tf.config.experimental.list_physical_devices("GPU")
print(gpus, file=sys.stderr)
if len(gpus):
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            print("Restricting Memory", file=sys.stderr)
    except RuntimeError as e:
        print(e, file=sys.stderr)

h, w = 64, 64
wt_path = "wts4"  # Where to store network weights
batch_size = 32

if not os.path.isdir(path + wt_path):
    os.mkdir(path + wt_path)
data_diffr = np.load(path + "../data/20191008_39_diff.npz")["arr_0"]

print(data_diffr.shape, file=sys.stderr)

data_diffr_red = np.zeros((data_diffr.shape[0], data_diffr.shape[1], 64, 64), float)
for i in tqdm(range(data_diffr.shape[0])):
    for j in range(data_diffr.shape[1]):
        data_diffr_red[i, j] = resize(
            data_diffr[i, j, 32:-32, 32:-32],
            (64, 64),
            preserve_range=True,
            anti_aliasing=True,
        )
        data_diffr_red[i, j] = np.where(
            data_diffr_red[i, j] < 3, 0, data_diffr_red[i, j]
        )
real_space = np.load(path + "../data/20191008_39_amp_pha_10nm_full.npy")
amp = np.abs(real_space)
ph = np.angle(real_space)
amp.shape
fig, ax = plt.subplots(1, 2, figsize=(20, 10))
ax[0].imshow(amp[:, :, 32, 32])
ax[1].imshow(ph[:, :, 32, 32])
nlines = 100  # How many lines of data to use for training?
nltest = 60  # How many lines for the test set?
tst_strt = amp.shape[0] - nltest  # Where to index from
print(tst_strt, file=sys.stderr)

X_train = data_diffr_red[:nlines, :].reshape(-1, h, w)[:, :, :, np.newaxis]
X_test = data_diffr_red[tst_strt:, tst_strt:].reshape(-1, h, w)[:, :, :, np.newaxis]
Y_I_train = amp[:nlines, :].reshape(-1, h, w)[:, :, :, np.newaxis]
Y_I_test = amp[tst_strt:, tst_strt:].reshape(-1, h, w)[:, :, :, np.newaxis]
Y_phi_train = ph[:nlines, :].reshape(-1, h, w)[:, :, :, np.newaxis]
Y_phi_test = ph[tst_strt:, tst_strt:].reshape(-1, h, w)[:, :, :, np.newaxis]

ntrain = X_train.shape[0] * X_train.shape[1]
ntest = X_test.shape[0] * X_test.shape[1]

print(X_train.shape, X_test.shape, file=sys.stderr)

X_train, Y_I_train, Y_phi_train = shuffle(
    X_train, Y_I_train, Y_phi_train, random_state=0
)

if args.save:
    np.save("data/X_test.npy", X_test)  # Diffraction data
    np.save("data/Y_I_test.npy", Y_I_test)  # Intensity data
    np.save("data/Y_phi_test.npy", Y_phi_test)  # Intensity data


tf.keras.backend.clear_session()
np.random.seed(123)

files = glob.glob("%s/*" % wt_path)
for file in files:
    os.remove(file)


input_img = Input(shape=(h, w, 1))

x = Conv_Pool_block(
    input_img, 32, w1=3, w2=3, p1=2, p2=2, padding="same", data_format="channels_last"
)
x = Conv_Pool_block(
    x, 64, w1=3, w2=3, p1=2, p2=2, padding="same", data_format="channels_last"
)
x = Conv_Pool_block(
    x, 128, w1=3, w2=3, p1=2, p2=2, padding="same", data_format="channels_last"
)
# Activations are all ReLu

encoded = x

# Decoding arm 1
x1 = Conv_Up_block(
    encoded, 128, w1=3, w2=3, p1=2, p2=2, padding="same", data_format="channels_last"
)
x1 = Conv_Up_block(
    x1, 64, w1=3, w2=3, p1=2, p2=2, padding="same", data_format="channels_last"
)
x1 = Conv_Up_block(
    x1, 32, w1=3, w2=3, p1=2, p2=2, padding="same", data_format="channels_last"
)

decoded1 = Conv2D(1, (3, 3), padding="same")(x1)


# Decoding arm 2
x2 = Conv_Up_block(
    encoded, 128, w1=3, w2=3, p1=2, p2=2, padding="same", data_format="channels_last"
)
x2 = Conv_Up_block(
    x2, 64, w1=3, w2=3, p1=2, p2=2, padding="same", data_format="channels_last"
)
x2 = Conv_Up_block(
    x2, 32, w1=3, w2=3, p1=2, p2=2, padding="same", data_format="channels_last"
)

decoded2 = Conv2D(1, (3, 3), padding="same")(x2)

# Put together
autoencoder = Model(input_img, [decoded1, decoded2])

# parallel_model = ModelMGPU(autoencoder, gpus=num_GPU)
# parallel_model.compile(optimizer='adam', loss='mean_absolute_error')
autoencoder.compile(optimizer="adam", loss="mean_absolute_error")

print(autoencoder.summary(), file=sys.stderr)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss", factor=0.5, patience=2, min_lr=0.0001, verbose=1
)

checkpoints = tf.keras.callbacks.ModelCheckpoint(
    "%s/weights.{epoch:02d}.hdf5" % wt_path,
    monitor="val_loss",
    verbose=1,
    save_best_only=True,
    save_weights_only=False,
    mode="auto",
    period=1,
)

history = autoencoder.fit(
    X_train,
    [Y_I_train, Y_phi_train],
    shuffle=True,
    batch_size=batch_size,
    verbose=1,
    epochs=nepochs,
    validation_split=0.05,
    callbacks=[checkpoints, reduce_lr],
)

hist = history
epochs = np.asarray(history.epoch) + 1

plt.style.use("seaborn-white")
matplotlib.rc("font", family="Times New Roman")
matplotlib.rcParams["font.size"] = 20

f, axarr = plt.subplots(3, sharex=True, figsize=(12, 8))

axarr[0].set(ylabel="Loss")
axarr[0].plot(epochs, hist.history["loss"], "C3o", label="Total Training")
axarr[0].plot(epochs, hist.history["val_loss"], "C3-", label="Total Validation")
axarr[0].grid()
axarr[0].legend(loc="center right", bbox_to_anchor=(1.5, 0.5))

axarr[1].set(ylabel="Loss")
axarr[1].plot(epochs, hist.history["conv2d_12_loss"], "C0o", label="Structure Training")
axarr[1].plot(
    epochs, hist.history["val_conv2d_12_loss"], "C0-", label="Structure Validation"
)
axarr[1].legend(loc="center right", bbox_to_anchor=(1.5, 0.5))
plt.xlabel("Epochs")
plt.tight_layout()
axarr[1].grid()


axarr[2].set(ylabel="Loss")
axarr[2].plot(epochs, hist.history["conv2d_19_loss"], "C0o", label="Phase Training")
axarr[2].plot(
    epochs, hist.history["val_conv2d_19_loss"], "C0-", label="Phase Validation"
)
axarr[2].legend(loc="center right", bbox_to_anchor=(1.5, 0.5))
plt.tight_layout()
axarr[2].grid()


np.save(path + "str_history", history.history)

val_losses = hist.history["val_loss"]
min_epoch = np.argmin(val_losses) + 1
print(min_epoch)
np.save(path + "%s/min_epoch" % (wt_path), min_epoch)
