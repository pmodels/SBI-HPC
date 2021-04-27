#!/usr/bin/env python

import sys
import tensorflow as tf
from skimage.transform import resize
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error as mse
from keras.models import load_model

path = "/home/beams/XYU/ptychoNN/PtychoNN/TF2"
sys.path.append(path)

gpus = tf.config.experimental.list_physical_devices("GPU")
print(gpus)
if len(gpus):
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            print("Restricting Memory")
    except RuntimeError as e:
        print(e)


# Config the matplotlib backend as plotting inline in IPython


X_test = np.load(path + "../data/X_test.npy")
Y_I_test = np.load(path + "../data/Y_I_test.npy")
Y_phi_test = np.load(path + "../data/Y_phi_test.npy")
nltest = int(Y_I_test.shape[0] ** 0.5)

wt_path = path + "wts4"  # Where to read network weights
min_epoch = np.load("%s/min_epoch.npy" % wt_path)

model = load_model("%s/weights.%02d.hdf5" % (wt_path, min_epoch))
print("Loaded model from epoch:%d" % min_epoch)

preds_intens = model.predict(X_test)
print(preds_intens[0].shape)

h, w = 64, 64
ntest = preds_intens[0].shape[0]
plt.viridis()
n = 5
f, ax = plt.subplots(7, n, figsize=(20, 15))
plt.gcf().text(0.02, 0.85, "Input", fontsize=20)
plt.gcf().text(0.02, 0.72, "True I", fontsize=20)
plt.gcf().text(0.02, 0.6, "Predicted I", fontsize=20)
plt.gcf().text(0.02, 0.5, "Difference I", fontsize=20)
plt.gcf().text(0.02, 0.4, "True Phi", fontsize=20)
plt.gcf().text(0.02, 0.27, "Predicted Phi", fontsize=20)
plt.gcf().text(0.02, 0.17, "Difference Phi", fontsize=20)

for i in range(0, n):
    j = int(round(np.random.rand() * ntest))

    # display FT
    im = ax[0, i].imshow(np.log10(X_test[j].reshape(h, w) + 1))
    plt.colorbar(im, ax=ax[0, i], format="%.2f")
    ax[0, i].get_xaxis().set_visible(False)
    ax[0, i].get_yaxis().set_visible(False)

    # display original intens
    im = ax[1, i].imshow(Y_I_test[j].reshape(h, w))
    plt.colorbar(im, ax=ax[1, i], format="%.2f")
    ax[1, i].get_xaxis().set_visible(False)
    ax[1, i].get_yaxis().set_visible(False)

    # display predicted intens
    im = ax[2, i].imshow(preds_intens[0][j].reshape(h, w))
    plt.colorbar(im, ax=ax[2, i], format="%.2f")
    ax[2, i].get_xaxis().set_visible(False)
    ax[2, i].get_yaxis().set_visible(False)

    # display original phase
    im = ax[4, i].imshow(Y_phi_test[j].reshape(h, w))
    plt.colorbar(im, ax=ax[4, i], format="%.2f")
    ax[4, i].get_xaxis().set_visible(False)
    ax[4, i].get_yaxis().set_visible(False)

    # display predicted phase
    im = ax[5, i].imshow(preds_intens[1][j].reshape(h, w))
    plt.colorbar(im, ax=ax[5, i], format="%.2f")
    ax[5, i].get_xaxis().set_visible(False)
    ax[5, i].get_yaxis().set_visible(False)

    # Difference in amplitude
    im = ax[3, i].imshow(Y_I_test[j].reshape(h, w) - preds_intens[0][j].reshape(h, w))
    plt.colorbar(im, ax=ax[3, i], format="%.2f")
    ax[3, i].get_xaxis().set_visible(False)
    ax[3, i].get_yaxis().set_visible(False)

    # Difference in phase
    im = ax[6, i].imshow(Y_phi_test[j].reshape(h, w) - preds_intens[1][j].reshape(h, w))
    plt.colorbar(im, ax=ax[6, i], format="%.2f")
    ax[6, i].get_xaxis().set_visible(False)
    ax[6, i].get_yaxis().set_visible(False)

plt.show()

tst_side = 60
fig, ax = plt.subplots(3, 3, figsize=(12, 6))
ax[0, 0].imshow((preds_intens[1].reshape(tst_side, tst_side, 64, 64)[9, 9]))
ax[0, 1].imshow((preds_intens[1].reshape(tst_side, tst_side, 64, 64)[9, 10]))
ax[0, 2].imshow((preds_intens[1].reshape(tst_side, tst_side, 64, 64)[9, 11]))
ax[1, 0].imshow((preds_intens[1].reshape(tst_side, tst_side, 64, 64)[10, 9]))
ax[1, 1].imshow((preds_intens[1].reshape(tst_side, tst_side, 64, 64)[10, 10]))
ax[1, 2].imshow((preds_intens[1].reshape(tst_side, tst_side, 64, 64)[10, 11]))
ax[2, 0].imshow((preds_intens[1].reshape(tst_side, tst_side, 64, 64)[11, 9]))
ax[2, 1].imshow((preds_intens[1].reshape(tst_side, tst_side, 64, 64)[11, 10]))
ax[2, 2].imshow((preds_intens[1].reshape(tst_side, tst_side, 64, 64)[11, 11]))

point_size = 3
overlap = 4 * point_size


composite_amp = np.zeros(
    (tst_side * point_size + overlap, tst_side * point_size + overlap), float
)
ctr = np.zeros_like(composite_amp)
data_reshaped = preds_intens[0].reshape(tst_side, tst_side, 64, 64)[
    :,
    :,
    32 - int(overlap / 2): 32 + int(overlap / 2),
    32 - int(overlap / 2): 32 + int(overlap / 2),
]

for i in range(tst_side):
    for j in range(tst_side):
        composite_amp[
            point_size * i: point_size * i + overlap,
            point_size * j: point_size * j + overlap,
        ] += data_reshaped[i, j]
        ctr[
            point_size * i: point_size * i + overlap,
            point_size * j: point_size * j + overlap,
        ] += 1


composite_phase = np.zeros(
    (tst_side * point_size + overlap, tst_side * point_size + overlap), float
)
ctr = np.zeros_like(composite_phase)
data_reshaped = preds_intens[1].reshape(tst_side, tst_side, 64, 64)[
    :,
    :,
    32 - int(overlap / 2): 32 + int(overlap / 2),
    32 - int(overlap / 2): 32 + int(overlap / 2),
]

for i in range(tst_side):
    for j in range(tst_side):
        composite_phase[
            point_size * i: point_size * i + overlap,
            point_size * j: point_size * j + overlap,
        ] += data_reshaped[i, j]
        ctr[
            point_size * i: point_size * i + overlap,
            point_size * j: point_size * j + overlap,
        ] += 1


stitched_phase = (
    composite_phase[
        int(overlap / 2): -int(overlap / 2), int(overlap / 2): -int(overlap / 2)
    ]
    / ctr[int(overlap / 2): -int(overlap / 2), int(overlap / 2): -int(overlap / 2)]
)

stitched_amp = (
    composite_amp[
        int(overlap / 2): -int(overlap / 2), int(overlap / 2): -int(overlap / 2)
    ]
    / ctr[int(overlap / 2): -int(overlap / 2), int(overlap / 2): -int(overlap / 2)]
)

stitched_amp_down = resize(
    stitched_amp, (60, 60), preserve_range=True, anti_aliasing=True
)
stitched_phase_down = resize(
    stitched_phase, (60, 60), preserve_range=True, anti_aliasing=True
)

amp = Y_I_test.reshape(nltest, nltest, 64, 64)
ph = Y_phi_test.reshape(nltest, nltest, 64, 64)

plt.rcParams.update({"font.size": 26})
fig, ax = plt.subplots(1, 3, figsize=(20, 12))

im = ax[0].imshow(amp[:, :, 32, 32])  # ,vmin=minc, vmax=maxc)
ax[0].axis("off")
plt.colorbar(im, ax=ax[0], fraction=0.046, pad=0.04)

im = ax[1].imshow(stitched_amp_down)  # ,vmin=minc, vmax=maxc)
plt.colorbar(im, ax=ax[1], fraction=0.046, pad=0.04)
ax[1].axis("off")

im = ax[2].imshow(stitched_amp_down - amp[:, :, 32, 32])
plt.colorbar(im, ax=ax[2], fraction=0.046, pad=0.04)
ax[2].axis("off")


fig, ax = plt.subplots(1, 3, figsize=(20, 10))

im = ax[0].imshow(ph[:, :, 32, 32])  # ,vmin=minc, vmax=maxc)
plt.colorbar(im, ax=ax[0], fraction=0.046, pad=0.04)
ax[0].axis("off")

im = ax[1].imshow(stitched_phase_down)  # ,vmin=minc, vmax=maxc)
plt.colorbar(im, ax=ax[1], fraction=0.046, pad=0.04)
ax[1].axis("off")

im = ax[2].imshow(stitched_phase_down - ph[:, :, 32, 32], vmin=-1.35, vmax=1.2)
plt.colorbar(im, ax=ax[2], fraction=0.046, pad=0.04)
ax[2].axis("off")

(stitched_phase_down - ph[:, :, 32, 32]).max()

print("MSE in amplitude: ", mse(stitched_amp_down, amp[:, :, 32, 32]))
print("MSE in phase: ", mse(stitched_phase_down, ph[:, :, 32, 32]))
