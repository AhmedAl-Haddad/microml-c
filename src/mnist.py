import numpy as np
import tensorflow_datasets as tfds


def ds_to_numpy(ds):
    images = []
    labels = []

    for images, label in ds:
        images.append(images.numpy())
        labels.append(label.numpy())

    return np.array(images), np.array(labels)


train_ds, test_ds = tdfs.load("mnist", split=["train", "test"], as_supervised=True)

train_images, train_labels = ds_to_numpy(train_ds)
test_images, test_labels = ds_to_numpy(test_ds)

train_images = train_images.astype(np.float32) / 255.0
test_images = test_images.astype(np.float32) / 255.0

train_labels = train_labels.astype(np.float32)
test_labels = test_labels.astype(np.float32)
