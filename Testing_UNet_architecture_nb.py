import os
import gc
import copy
from os import makedirs
import numpy as np
from PIL import Image

from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import TensorBoard, ModelCheckpoint, LambdaCallback

import matplotlib
import matplotlib.pyplot as plt

from Mask_generator import MaskGenerator
from PConv_UNet_model import PConvUnet
import cv2

# Settings
BATCH_SIZE = 3

# Imagenet Rescaling
MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])

# Instantiate mask generator
mask_generator = MaskGenerator(512, 512, 3, rand_seed=42)

# Load image
img = np.array(Image.open('sample_image.jpg').resize((512, 512))) / 255

# Load mask
mask = mask_generator.sample()

# Image + mask
masked_img = copy.deepcopy(img)
masked_img[mask == 0] = 1

# Show side by side
_, axes = plt.subplots(1, 3, figsize=(20, 5))
axes[0].imshow(img)
axes[1].imshow(mask * 255)
axes[2].imshow(masked_img)
#plt.show()


def plot_sample_data(masked, mask, ori, middle_title='Raw Mask'):
    #_, axes = plt.subplots(1, 3, figsize=(20, 5))
    #axes[0].imshow(masked[:, :, :])
    #axes[0].set_title('Masked Input')
    #axes[1].imshow(mask[:, :, :])
    #axes[1].set_title(middle_title)
    #axes[2].imshow(ori[:, :, :])
    #axes[2].set_title('Target Output')
    #plt.show()
    return

def plot_predicted_data(masked, mask, ori, middle_title='Raw Mask'):
    _, axes = plt.subplots(1, 3, figsize=(20, 5))
    axes[0].imshow(masked[:, :, :])
    axes[0].set_title('Masked Input')
    axes[1].imshow(mask[:, :, :])
    axes[1].set_title(middle_title)
    axes[2].imshow(ori[:, :, :])
    axes[2].set_title('Target Output')
    #plt.savefig('predicted_sample_image_1.jpg')

class DataGenerator(ImageDataGenerator):
    def flow(self, x, *args, **kwargs):
        while True:
            # Get augmentend image samples
            ori = next(super().flow(x, *args, **kwargs))

            # Get masks for each image sample
            mask = np.stack([mask_generator.sample() for _ in range(ori.shape[0])], axis=0)

            # Apply masks to all image sample
            masked = copy.deepcopy(ori)
            masked[mask == 0] = 1

            # Yield ([ori, masl],  ori) training batches
            # print(masked.shape, ori.shape)
            gc.collect()
            yield [masked, mask], ori

        # Create datagen


datagen = DataGenerator(
    rotation_range=0,
    width_shift_range=0,
    height_shift_range=0,
    horizontal_flip=True
)

# make log folder and its recipients
makedirs('logs', exist_ok=True)
makedirs('logs/sample_image_1', exist_ok=True)

# Create generator from numpy array
batch = np.stack([img for _ in range(BATCH_SIZE)], axis=0)

generator = datagen.flow(x=batch, batch_size=BATCH_SIZE)

[m1, m2], o1 = next(generator)

plot_sample_data(m1[0], m2[0] * 255, o1[0])

# Instantiate model
model = PConvUnet(vgg_weights='./h5/pytorch_to_keras_vgg16.h5')

model.fit_generator(
    generator,
    verbose=0,
    steps_per_epoch=2000,
    epochs=10,
    callbacks=[
        TensorBoard(
            log_dir='./logs/sample_image_1',
            write_graph=False
        ),
        ModelCheckpoint(
            './logs/sample_image_1/weights.{epoch:02d}-{loss:.2f}.h5',
            monitor='loss',
            save_best_only=True,
            save_weights_only=True
        ),
        LambdaCallback(
            on_epoch_end=lambda epoch, logs: plot_sample_data(
                masked_img,
                model.predict(
                    [
                        np.expand_dims(masked_img, 0),
                        np.expand_dims(mask, 0)
                    ]
                )[0]
                ,
                img,
                middle_title='Prediction'
            )
        )
    ],
);



#model.predict print/plot results

