# Import Libraries and Packages
import os
import sys
import tensorflow as tf
import keras
import platform
import matplotlib.pyplot as plt
import numpy as np
from numpy import zeros, ones
from numpy.random import randn, randint
import cv2
from keras.layers import Dense, Reshape, Flatten, LeakyReLU, Conv2DTranspose, Conv2D, Dropout, Embedding, Concatenate
from keras.layers import Input
from keras.models import Model
from keras.optimizers import Adam
from sklearn import preprocessing
from numpy import asarray
from keras.models import load_model

# Versions
print(f"Python Platform: {platform.platform()}")
print(f"Tensor Flow Version: {tf.__version__}")
print(f"Keras Version: {keras.__version__}")
print(f"Python {sys.version}")

# Set Artists Names for 10 Artists, with 100 Arts per each
list = ['Alfred_Sisley',
        'Amedeo_Modigliani',
        'Gustav_Klimt',
        'Marc_Chagall',
        'Pablo_Picasso',
        'Paul_Klee',
        'Peter_Paul_Rubens',
        'Pieter_Bruegel',
        'Raphael',
        'Rembrandt']
print(f"Number of Artists: {len(list)}")

# Import dataset with Arts and Artist Names
print("Importing the Dataset...")
directory = '/Users/Sachith/dataset/Arts'
categories = list

directory = directory
categories = categories

data = []
labels = []

for category in categories:
    path = os.path.join(directory, category)
    for img in os.listdir(path):
        img_path = os.path.join(path, img)
        image = cv2.imread(img_path)
        data.append(image)
        labels.append(category)

n_arts = f"Number of Arts: {len(data)} loaded"
n_artists = f"Number of Artists: {len(list)}"

print(n_arts)
print(n_artists)
print("Loading Completed!!")

# Resize to Similar Size
# Set the image dimensions to resize
width = 128
height = 128

print("Resizing the Arts...")
dim = (width, height)

resized_images = []

for image in data:
    resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    resized_images.append(resized)

arts = np.array(resized_images)

print(f"Resizing Completed!! New size is {width}x{height}")
artist_names = np.array(labels)

# Assign some useful parameters
image_shape = arts[0].shape
print(f"Image shape is: {image_shape}")

# Label Encoding
le = preprocessing.LabelEncoder()
le.fit(artist_names)
artist_names = le.transform(artist_names).astype('uint8')


# Load function for training the model (Returning the art and artist name)
def load_arts(arts=arts, artist_names=artist_names):
    arts = arts.astype('float32')  # Convert to floats
    arts = arts / 255  # Rescale
    artist_names = artist_names.reshape(-1, 1)
    return [arts, artist_names]


## Build the Discriminator
def build_discriminator(input_shape=(128, 128, 3), n_artists=10):
    ## Considering the labels
    label = Input(shape=(1,))  ## Label input
    ## Each artist will be represented by a vector of size 50
    label_input = Embedding(n_artists, 50)(label)
    ## Image dimensions
    dim_in = input_shape[0] * input_shape[1]  ## 128x128
    label_input = Dense(dim_in)(label_input)
    ## Reshape
    label_input = Reshape((input_shape[0], input_shape[1], 1))(
        label_input)  ## 128x128x1 This is the label input (As a channel)

    ## Image
    input_image = Input(shape=input_shape)  ## 128x128x3
    ## Add the label as a new channel
    merge = Concatenate()([input_image, label_input])  ## 128x128x4 (3 for image and 1 for label)

    down = Conv2D(128, (3, 3), strides=(2, 2), padding='same')(merge)  ## 64x64x128
    down - LeakyReLU(alpha=0.2)(down)

    down = Conv2D(128, (3, 3), strides=(2, 2), padding='same')(down)  ## 32x32x128
    down - LeakyReLU(alpha=0.2)(down)

    down = Flatten()(down)
    down = Dropout(0.4)(down)

    output_layer = Dense(1, activation='sigmoid')(down)

    ## Input image + Input label
    model = Model([input_image, label], output_layer)

    ## Compile
    optimizer = Adam(learning_rate=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return model


## Testing the discriminator
test_discriminator = build_discriminator()
print(test_discriminator.summary())


def build_generator(noise_dim=100, n_artists=10):
    label = Input(shape=(1,))
    ## Embedding for label
    label_input = Embedding(n_artists, 50)(label)
    ## Label input dimension
    dim_in = 32 * 32

    label_input = Dense(dim_in)(label_input)
    label_input = Reshape((32, 32, 1))(label_input)

    input_noise = Input(shape=(noise_dim,))  ## Input dimension: 100

    dim_in = 32 * 32 * 128
    generator = Dense(dim_in)(input_noise)  ## 131072
    generator = LeakyReLU(alpha=0.2)(generator)
    generator = Reshape((32, 32, 128))(generator)  ## 32x32x128

    merge = Concatenate()([generator, label_input])  ## 32x32x(128+1)

    generator = Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same')(merge)
    generator = LeakyReLU(alpha=0.2)(generator)

    generator = Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same')(generator)
    generator = LeakyReLU(alpha=0.2)(generator)

    output_layer = Conv2D(3, (8, 8), activation='sigmoid', padding='same')(generator)

    model = Model([input_noise, label], output_layer)

    return model


## Testing the generator
test_generator = build_generator()
print(test_generator.summary())
print(f"The output image shape is {test_generator.output_shape}")


## Combine the GAN model
def build_gan(discriminator, generator):
    discriminator.trainable = False  ## Since we train the dicsriinator separately

    ## Connect
    generated_noise, generated_label = generator.input
    generated_output = generator.output  ## 128x128x3

    ## Go through the discriminator
    gan_output = discriminator([generated_output, generated_label])
    model = Model([generated_noise, generated_label], gan_output)

    optimizer = Adam(learning_rate=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=optimizer)

    return model


## Helper Functions for Training Loop

def real_samples(arts_artists, n_samples):
    # split into images and labels
    images, labels = arts_artists
    # choose random instances
    ix = randint(0, images.shape[0], n_samples)
    # select images and labels
    X, labels = images[ix], labels[ix]
    # generate class labels and assign to y (don't confuse this with the above labels that correspond to cifar labels)
    y = ones((n_samples, 1))  # Label=1 indicating they are real
    return [X, labels], y


def random_noise(noise_dim, samples, n_artists=10):
    generator_input = randn(noise_dim * samples)
    ## Reshape
    generator_input = generator_input.reshape(samples, noise_dim)
    ## Generate Categories(Artists)
    artists_labels = randint(0, n_artists, samples)
    return [generator_input, artists_labels]


def fake_samples(generator, noise_dim, samples):
    generator_input, artists = random_noise(noise_dim, samples)
    arts = generator.predict([generator_input, artists])
    y = zeros((samples, 1))  ## Since the fake arts
    return [arts, artists], y


def train_the_GAN(generator_model, discriminator_model, gan_model, arts_artists, noise_dim=100, epochs=100,
                  batch_size=64):
    batches_per_epoch = int(arts_artists[0].shape[0] / batch_size)
    half_batch = int(batch_size / 2)

    ## We train the discriminator with half batch size real arts and fake arts.
    epoch_loss_discriminator = []
    epoch_loss_generator = []

    for epoch in range(epochs):
        discriminator_loss_acc = 0.
        generator_loss_acc = 0.

        for batch_per_epoch in range(batches_per_epoch):
            print(f"epoch: {epoch + 1} ----- {batch_per_epoch + 1}/{batches_per_epoch}")

            [real_arts, real_artists], real_labels = real_samples(arts_artists, half_batch)
            discriminator_loss_real, _ = discriminator_model.train_on_batch([real_arts, real_artists], real_labels)

            [fake_arts, fake_artists], fake_labels = fake_samples(generator_model, noise_dim, half_batch)
            discriminator_loss_fake, _ = discriminator_model.train_on_batch([fake_arts, fake_artists], fake_labels)

            discriminator_loss = 0.5 * np.add(discriminator_loss_real, discriminator_loss_fake)

            [generator_input, artists_labels] = random_noise(noise_dim, batch_size)
            y_gan = ones((batch_size, 1))

            generator_loss = gan_model.train_on_batch([generator_input, artists_labels], y_gan)
            discriminator_loss_acc += discriminator_loss
            generator_loss_acc += generator_loss

        epoch_loss_discriminator.append(discriminator_loss_acc / batches_per_epoch)
        epoch_loss_generator.append(generator_loss_acc / batches_per_epoch)

    ## Save the Generator
    generator_model.save('Rootcode_conditional.h5')

    ## Plot the loss
    x = []
    for i in range(epochs):
        x.append(i + 1)
        i = i + 1
    y1points = np.array(epoch_loss_discriminator)
    y2points = np.array(epoch_loss_generator)
    xpoints = np.array(x)
    plt.plot(xpoints, y1points, xpoints, y2points)
    plt.show()


## Define Training Parameters
generator = build_generator()
discriminator = build_discriminator()
gan = build_gan(discriminator, generator)
arts_artists = load_arts()
noise_dim = 100
epochs = 200
batch_size = 64

## Train the Combined GAN model and Save the Generator
train_the_GAN(generator, discriminator, gan, arts_artists, noise_dim, epochs, batch_size)

## Import Saved Generator and Generate Arts!!!
# load model
model = load_model('Rootcode_conditional.h5')
noise, labels = random_noise(100, 100)
# specify labels - generate 10 sets of labels each gping from 0 to 9
labels = asarray([x for i in range(10) for x in range(10)])
# generate images
art = model.predict([noise, labels])
## Rescale to 0-256
art = (art * 255).astype(np.uint8)


# plot the result (10 sets of images, all images in a column should be of same class in the plot)
# Plot generated images 
def show_plot(examples, n):
    for i in range(n * n):
        plt.subplot(n, n, 1 + i)
        plt.axis('off')
        plt.imshow(examples[i, :, :, :])
    plt.show()


show_plot(art, 10)

print("END!")
