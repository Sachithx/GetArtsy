import sys
import os
import tensorflow as tf
import keras
import platform
import matplotlib.pyplot as plt
import numpy as np
import cv2
from keras.layers import Dense, Reshape, Flatten, LeakyReLU, Conv2DTranspose, Conv2D, Dropout
from keras.models import Sequential
from keras.optimizers import Adam
from sklearn import preprocessing
from numpy import zeros, ones
from numpy.random import randint, randn
from keras.models import load_model
from numpy.random import randn
import numpy as np
import matplotlib.pyplot as plt

print(f"Python Platform: {platform.platform()}")
print(f"Tensor Flow Version: {tf.__version__}")
print(f"Keras Version: {keras.__version__}")
print(f"Python {sys.version}")

## Creating a list of strings to represent the labels(Artists) of the images
list = []
for i in range(20):
    a = "New Folder With Items "+str(i+1)
    list.append(a)
list
f"Number of Artists: {len(list)}"

print("Importing the Dataset...")
directory = '/Users/Sachith/dataset/Data'
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

## Set the image dimensions to resize
width = 128
height = 128

print("Resizing the Arts...")
dim = (width, height)
resized_images = []

for image in data:
    resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
    resized_images.append(resized)

resized_images = np.array(resized_images)  
print(f"Resizing Completed!! New size is {width}x{height}")
print("Label Encoding!!")
le = preprocessing.LabelEncoder()
le.fit(labels)
labels = le.transform(labels).reshape(-1,1)

## Assign some useful parameters
image_shape = resized_images[0].shape
print(f"Image shape is: {image_shape}")

## Load thr Arts for Training
def load_arts(resized_images, labels):
    (train_images, train_labels) = (resized_images, labels)
    train_images = np.array(train_images) ## Convert to NumPy array
    arts = train_images.astype('float32')  ## Convert to floats
    arts = arts / 255 ## Rescale
    return arts

## Building the Dicriminator
def build_discriminator(input_shape=(128,128,3)):
    model = Sequential()
    model.add(Conv2D(128, (4,4), strides=(2,2), padding='same', input_shape=input_shape)) ## Output shape = 64x64x128
    model.add(LeakyReLU(alpha=0.2)) ## Activation: Leaky ReLU
    model.add(Conv2D(128, (4,4), strides=(2,2), padding='same')) ## Output shape = 32x32x128
    model.add(LeakyReLU(alpha=0.2)) ## Activation: Leaky ReLU
    model.add(Flatten())  ## Output: 131072
    model.add(Dropout(0.4))
    model.add(Dense(1, activation='sigmoid')) # Shape of 1

    ## Compile the model
    optimizer = Adam(learning_rate=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

## Testing the discriminator
test_discriminator = build_discriminator()
print(test_discriminator.summary())

## Building the Generator
def build_generator(noise_dim=100):
    model = Sequential()
    model.add(Dense(32*32*128, input_shape=(noise_dim,)))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Reshape((32,32,128))) ## Output: 32x32x128 dataset
    model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same')) ## Upsampling to output: 16x16x128
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same')) ## Upsampling to output: 32x32x128
    model.add(LeakyReLU(alpha=0.2))
    #model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same')) ## Upsampling to output: 64x64x128
    #model.add(LeakyReLU(alpha=0.2))
    #model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same')) ## Upsampling to output: 128x128x256
    #model.add(LeakyReLU(alpha=0.2))  
    model.add(Conv2D(3, (8,8), activation='sigmoid', padding='same')) ## Output: 128x128x3 
    return model 

## Testing the generator
test_generator = build_generator()
print(test_generator.summary())
noise = tf.random.normal([1, 100], 100)
generated_image = test_generator(noise, training=False)
plt.imshow(generated_image[0, :, :, :])

## Since we have used Sigmoid activation, the outputs should lay between 0 and 1
## If it is below 0.5, the image is fake(generated) and greater than 0.5 it is a real Image.
decision = test_discriminator(generated_image)
print (decision)

## Combine the GAN
def build_gan(discriminator, generator):
    discriminator.trainable = False ## Since we train the dicsriinator separately
    model = Sequential()
    model.add(generator)
    model.add(discriminator)
    ## Compile
    optimizer = Adam(learning_rate=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=optimizer)
    return model

def real_samples(arts, samples):
    index = randint(0, arts.shape[0], samples)
    X = arts[index]
    y = ones((samples, 1)) ## Since real arts
    return X, y

def random_noise(noise_dim, samples):
    generator_input = randn(noise_dim * samples)
    generator_input = generator_input.reshape(samples, noise_dim)
    return generator_input

def fake_samples(generator, noise_dim, samples):
    generator_input = random_noise(noise_dim, samples)
    X = generator.predict(generator_input)
    y = zeros((samples, 1)) ## Since the fake arts
    return X, y

## Define the Training Loop

## Define the parameters
generator = build_generator()
discriminator = build_discriminator()
gan  = build_gan(discriminator, generator)
arts = load_arts(resized_images, labels)
noise_dim = 100
epochs = 100
batch_size = 64

def train_the_GAN(generator_model, discriminator_model, gan_model, arts, noise_dim=100, epochs=100, batch_size=64):

    batches_per_epoch = int(arts.shape[0] / batch_size)
    half_batch = int(batch_size / 2)

    ## We train the discriminator with half batch size real arts and fake arts.
    epoch_loss_discriminator = []
    epoch_loss_generator = []
    for epoch in range(epochs):

        discriminator_loss_acc = 0.
        generator_loss_acc = 0.

        for batch_per_epoch in range(batches_per_epoch):
            print(f"epoch: {epoch+1}")

            X_real, y_real = real_samples(arts, half_batch)
            discriminator_loss_1, _ = discriminator_model.train_on_batch(X_real, y_real)
            X_fake, y_fake = fake_samples(generator_model, noise_dim, half_batch)
            discriminator_loss_2, _ = discriminator_model.train_on_batch(X_fake, y_fake)
            discriminator_loss = 0.5 * np.add(discriminator_loss_1, discriminator_loss_2)

            X_gan = random_noise(noise_dim, batch_size)
            y_gan = ones((batch_size, 1))

            generator_loss = gan_model.train_on_batch(X_gan, y_gan)

            discriminator_loss_acc += discriminator_loss
            generator_loss_acc += generator_loss

        epoch_loss_discriminator.append(discriminator_loss_acc / batches_per_epoch)
        epoch_loss_generator.append(generator_loss_acc / batches_per_epoch)

    generator_model.save('Rootcode_100_new3.h5') 
    x = []
    for i in range(epochs):
      x.append(i+1)
      i = i + 1
    y1points = np.array(epoch_loss_discriminator)
    y2points = np.array(epoch_loss_generator)
    xpoints = np.array(x)
    plt.plot(xpoints, y1points,xpoints, y2points)
    plt.show()

## Train and Save the Generator
train_the_GAN(generator, discriminator, gan, arts, noise_dim, epochs, batch_size)

## Import saved Generator and Generate Arts!!

def show_plot(examples, n):
	for i in range(n * n):
		plt.subplot(n, n, 1 + i)
		plt.axis('off')
		plt.imshow(examples[i, :, :, :])
	plt.show()

# load model
model = load_model('Rootcode_100_new3.h5') #Model trained for 100 epochs
# generate images
random_seed = random_noise(100, 4)  ## Noise 
# generate images
new_art = model.predict(random_seed)
new_art = (new_art * 255).astype(np.uint8)

# plot the result
show_plot(new_art, 2)