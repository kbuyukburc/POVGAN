import imp
from models.povgan_tf import POVGANVanilla
import tensorflow as tf

import os
import pathlib
import time
import datetime

from matplotlib import pyplot as plt

import cv2
from glob import glob
import numpy as np
import argparse

args = argparse.ArgumentParser("Point of View GAN Trainer")
args.add_argument("--batch", type=int, help="Batch Size", default=1)
args.add_argument("--buffer", type=int, help="Pre-Load buffer size", default=500)
args.add_argument("--size", type=int, help="Input size", default=256)

CAMERA_SENSORS = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_FRONT_LEFT']

lidar_files = []
image_files = []
# Fetch sensor files
for cam_sensor in CAMERA_SENSORS:
    lidar_files += sorted(glob(f'./dataset_256_256/LIDAR/{cam_sensor}/*.npy'))
    image_files += sorted(glob(f'./dataset_256_256/{cam_sensor}/*.jpg')) 

def load_image_train(sample_id):
  # Load and Normalize images and points
  img = cv2.imread(image_files[sample_id])
  points = np.load(lidar_files[sample_id])
  points[0] = points[0] / 255 # Lidar Depth
  points[1] = points[1] / 255 # Lidar Reflection
  points = points - 0.5
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  img = (img / 127.5) - 1
  points = np.moveaxis(points, 0, -1)
  img = tf.cast(img, tf.float32)
  points = tf.cast(points, tf.float32)
  return points, img

def generate_images(model, test_input, tar):
  prediction = model(test_input, training=True)
  plt.figure(figsize=(15, 15))

  display_list = [test_input[0], test_input[0], tar[0], prediction[0]]
  title = ['Depth', 'Reflection', 'Ground Truth', 'Predicted Image']

  for i in range(4):
    plt.subplot(1, 4, i+1)
    plt.title(title[i])
    # Getting the pixel values in the [0, 1] range to plot.
    if(i==0):      
      plt.imshow(display_list[i][..., 0] * 0.5 + 0.5)
    elif(i==1):      
      plt.imshow(display_list[i][..., 1] * 0.5 + 0.5)
    else:
      plt.imshow(display_list[i] * 0.5 + 0.5)
    plt.axis('off')
  plt.show()

length = len(image_files)
train_length = int(length * 0.8)
train_dataset = tf.data.Dataset.range(train_length)
train_dataset = train_dataset.map(lambda x : \
                tf.py_function(func=load_image_train, inp=[x], Tout=(tf.float32, tf.float32)), \
                num_parallel_calls=tf.data.AUTOTUNE)
train_dataset = train_dataset.shuffle(args.buffer)
train_dataset = train_dataset.batch(args.buffer)

test_dataset = tf.data.Dataset.range(train_length, length)
test_dataset = test_dataset.map(lambda x : \
                tf.py_function(func=load_image_train, inp=[x], Tout=(tf.float32, tf.float32)), \
                num_parallel_calls=tf.data.AUTOTUNE)
test_dataset = test_dataset.shuffle(args.buffer)
test_dataset = test_dataset.batch(args.buffer)

model = POVGANVanilla()
generator_optimizer = model.generator_optimizer
discriminator_optimizer = model.discriminator_optimizer
generator = model.generator()
discriminator = model.discriminator()

LAMBDA = 100
loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
def generator_loss(disc_generated_output, gen_output, target):
  gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)

  # Mean absolute error
  l1_loss = tf.reduce_mean(tf.abs(target - gen_output))

  total_gen_loss = gan_loss + (LAMBDA * l1_loss)

  return total_gen_loss, gan_loss, l1_loss

def discriminator_loss(disc_real_output, disc_generated_output):
  real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)

  generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)

  total_disc_loss = real_loss + generated_loss

  return total_disc_loss

@tf.function
def train_step(input_image, target, step):
  with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
    gen_output = generator(input_image, training=True)

    disc_real_output = discriminator([input_image, target], training=True)
    disc_generated_output = discriminator([input_image, gen_output], training=True)

    gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(disc_generated_output, gen_output, target)
    disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

  generator_gradients = gen_tape.gradient(gen_total_loss,
                                          generator.trainable_variables)
  discriminator_gradients = disc_tape.gradient(disc_loss,
                                               discriminator.trainable_variables)

  generator_optimizer.apply_gradients(zip(generator_gradients,
                                          generator.trainable_variables))
  discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                              discriminator.trainable_variables))

  with summary_writer.as_default():
    tf.summary.scalar('gen_total_loss', gen_total_loss, step=step//1000)
    tf.summary.scalar('gen_gan_loss', gen_gan_loss, step=step//1000)
    tf.summary.scalar('gen_l1_loss', gen_l1_loss, step=step//1000)
    tf.summary.scalar('disc_loss', disc_loss, step=step//1000)


log_dir="logs/"
summary_writer = tf.summary.create_file_writer(
  log_dir + "fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

checkpoint_dir = './checkpoint/'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

def fit(train_ds, test_ds, steps):
  example_input, example_target = next(iter(test_ds.take(1)))
  start = time.time()

  for step, (input_image, target) in train_ds.repeat().take(steps).enumerate():
    if (step) % 1000 == 0:
      print(f'Time taken for 1000 steps: {time.time()-start:.2f} sec\n')
      start = time.time()
      example_input, example_target = next(iter(test_ds.take(1)))
      generate_images(generator, example_input, example_target)
      print(f"Step: {step//1000}k")

    train_step(input_image, target, step)

    # Training step
    if (step+1) % 10 == 0:
      print('.', end='', flush=True)

    # All dataset 1 epoch
    if (step + 1) % 39000 == 0:
      checkpoint.save(file_prefix=checkpoint_prefix)
      example_input, example_target = next(iter(test_ds.take(1)))
      
checkpoint.restore(tf.train.latest_checkpoint('./checkpoint/'))
fit(train_dataset, test_dataset, steps=400000)