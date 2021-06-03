from __future__ import print_function
import os
import argparse
import pickle
import numpy as np
import PIL.Image as pil
import PIL.ImageOps	
from keras.applications import DenseNet121
from keras.utils import multi_gpu_model
from keras.models import Model
from keras.applications.densenet import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from keras.layers import GlobalAveragePooling2D, Dense
from keras import optimizers
from auc_callback import AucRoc

parser = argparse.ArgumentParser()

parser.add_argument(
    '--data_dir', type=str, default='/workspace/data/chestxray/images/',
    help='The directory where the input data is stored.')

parser.add_argument(
    '--model_dir', type=str, default=os.getcwd(),
    help='The directory where the model will be stored.')

parser.add_argument(
    '--batch_size', type=int, default='16',
    help='Batch size for SGD')

parser.add_argument(
    '--image_size', type=int, default='256',
    help='Image size')

parser.add_argument(
    '--lr', type=float, default='1e-3',
    help='Learning rate for SGD')

parser.add_argument(
    '--epochs', type=int, default=15,
    help='Number of epochs to train')

with open('/workspace/cheXnet_tf_hvd/training_labels_new.pkl', 'rb') as f:
  training_labels = pickle.load(f)
training_files = np.asarray(list(training_labels.keys()))

with open('/workspace/cheXnet_tf_hvd/validation_labels_new.pkl', 'rb') as f:
  validation_labels = pickle.load(f)
validation_files = np.asarray(list(validation_labels.keys()))
labels = dict(list(training_labels.items()) + list(validation_labels.items()))

def load_batch(batch_of_files,is_training=False):
  batch_images = []
  batch_labels = []
  for filename in batch_of_files:
    img = pil.open(os.path.join(FLAGS.data_dir, filename))
    img = img.convert('RGB')
    img = img.resize((FLAGS.image_size, FLAGS.image_size),pil.NEAREST)
    if is_training and np.random.randint(2):
      img = PIL.ImageOps.mirror(img)
    batch_images.append(np.asarray(img))
    batch_labels.append(labels[filename])
  return preprocess_input(np.float32(np.asarray(batch_images))), np.asarray(batch_labels)

def train_generator(num_of_steps):
  while True:
    np.random.shuffle(training_files)
    batch_size = FLAGS.batch_size
    for i in range(num_of_steps):
      batch_of_files = training_files[i*batch_size: i*batch_size + batch_size]
      batch_images, batch_labels = load_batch(batch_of_files, True)
      yield batch_images, batch_labels

def val_generator(num_of_steps):
  while True:
    batch_size = FLAGS.batch_size
    for i in range(num_of_steps):
      batch_of_files = validation_files[i*batch_size: i*batch_size + batch_size]
      batch_images, batch_labels = load_batch(batch_of_files, True)
      yield batch_images, batch_labels

def main():
  print("Running with the following config:")
  for item in FLAGS.__dict__.items():
    print('%s = %s' %(item[0], str(item[1])))

  base_model = DenseNet121(include_top=False,
                   weights='imagenet',
                   input_shape=(FLAGS.image_size, FLAGS.image_size, 3))
  x = base_model.output
  x = GlobalAveragePooling2D()(x)
  predictions = Dense(14, activation='sigmoid', bias_initializer='ones')(x)
  model = Model(inputs=base_model.input, outputs=predictions)
  adam = optimizers.Adam(lr=FLAGS.lr)
  
  model.compile(loss='binary_crossentropy',
                         optimizer=adam,
                         metrics=['accuracy'])
  # Path to weights file
  weights_file=FLAGS.model_dir + '/lr_{:.3f}_bz_{:d}'.format(FLAGS.lr, FLAGS.batch_size) + '_loss_{val_loss:.3f}_epoch_{epoch:02d}.h5'
  
  # Callbacks
  steps_per_epoch = 77871 // FLAGS.batch_size
  val_steps = 8653 // FLAGS.batch_size
  lr_reducer =  ReduceLROnPlateau(monitor='val_loss', factor=0.1, epsilon=0.01,
                                      cooldown=0, patience=1, min_lr=1e-15, verbose=1)
  auc = AucRoc(val_generator(val_steps), val_steps)
  model_checkpoint= ModelCheckpoint(weights_file, monitor="val_loss",save_best_only=False,
                                    save_weights_only=True, verbose=1)
  # specify training params and start training
  model.fit_generator(
  		train_generator(steps_per_epoch),
  		steps_per_epoch=steps_per_epoch,
        	epochs=FLAGS.epochs,
       		validation_data=val_generator(val_steps),
  		validation_steps=val_steps,
  	        callbacks=[lr_reducer, auc, model_checkpoint],
        	verbose=2)

if __name__ == '__main__':
 FLAGS, _ = parser.parse_known_args()
 main()
