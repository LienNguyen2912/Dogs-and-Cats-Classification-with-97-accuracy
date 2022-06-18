from os import listdir, makedirs
import os
import sys
from random import random, seed
from shutil import copyfile
from matplotlib import pyplot
from matplotlib.image import imread
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Model

def plot_some_images():
    folder = 'train/'
    for i in range(18):
        pyplot.subplot(3, 6, i + 1 )
        filename = folder + 'dog.' + str(i) + '.jpg' if (i < 9) else folder + 'cat.' + str(i) + '.jpg'
        image = imread(filename)
        pyplot.imshow(image)
        pyplot.title('image.shape: ' + str(image.shape))
    #pyplot.suptitle('First 9 photos of dogs and cats ')
    fig = pyplot.gcf()
    fig.set_size_inches(18, 18)
    # set the spacing between subplots
    pyplot.subplots_adjust(wspace=0, hspace=0)
    pyplot.tight_layout()
    pyplot.show()
    
def divide_images_into_directories():
    #  dataset_dogs_cats
    # |---test
    # |   |---cats
    # |   |---dogs
    # |---train
    #    |---cats
    #    |---dogs
    dataset_home = 'dataset_dogs_cats/'
    if os.path.isdir(dataset_home):
        return
    subdirs = ['train/', 'test/']
    for subdir in subdirs:
        labeldirs = ['dogs/', 'cats/']
        for labeldir in labeldirs:
            newdir = dataset_home + subdir + labeldir
            makedirs(newdir, exist_ok=True)
    # seed random number generator
    seed(1)
    validation_ratio = 0.25
    src_directory = 'train/'
    for file in listdir(src_directory):
        src = src_directory + '/' + file
        dst_dir = 'train/' if random() < validation_ratio else 'test/'
        if file.startswith('cat'):
            dst = dataset_home + dst_dir + 'cats/' + file
            copyfile(src, dst)
        elif file.startswith('dog'):
            dst = dataset_home + dst_dir + 'dogs/' + file
            copyfile(src, dst)

def define_model():
    # load model
	model = VGG16(include_top=False, input_shape=(224, 224, 3))
	# mark loaded layers as not trainable
	for layer in model.layers:
		layer.trainable = False
	# add new classifier layers
	flat1 = Flatten()(model.layers[-1].output)
	class1 = Dense(128, activation='relu', kernel_initializer='he_uniform')(flat1)
	output = Dense(1, activation='sigmoid')(class1)
	# define new model
	model = Model(inputs=model.inputs, outputs=output)
	# compile model
	opt = SGD(learning_rate=0.001, momentum=0.9)
	model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
	return model

# plot diagnostic learning curves
def summarize_diagnostics(history):
	# plot loss
	pyplot.subplot(211)
	pyplot.title('Cross Entropy Loss')
	pyplot.plot(history.history['loss'], color='blue', label='train')
	pyplot.plot(history.history['val_loss'], color='orange', label='test')
	# plot accuracy
	pyplot.subplot(212)
	pyplot.title('Classification Accuracy')
	pyplot.plot(history.history['accuracy'], color='blue', label='train')
	pyplot.plot(history.history['val_accuracy'], color='orange', label='test')
	# save plot to file
	filename = sys.argv[0].split('/')[-1]
	pyplot.savefig(filename + '_plot.png')
	pyplot.subplots_adjust(wspace=0, hspace=0)
	pyplot.tight_layout()
	pyplot.show()
	#pyplot.close()

def execute():
    # define model
	model = define_model()
	# create data generators
	train_datagen = ImageDataGenerator(rotation_range=90, 
                                     brightness_range=[0.1, 0.7],
                                     width_shift_range=0.5, 
                                     height_shift_range=0.5,
                                     horizontal_flip=True, 
                                     vertical_flip=True,
                                     validation_split=0.15,
                                     preprocessing_function=preprocess_input) # VGG16 preprocessing

	test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input) # VGG16 preprocessing
	# prepare iterators
	train_it = train_datagen.flow_from_directory('dataset_dogs_cats/train/',
		class_mode='binary', batch_size=32, target_size=(224, 224))
	test_it = test_datagen.flow_from_directory('dataset_dogs_cats/test/',
		class_mode='binary', batch_size=32, target_size=(224, 224))
	
	history = model.fit(train_it, steps_per_epoch=train_it.samples//train_it.batch_size,
		validation_data=test_it, validation_steps=test_it.samples//test_it.batch_size, epochs=35, verbose=1)
 
	# evaluate model
	_, acc = model.evaluate(test_it, steps=len(test_it), verbose=0)
	print('> %.3f' % (acc * 100.0))
	# learning curves
	summarize_diagnostics(history)
 
# main program
# plot_some_images()
#divide_images_into_directories()
execute()