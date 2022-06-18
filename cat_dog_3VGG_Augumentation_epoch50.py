from os import listdir, makedirs
import os
import sys
from random import random, seed
from shutil import copyfile
from matplotlib import pyplot
from matplotlib.image import imread
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def plot_some_images():
    folder = 'train/'
    for i in range(18):
        pyplot.subplot(3, 6, i + 1 )
        filename = folder + 'dog.' + str(i) + '.jpg' if (i < 9) else folder + 'cat.' + str(i) + '.jpg'
        image = imread(filename)
        pyplot.imshow(image)
        pyplot.title('image.shape: ' + str(image.shape))
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
    model = Sequential()
    model.add(Conv2D(32, (3,3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape = (200, 200, 3) ))
    model.add(MaxPooling2D((2,2)))

    model.add(Conv2D(64, (3,3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2,2)))

    model.add(Conv2D(128, (3,3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2,2)))

    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))

    model.add(Dense(1, activation='sigmoid'))
    # compile model
    opt = SGD(learning_rate = 0.001, momentum=0.9)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    return model

def plot_learning_curves(history):
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
	pyplot.close()

def execute():
    # define model
	model = define_model()
	# create data generators
	train_datagen = ImageDataGenerator(rescale = 1./255,
							   shear_range = 0.2,
							   zoom_range = 0.2,
							   horizontal_flip = True) 
	test_datagen = ImageDataGenerator(rescale=1.0/255.0)
	# prepare iterators
	train_it = train_datagen.flow_from_directory('dataset_dogs_cats/train/',
		class_mode='binary', batch_size=64, target_size=(200, 200))
	test_it = test_datagen.flow_from_directory('dataset_dogs_cats/test/',
		class_mode='binary', batch_size=64, target_size=(200, 200))
	# fit model
	history = model.fit(train_it, steps_per_epoch=len(train_it),
		validation_data=test_it, validation_steps=len(test_it), epochs=50, verbose=1)
	# evaluate model
	_, acc = model.evaluate(test_it, steps=len(test_it), verbose=0)
	print('> %.3f' % (acc * 100.0))
	# learning curves
	plot_learning_curves(history)
 
execute()