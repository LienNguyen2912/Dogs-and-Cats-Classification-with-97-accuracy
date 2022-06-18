# Image Classification between Dogs and Cats (with 97% accuracy)
![dog-g2b26f8e3c_1920](https://user-images.githubusercontent.com/73010204/166631913-1c06b24f-7096-4b9d-abf6-fcc0b2a95020.jpg)</br>
We are given a set of dog and cat images. **The task is to create a model to classify whether images contain either a dog or a cat.**</br>
Data is provided by Kaggle. In the train.zip, there are 25,000 labeled images, 50% images of dogs and the rest are cats.
In the test1.zip, there are 12,500 unlabeled images for testing.</br>
![train_folder_img](https://user-images.githubusercontent.com/73010204/166633679-0eb401cb-d681-40e1-ac9a-2196db5575fb.png)</br>
We will:
- Load and prepare photos of dogs and cats for modeling.
- Develop a convolutional neural network
- Improve model performance by appyling
    - Batch normalization
    - Dropout
    - Data Augmentation
- Apply transfer learning with
    - VGG16
    - ResNet
- Finalize the model and make Predictions


## Load and prepare photos of dogs and cats for modeling.
### Plot some dog and cat photos
We will show 9 dog images and 9 cat images.
```sh
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

# execute
plot_some_images()
```
![plot_images](https://user-images.githubusercontent.com/73010204/166637451-53c86018-f681-4e45-a253-15f4c6af238f.png)</br>
We can see that the photos are color and they are all different sizes. So we need to reshape all images before start training. In this example, we choose a fixed size of 200×200x3 pixels.
### Divide photos of dogs and cats into separated folders
We are using _flow_from_directory()_ method of the _ImageDataGenerator_ class to load images from the disk. This API requires data to be divided into separate directories, and under each directory to have a sub-directory for each class. We will randomly select 25% of the images to be test dataset and the rest to be training dataset.</br>
![divide_train_validation_folder](https://user-images.githubusercontent.com/73010204/166656368-0877584e-b8e5-451f-a081-859c23202f27.PNG)
```sh
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

# execute
divide_images_into_directories()
```
And later on
```sh
datagen = ImageDataGenerator(rescale=1.0/255.0)
# prepare iterators
train_it = datagen.flow_from_directory('dataset_dogs_cats/train/', class_mode='binary', batch_size=64, target_size=(200, 200))
test_it = datagen.flow_from_directory('dataset_dogs_cats/test/',class_mode='binary', batch_size=64, target_size=(200, 200))
```
## Develop a convolutional neural network
In general, there are 4 steps to build a CNN: 
- Convolution
- Max Pooling
- Flattening
- Full connection

CNN does the processing of Images with the help of matrixes of weights known as filters. They detect low-level features like vertical and horizontal edges etc. Through each layer, the filters recognize high-level features.</br>
![CNN2](https://user-images.githubusercontent.com/73010204/166670184-3223aeb4-7c1d-4652-977b-9d9b8bc63723.png)</br>
![CNN_overview](https://user-images.githubusercontent.com/73010204/166671414-a7b42b1e-3db5-4174-922c-9f3b10071030.png)
)</br>
_"The activation function is added to help CNN learn complex patterns in the data. The main need for activation function is to add non-linearity into the neural network.</br>
The pooling operation provides spatial variance making the system capable of recognizing an object with some varied appearance. It involves adding a 2Dfilter over each channel of the feature map and thus summarise features lying in that region covered by the filter.</br>
So, pooling basically helps reduce the number of parameters and computations present in the network. It progressively reduces the spatial size of the network and thus controls overfitting._"

We can create a function named define_model() function for defining a convolutional neural network model 3 vgg-style blocks and plot the loss and accuracy of train and validation dataset.
```sh
from os import listdir, makedirs
import os
import sys
from random import random, seed
from shutil import copyfile
from matplotlib import pyplot
from matplotlib.image import imread
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator

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

def execute():
	# define model
	model = define_model()
	# create data generator
	datagen = ImageDataGenerator(rescale=1.0/255.0)
	# prepare iterators
	train_it = datagen.flow_from_directory('dataset_dogs_cats/train/',
		class_mode='binary', batch_size=64, target_size=(200, 200))
	test_it = datagen.flow_from_directory('dataset_dogs_cats/test/',
		class_mode='binary', batch_size=64, target_size=(200, 200))
	# fit model
	history = model.fit(train_it, steps_per_epoch=train_it.samples//train_it.batch_size,
		validation_data=test_it, validation_steps=test_it.samples//test_it.batch_size, epochs=20, verbose=1)
	# evaluate model
	_, acc = model.evaluate(test_it, steps=len(test_it), verbose=0)
	print('> %.3f' % (acc * 100.0))
	# learning curves
	plot_learning_curves(history)
 
# main program
execute()
```
Here is the result.
```sh
Found 6303 images belonging to 2 classes.
Found 18697 images belonging to 2 classes.
> 74.172
```
In this case, we can see that the model achieved an accuracy of about 74% on the test dataset.
Reviewing this plot, we can see that the model has overfit the training dataset at about 12 epochs.</br>
![3VGG](https://user-images.githubusercontent.com/73010204/166938149-530d3913-7af5-4573-8dde-1e8131cdfde7.png)

## Reduce overfitting 
### Dropout
Firstly, let's try to add _dropout()_ and keep everything as before
```sh
def define_model():
    model = Sequential()
    model.add(Conv2D(32, (3,3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape = (200, 200, 3) ))
    model.add(MaxPooling2D((2,2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(64, (3,3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2,2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(128, (3,3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2,2)))
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dropout(0.4))

    model.add(Dense(1, activation='sigmoid'))
    # compile model
    opt = SGD(learning_rate = 0.001, momentum=0.9)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    return model

def execute():
	# define model
	model = define_model()
	# create data generator
	datagen = ImageDataGenerator(rescale=1.0/255.0)
	# prepare iterators
	train_it = datagen.flow_from_directory('dataset_dogs_cats/train/',
		class_mode='binary', batch_size=64, target_size=(200, 200))
	test_it = datagen.flow_from_directory('dataset_dogs_cats/test/',
		class_mode='binary', batch_size=64, target_size=(200, 200))
	# fit model
	history = model.fit(train_it, steps_per_epoch=train_it.samples//train_it.batch_size,
		validation_data=test_it, validation_steps=test_it.samples//test_it.batch_size, epochs=20, verbose=1)
	# evaluate model
	_, acc = model.evaluate(test_it, steps=len(test_it), verbose=0)
	print('> %.3f' % (acc * 100.0))
	# learning curves
	plot_learning_curves(history)
```
The output is below
```sh
Found 6303 images belonging to 2 classes.
Found 18697 images belonging to 2 classes.
Epoch...
Epoch 20/20
99/99 [==============================] - 751s 8s/step - loss: 0.6218 - accuracy: 0.6454 - val_loss: 0.6258 - val_accuracy: 0.6456
> 64.561
```
We can set there is not much different between the accuracy of training and test dataset. So we can say the model is not overfitting anymore. But wait, we get only 64% accuracy.</br>
![3VGG_Dropout_20epochs_cat_dog_cnn py_plot](https://user-images.githubusercontent.com/73010204/170200162-a3f5e673-f74c-46f9-9341-e2fc5de87d0c.png)</br>
So let's increase epoch number to 50 to resolve high bias issue.
```sh
history = model.fit(train_it, steps_per_epoch=len(train_it)//64,
		validation_data=test_it, validation_steps=len(test_it)//64, epochs=50, verbose=1)
```
Here's the result
```sh
Found 6303 images belonging to 2 classes.
Found 18697 images belonging to 2 classes.
Epoch 1/50
98/98 [==============================] - 425s 4s/step - loss: 0.7268 - accuracy: 0.5169 - val_loss: 0.6893 - val_accuracy: 0.5621
...
Epoch 50/50
98/98 [==============================] - 455s 5s/step - loss: 0.4824 - accuracy: 0.7610 - val_loss: 0.5475 - val_accuracy: 0.7204
> 72.038
```
![cat_dog_3VGG_dropout_epoch50 py_plot](https://user-images.githubusercontent.com/73010204/170870545-481f896b-ebe7-4f47-9816-a3dd82488b44.png)

### Batch normalization
Let's try batch normalization for resolve overfitting 
```sh
def define_model():
    model = Sequential()
    model.add(Conv2D(32, (3,3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape = (200, 200, 3) ))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2,2)))

    model.add(Conv2D(64, (3,3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2,2)))

    model.add(Conv2D(128, (3,3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2,2)))

    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
    model.add(BatchNormalization())

    model.add(Dense(1, activation='sigmoid'))
    # compile model
    opt = SGD(learning_rate = 0.001, momentum=0.9)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    return model

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
 
# main program
execute()
```
And the output is quite better
```sh
Found 6303 images belonging to 2 classes.
Found 18697 images belonging to 2 classes.
Epoch 1/50
99/99 [==============================] - 554s 6s/step - loss: 0.6451 - accuracy: 0.6784 - val_loss: 0.7962 - val_accuracy: 0.5304
...
Epoch 50/50
99/99 [==============================] - 642s 7s/step - loss: 9.4815e-04 - accuracy: 1.0000 - val_loss: 0.7559 - val_accuracy: 0.7691
> 76.905
```
![3VGG_BN_50epochs py_plot](https://user-images.githubusercontent.com/73010204/170487565-3c5bd224-38dc-4433-b9af-c4450c0016a2.png)
### Augmentation
Image augmentation is used to increase the number of training images by applying some transformations to the images. For example, we can randomly rotate or crop or flip the images
Here is the full source code
```sh
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
 
# main program
execute()
```
And we get thís
```sh
Found 6303 images belonging to 2 classes.
Found 18697 images belonging to 2 classes.
...
Epoch 49/50
99/99 [==============================] - 451s 5s/step - loss: 0.3119 - accuracy: 0.8683 - val_loss: 0.4646 - val_accuracy: 0.7961
Epoch 50/50
99/99 [==============================] - 455s 5s/step - loss: 0.3124 - accuracy: 0.8659 - val_loss: 0.4837 - val_accuracy: 0.7902
> 79.018
```
![3VGG_augument_50epoch py_plot](https://user-images.githubusercontent.com/73010204/172162960-50377b8b-04b2-4be3-b9bd-613c12bcef10.png)</br>
### Review what we got
- **VGG 3 blocks: 74.172%**
- **VGG 3 blocks + Dropout: 72.038%**
- **VGG 3 blocks + Batch normalization: 76.905%**
- **VGG 3 blocks + Augmentation: 79.018%**
 
## Model improvement by transfer learning
_Transfer Learning is an approach where we use one model trained on a machine learning task and reuse it as a starting point for a different job. Multiple deep learning domains use this approach, including Image Classification, Natural Language Processing, and even Gaming! The ability to adapt a trained model to another task is incredibly valuable._</br>
![TransferLearning_Def](https://user-images.githubusercontent.com/73010204/173172896-4336a666-0393-4eb9-80a6-14bc5f6c2201.png)</br>

### Why Transfer Learning for CNN
- Because it is very difficult to have enough dataset, generally there are very few people train a Convolution network from scratch (random initialization). Mostly we use pre-trained network weights as initialization for solving our problems in hand.
- Deep Networks are expensive to train. Fortunately we have Pre-trained models that are usually shared in the form of the millions of parameters/weights the model achieved while being trained to a stable state.
- From the https://keras.io/api/applications/ we can pick and select any of the state-of-the-art models and use it for our problem.
- ![KerasProvidedModels](https://user-images.githubusercontent.com/73010204/173173168-1ff4c2bf-a8c6-4ee2-a581-d6b1b919593d.png)

### Transfer learning with VGG16
VGG16 is a convolutional neural network trained on a subset of the ImageNet dataset, a collection of over 14 million images belonging to 22,000 categories. K. Simonyan and A. Zisserman proposed this model in the 2015 paper, Very Deep Convolutional Networks for Large-Scale Image Recognition.In the 2014 ImageNet Classification Challenge, VGG16 achieved a 92.7% classification accuracy.</br>
First, consider the architecture of the VGG16 convolutional network, shown below.
![VGG16_model](https://user-images.githubusercontent.com/73010204/173173749-1056ff66-247d-4fce-bb87-d488c07f94b5.png)</br>

We can defines how many layers to freeze during training.
![FineTuning](https://user-images.githubusercontent.com/73010204/173174894-ac70e67b-1042-4dea-aa5d-e05c5eb57b0a.png)
```sh
input_shape = (224, 224, 3)
conv_base = VGG16(include_top=False, weights='imagenet',  input_shape=input_shape)
# Fine-Tuning: leaving the last fine_tune layers(2 or...) unfrozen
if fine_tune > 0:
    for layer in conv_base.layers[:-fine_tune]:
        layer.trainable = False
# Without fine-tuning
else:
    for layer in conv_base.layers:
        layer.trainable = False
```
Below is full source code. 
- We are keeping the batch size as 32. If you are working on a system with lower ram configuration, you can reduce the batch size further.
- We must set _include_top=False_. We can't use the entirety of the pre-trained VGG16 model's architecture. Because the Fully-Connected layer generates 1,000 different output labels, whereas we have only two classes for prediction, cat or dog. So we'll import a pre-trained model like VGG16, but "cut off" the Fully-Connected layer - also called the "top" model
- Finally, we choose training without Fine-Tuning by implementing _layer.trainable= False_ . This ensures that the model does not learn all the weights again, saving us a lot of time and space complexity. 
- To match with VGG16 model's architecture, the input image size is (224, 224, 3) and we have to set _preprocessing_function=preprocess_input_ which is imported by _from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input_. This _preprocessing_function_ parameter is meant to adequate your image to the format the model requires. 
```sh
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
    
	# fit model
	history = model.fit(train_it, steps_per_epoch=train_it.samples//train_it.batch_size,
		validation_data=test_it, validation_steps=test_it.samples//test_it.batch_size, epochs=35, verbose=1)
 
	# evaluate model
	_, acc = model.evaluate(test_it, steps=len(test_it), verbose=0)
	print('> %.3f' % (acc * 100.0))
	# learning curves
	plot_learning_curves(history)
 
# main program
execute()
```
We get much better accuracy, about 97%. Great! Well, actuall after around 15 epochs the result is almost not changed. We should implement _early stopping_ to reduce  waiting time for the training process completed.
```sh
Epoch 1/35
196/196 [==============================] - 5276s 27s/step - loss: 0.6383 - accuracy: 0.7756 - val_loss: 0.1282 - val_accuracy: 0.9587
...
Epoch 35/35
196/196 [==============================] - 5403s 28s/step - loss: 0.2556 - accuracy: 0.8924 - val_loss: 0.0818 - val_accuracy: 0.9696
> 96.962
```
![3VGG_augument_TransferLearningVGG16_35epoch py_plot](https://user-images.githubusercontent.com/73010204/173174902-8a617f41-1596-4ac9-9eeb-de3c8c478ada.png)

### Transfer learning with Resnet50(Residual Networks)
As the neural networks architectures have become deeper, from just a few layers (e.g., VGG16) to over a hundred layers, a very deep network can represent very complex functions so that it can learn features at many different levels of abstraction, for example,  edges (at the lower layers) to very complex features (at the deeper layers). However, using a deeper network faces to vanishing gradients problem. Very deep networks often have a gradient signal that goes to zero quickly, thus making gradient descent slow.</br>
Resnet addressed this problem by using  a “shortcut” or a “skip connection”, below is Resnet50's architecture.</br>
![resnet50](https://user-images.githubusercontent.com/73010204/173260935-63b6a85a-12d1-43a3-a691-ba73d7856b06.png)</br>
We keep almost the implement using resnet50 almost similar to vgg16. This time, we use _early stopping_ and _model checkpoint_. </br>
I run this on the _google colab_. The completed example is listed below.
```sh
from os import listdir, makedirs
import os
import sys
from random import random, seed
from shutil import copyfile
from matplotlib import pyplot
from matplotlib.image import imread
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint

def define_model():
    # load model
	model = ResNet50(include_top=False, input_shape=(224, 224, 3))
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
                                     preprocessing_function=preprocess_input) # ResNet50 preprocessing

	test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input) # ResNet50 preprocessing
	# prepare iterators 
	train_it = train_datagen.flow_from_directory('/content/drive/MyDrive/Cat_Dog_CNN/dataset_dogs_cats/train/',
		class_mode='binary', batch_size=32, target_size=(224, 224))
	test_it = test_datagen.flow_from_directory('/content/drive/MyDrive/Cat_Dog_CNN/dataset_dogs_cats/test/',
		class_mode='binary', batch_size=32, target_size=(224, 224))
	filename = sys.argv[0].split('/')[-1] + 'best_model_resnet50.h5'
	es = EarlyStopping(monitor='accuracy', mode='max', verbose=1,  min_delta=0, patience=20, restore_best_weights=True)
	mc = ModelCheckpoint(filename, monitor='accuracy', mode='max', verbose=1, save_best_only=True)
	# fit model
	history = model.fit(train_it, steps_per_epoch=len(train_it)//train_it.batch_size,
		validation_data=test_it, validation_steps=len(test_it)//test_it.batch_size, epochs=35, verbose=1, callbacks=[es,mc])

	# evaluate model
	_, acc = model.evaluate(test_it, steps=len(test_it), verbose=0)
	print('> %.3f' % (acc * 100.0))
	plot_learning_curves(history)
 
# main program
execute()
```
```sh
from google.colab import drive
drive.mount('/content/drive')
!python /content/drive/MyDrive/cat_dog_cnn_transfer_learning_resnet50.py
```
```sh
Found 6303 images belonging to 2 classes.
Found 18697 images belonging to 2 classes.
Epoch 1/35
6/6 [==============================] - ETA: 0s - loss: 0.7941 - accuracy: 0.6667
Epoch 1: accuracy improved from -inf to 0.66667, saving model to cat_dog_cnn_transfer_learning_resnet50.pybest_model_resnet50.h5
6/6 [==============================] - 439s 86s/step - loss: 0.7941 - accuracy: 0.6667 - val_loss: 0.2309 - val_accuracy: 0.9253
...
Epoch 9/35
6/6 [==============================] - ETA: 0s - loss: 0.3229 - accuracy: 0.9010
Epoch 9: accuracy improved from 0.84375 to 0.90104, saving model to cat_dog_cnn_transfer_learning_resnet50.pybest_model_resnet50.h5
6/6 [==============================] - 338s 67s/step - loss: 0.3229 - accuracy: 0.9010 - val_loss: 0.0813 - val_accuracy: 0.9774
...
Epoch 29: accuracy did not improve from 0.90104
6/6 [==============================] - 201s 40s/step - loss: 0.3590 - accuracy: 0.8333 - val_loss: 0.0619 - val_accuracy: 0.9826
Epoch 29: early stopping
```
We got accuracy ~90% with transfer learning by resnet50
## Finalize the Model and Make Predictions
The process of model improvement may continue for as long as we have ideas to test them out. Review what we got
- **VGG 3 blocks: 74.172%**
- **VGG 3 blocks + Dropout: 72.038%**
- **VGG 3 blocks + Batch normalization: 76.905%**
- **VGG 3 blocks + Augmentation: 79.018%**
- **VGG-16 transfer learning + Augmentation: 96.962%**
- **Resnet50 transfer learning+ Augmentation: 90.104%**

So we choose the VGG-16 transfer learning approach as the final model.</br>
Till now, we divided the training dataset by 75%-25% for deciding which model should be the final one. Now, we can finally create our model by fitting on the 100% training dataset. Then save the model to a _.h5_ file. We will then load the saved model and use it to make a prediction on a single image.
So we create the folder as below</br> ![finalize_folder](https://user-images.githubusercontent.com/73010204/173277979-f0294c34-e71c-4d96-8e45-43d63f99a312.png)</br>
and here is completed final model creation implementation
```sh
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
from tensorflow.keras.callbacks import ModelCheckpoint

def prepare_final_directories():
    #  final_dogs_cats
    # |---cats
    # |---dogs
    dataset_home = 'final_dogs_cats/'
    if os.path.isdir(dataset_home):
        return
    labeldirs = ['dogs/', 'cats/']
    for labeldir in labeldirs:
        newdir = dataset_home + labeldir
        makedirs(newdir, exist_ok=True)
    # copy training dataset images into subdirectories
    src_directory = 'train/'
    for file in listdir(src_directory):
        src = src_directory + '/' + file
        if file.startswith('cat'):
            dst = dataset_home + 'cats/' + file
            copyfile(src, dst)
        elif file.startswith('dog'):
            dst = dataset_home + 'dogs/' + file
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

def execute():
    # define model
	model = define_model()
	# create data generators
	datagen = ImageDataGenerator(rotation_range=90, 
                                     brightness_range=[0.1, 0.7],
                                     width_shift_range=0.5, 
                                     height_shift_range=0.5,
                                     horizontal_flip=True, 
                                     vertical_flip=True,
                                     validation_split=0.15,
                                     preprocessing_function=preprocess_input) # VGG16 preprocessing
	# prepare iterators
	train_it = datagen.flow_from_directory('final_dogs_cats/',
		class_mode='binary', batch_size=32, target_size=(224, 224))
    
	es = EarlyStopping(monitor='accuracy', mode='max', verbose=1,  min_delta=0, patience=5, restore_best_weights=True)
	mc = ModelCheckpoint('best_model.h5', monitor='accuracy', mode='max', verbose=1, save_best_only=True)
 
	model.fit(train_it, steps_per_epoch=train_it.samples//train_it.batch_size, epochs=50, verbose=1, callbacks=[es,mc])
 
	# save model
	model.save('final_model.h5')
 
# main program
prepare_final_directories()
execute()
```
**Notice:** Saving and loading models requires that HDF5 support by
```sh
sudo pip install h5py
```
Note: 
- _decode_predictions_ which is imported from _tensorflow.keras.applications.vgg16_ is used for decoding predictions of 1000 labels of classes in the ImageNet dataset. However,our model has only 1 class according to only 1 output. Therefore, it does not make sense to use _decode_predictions_ here. 
- The subdirectories of images, one for each class, are loaded by the _flow_from_directory()_ function in alphabetical order and assigned an integer for each class. The subdirectory “cat” comes before “dog“, therefore the class labels are assigned the integers: cat=0, dog=1. This can be changed via the “classes” argument in calling _flow_from_directory()_ when training the model.

In the prediction model below, we input 'test1/27.jpg' for predicting and expect the output will be 1 and it does!
![27](https://user-images.githubusercontent.com/73010204/173303633-d4ea500a-f3fb-480d-9d9d-8eafb78b6bd8.jpg)</br>
```sh
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions
import os.path

# load the model
if os.path.exists('final_model.h5') == False:
    print(f'No model found')
    exit
model = load_model('final_model.h5')
image = load_img('test1/27.jpg', target_size=(224, 224))
image  = img_to_array(image)
image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
# prepare the image for the VGG16  model
image = preprocess_input(image)
# predict the probability across all output classes
yhat = model.predict(image)
print(yhat[0])
```

## Reference
https://www.analyticsvidhya.com/blog/2021/06/beginner-friendly-project-cat-and-dog-classification-using-cnn/#:~:text=Python%20Structured%20Data-,Cat%20and%20dog%20classification%20using%20CNN,differentiates%20one%20from%20the%20other.
https://chroniclesofai.com/transfer-learning-with-keras-resnet-50/
https://www.learndatasci.com/tutorials/hands-on-transfer-learning-keras/
https://machinelearningknowledge.ai/keras-implementation-of-resnet-50-architecture-from-scratch/

