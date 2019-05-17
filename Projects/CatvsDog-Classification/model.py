from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from keras.callbacks import ModelCheckpoint


''' Initializing the CNN '''
classifier = Sequential()


''' Convolutional layer '''
classifier.add(Conv2D( 32, (3, 3), input_shape= (64, 64, 3), activation= 'relu' ))


''' Pooling layer '''
classifier.add(MaxPooling2D(pool_size= (2, 2)))


''' Adding another convolutional layer '''
classifier.add(Conv2D( 32, (3, 3), activation= 'relu' ))
classifier.add(MaxPooling2D(pool_size= (2, 2)))


''' Flattening '''
classifier.add(Flatten())


''' Full connection '''
classifier.add(Dense( units= 128, activation= 'relu' ))
classifier.add(Dense( units= 1, activation= 'sigmoid' ))


''' Compiling the CNN '''
classifier.compile(optimizer= 'adam', loss= 'binary_crossentropy', metrics= ['accuracy'])


''' Fitting CNN to the images '''
train_datagen = ImageDataGenerator(rescale= 1./255, shear_range= 0.2, zoom_range= 0.2, horizontal_flip= True)
test_datagen = ImageDataGenerator(rescale= 1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set', target_size= (64, 64), batch_size= 32, class_mode= 'binary')

test_size = test_datagen.flow_from_directory('dataset/test_set', target_size= (64, 64), batch_size= 32, class_mode= 'binary')

classifier.fit_generator(training_set, steps_per_epoch= 8000, epochs= 25, validation_data= test_size, validation_steps=2000)


''' Predicting of an Image'''
d = test_data[0]
img_data, img_num = d

data = img_data.reshape(IMG_SIZE, IMG_SIZE, 1)
prediction = model.predict([data])[0]

fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111)
ax.imshow(img_data, cmap="gray")
print(f"cat: {prediction[0]}, dog: {prediction[1]}")
