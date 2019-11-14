from keras.models import Sequential, Model, load_model
from keras.layers import Activation, Dense, GlobalAveragePooling2D, Dropout
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import VGG16
from keras.applications.xception import Xception, preprocess_input, decode_predictions
import keras.callbacks as cb


training_data_dir = './chest_xray/train'
val_data_dir = './chest_xray/val'
test_data_dir = './chest_xray/test'

# Parameters
output_classes = 2
learning_rate = 0.001
img_width, img_height, channel = 299, 299, 3  # dimension required for the Xception model
training_examples = 5216  # total number of training images
retrain_layers = 100
batch_size = 10
epochs = 3


xc = Xception(weights='imagenet', include_top=False, pooling='avg')
for layer in xc.layers[:-retrain_layers]:
    layer.trainable = False
dense = Dense(units=output_classes, activation='softmax')(xc.output)

model = Model(inputs=xc.input, outputs=dense)
model.compile(
              optimizer='rmsprop',
              # optimizer=Adam(lr=learning_rate),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# # Check the trainable status of the individual layers
# for layer in model.layers:
#     print(layer, layer.trainable)
# print(model.summary())

img_generator = ImageDataGenerator(preprocessing_function=preprocess_input)

print('Training set:   ', end='')
train_img_generator = img_generator.flow_from_directory(
    training_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

print('Validation set: ', end='')
val_img_generator = img_generator.flow_from_directory(
    val_data_dir,
    target_size=(img_width, img_height),
    class_mode='categorical')
#
# print('damjan set: ', end='')
# dam_img_generator = img_generator.flow_from_directory(
#     './chest_xray/damjan',
#     target_size=(img_width, img_height),
#     class_mode='categorical')

print('Test set: ', end='')

test_img_generator = img_generator.flow_from_directory(
    test_data_dir,
    target_size=(img_width, img_height),
    class_mode='categorical',
    batch_size=batch_size,
    shuffle=False)

# Callbacks
early_stop = cb.EarlyStopping(monitor='loss', min_delta=0.0001)
tensorboard = cb.TensorBoard('.\\xcpn3', histogram_freq=1,write_graph=True, write_images=True,update_freq='batch')

# py -m tensorboard.main --logdir="C:\Users\User\PycharmProjects\pneumonia\xcpn3"

model.fit_generator(train_img_generator,
                    steps_per_epoch=training_examples // batch_size,
                    epochs=epochs,
                    validation_data=val_img_generator,
                    validation_steps=1,
                    callbacks=[early_stop, tensorboard])

## saving model
model.save('batch10_layer100_rms.h5')

# model = load_model('chest_xray_ep=10.h5')

# Evaluating the model
train_accu = model.evaluate_generator(train_img_generator,steps=training_examples // batch_size)
test_accu = model.evaluate_generator(test_img_generator,steps=624 // batch_size)
# test_accu_damj = model.predict_generator(dam_img_generator)

# Results accuracy
print('Accuracy on train data is:', train_accu[1])
print('Accuracy on test data is:', test_accu[1])
