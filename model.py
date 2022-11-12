import tensorflow as tf
from tensorflow import keras
from keras import layers


data_augmentation = tf.keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(.1)
    ]
)

def create_model(input_shape, num_classes):
    inputs = keras.Input(shape=input_shape)

    m = data_augmentation(inputs)

    m = layers.Rescaling(1/255)(m)
    m=  layers.Conv2D(32, 3, strides=2, padding="same")(m)
    m=  layers.BatchNormalization()(m)

    m=layers.Activation("relu")(m)
    m=layers.Conv2D(64,3,padding="same")(m)
    m=layers.BatchNormalization()(m)
    m=layers.Activation("relu")(m)

    activation1= m

    for size in [128,256,512,728]:
        m = layers.Activation("relu")(m)
        m=layers.SeparableConv2D(size,3,padding="same")(m)
        m=layers.BatchNormalization()(m)
        m=layers.Activation("relu")(m)
        m=layers.SeparableConv2D(size,3,padding="same")(m)
        m=layers.BatchNormalization()(m)
        m=layers.MaxPooling3D(3,strides=2,padding="same")(m)

        residual =layers.Conv2D(size,1,strides=2,padding="same")(activation1)

        m=layers.add([m,residual])
        activation1=m

    m=layers.SeparableConv2D(1024,3,padding="same")(m)
    m=layers.BatchNormalization()(m)
    m=layers.Activation("relu")(m)
    m=layers.GlobalAveragePooling2D()(m)

    if num_classes==2:
        activate="sigmoid"
        units=1
    else:
        activate="softmax"
        units=num_classes
    m=layers.Dropout(.5)(m)
    outputs=layers.Dense(units)(activation=activate)(m)

    return tf.keras.Model(inputs,outputs)



    