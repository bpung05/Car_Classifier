import tensorflow as tf

def getDataset (fileLocation) :

    image_size = (180,180)
    batch_size = 32

    train = tf.keras.preprocessing.image_dataset_from_directory(
        fileLocation,
        labels="inferred",
        validation_split=.2,
        subset="training",
        seed=1200,
        image_size=image_size,
        batch_size=batch_size,

        )


    test = tf.keras.preprocessing.image_dataset_from_directory(
        fileLocation,
        labels="inferred",
        validation_split=.2,
        subset="validation",
        seed=1200,
        image_size=image_size,
        batch_size=batch_size,
        )
    

    return train, test
