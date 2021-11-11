import tensorflow as tf


def preprocess_dataset(image_size, resize_bigger, num_classes, is_training=True, **__):
    def _pp(image, label):
        if is_training:
            # Resize to a bigger spatial resolution and take the random
            # crops.
            image = tf.image.resize(image, (resize_bigger, resize_bigger))
            image = tf.image.random_crop(image, (image_size, image_size, 3))
            image = tf.image.random_flip_left_right(image)
        else:
            image = tf.image.resize(image, (image_size, image_size))
        label = tf.one_hot(label, depth=num_classes)
        return image, label

    return _pp


def prepare_dataset(
    dataset, batch_size, image_size, resize_bigger, num_classes, is_training=True
):
    if is_training:
        dataset = dataset.shuffle(batch_size * 10)
    dataset = dataset.map(
        preprocess_dataset(image_size, resize_bigger, num_classes, is_training),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    return dataset
