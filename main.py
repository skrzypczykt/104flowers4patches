import tensorflow as tf

path = 'data/train'
(img_height, img_width) = 512, 512
(input_height, input_width) = (224, 224)
N_CLASSES = 22
BATCH_SIZE = 36

train_data = tf.keras.utils.image_dataset_from_directory(
    path,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=None)

val_data = tf.keras.utils.image_dataset_from_directory(
    path,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=None)

trainAug = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal_and_vertical"),
    tf.keras.layers.RandomZoom(
        height_factor=(-0.05, -0.15),
        width_factor=(-0.05, -0.15)),
    tf.keras.layers.RandomContrast(factor=0.2),
    tf.keras.layers.RandomRotation(0.3)
])


def prepare_patches(image, label):
    image_expanded = tf.expand_dims(input=image, axis=0)
    images_4 = tf.image.extract_patches(images=image_expanded,
                                        sizes=[1, int(0.5 * img_height), int(0.5 * img_width), 1],
                                        strides=[1, int(0.5 * img_height), int(0.5 * img_width), 1],
                                        rates=[1, 1, 1, 1],
                                        padding='VALID')
    reconstructed_4 = tf.reshape(images_4, [-1, int(img_height / 2), int(img_width / 2), 3])

    # resize images
    image_resized = tf.image.resize(images=image_expanded, size=(input_height, input_width))
    patches_4_resized = tf.image.resize(images=reconstructed_4, size=(input_height, input_width))

    return tf.concat([image_resized, patches_4_resized], axis=0), label


train_ds = train_data \
    .map(lambda x, y: (trainAug(x), y),
         num_parallel_calls=tf.data.AUTOTUNE) \
    .map(prepare_patches) \
    .batch(BATCH_SIZE) \
    .prefetch(tf.data.AUTOTUNE)

val_ds = val_data \
    .map(prepare_patches) \
    .batch(BATCH_SIZE) \
    .prefetch(tf.data.AUTOTUNE)


class PatchedModel(tf.keras.Model):
    def __init__(self):
        super(PatchedModel, self).__init__()
        self.backbone = tf.keras.applications.convnext.ConvNeXtTiny(
            include_top=False,
            weights='imagenet',
        pooling='max')
        self.backbone.trainable = False

        self.global_max_pooling = tf.keras.layers.GlobalMaxPool1D()
        self.output_layer = tf.keras.layers.Dense(N_CLASSES, activation="softmax")

    def call(self, inputs):
        embeddings = []
        for i in range(5):
            embeddings.append(self.backbone(inputs[:, i]))
        embeddings_stacked = tf.stack(embeddings, axis=1)
        embedding = self.global_max_pooling(embeddings_stacked)
        output = self.output_layer(embedding)
        return output


model = PatchedModel()
model.compile(loss="sparse_categorical_crossentropy", optimizer='adam', metrics=["accuracy"])
model.evaluate(val_ds)
print(model.summary())
model.fit(train_ds, validation_data=val_ds, epochs=5)
