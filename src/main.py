import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds
from google.protobuf import json_format
from metaflow import FlowSpec, step, conda_base
from tensorflow_datasets.core.proto import dataset_info_pb2

from model import create_mobilevit
from preprocessing import prepare_dataset


@conda_base(
    libraries={
        "tensorflow": "2.6.0",
        "keras": "2.6.0",
        "tensorflow-datasets": "4.4.0",
        "protobuf": "3.19.1",
    },
    python="3.8.8",
)
class MobileViTFlow(FlowSpec):
    @staticmethod
    def as_dataframe_with_proto(dataset, json_info):
        """Мы можем частично восстановить DatasetInfo объект как ProtoBuf из json-строки
        И использовать его для построения Pandas датафрейма"""
        ds_info_proto = json_format.Parse(json_info, dataset_info_pb2.DatasetInfo())
        x_key, y_key = (
            ds_info_proto.supervised_keys.input,
            ds_info_proto.supervised_keys.output,
        )
        dataset = dataset.map(lambda x, y: {x_key: x, y_key: y})
        return tfds.as_dataframe(dataset)

    @staticmethod
    def df_generator(list_of_images, labels):
        """Генератор для построения из пандаса обратно TF dataset"""
        for img, label in zip(list_of_images, labels):
            yield img, label

    @staticmethod
    def restore_tf_dataset_from_pandas(df, out_types, out_shape):
        dataset = tf.data.Dataset.from_generator(
            lambda: MobileViTFlow.df_generator(df.image.values, df.label.values),
            output_types=out_types,
            output_shapes=out_shape,
        ).prefetch(tf.data.AUTOTUNE)
        return dataset

    @step
    def start(self):
        self.hyperparams = {
            "patch_size": 4,
            "image_size": 256,
            "expansion_factor": 2,
        }
        self.preprocessing_params = {
            "batch_size": 64,
            "image_size": 256,
            "resize_bigger": 280,
            "num_classes": 5,
        }
        self.next(self.load_datasets)

    @step
    def load_datasets(self):
        """Загружаем датасет и сохраняем его в .metaflow"""
        datasets, info = tfds.load(
            "tf_flowers",
            split=["train[:90%]", "train[90%:]"],
            as_supervised=True,
            with_info=True,
        )
        train_tf_dataset, val_tf_dataset = datasets

        self.num_train = train_tf_dataset.cardinality()
        self.num_val = val_tf_dataset.cardinality()
        print(f"Number of training examples: {self.num_train}")
        print(f"Number of validation examples: {self.num_val}")

        self.train_dataset = tfds.as_dataframe(train_tf_dataset, info)
        self.val_dataset = tfds.as_dataframe(val_tf_dataset, info)
        self.json_info = info.as_json
        self.next(self.preprocessing)

    @step
    def preprocessing(self):
        """Препроцессим данные, а также сохраняем их перед обучением"""
        out_types = (tf.uint8, tf.int64)
        out_shapes = ((None, None, 3), ())
        # Считываем загруженные данные на предыдущем шаге
        train_tf_dataset = self.restore_tf_dataset_from_pandas(
            self.train_dataset, out_types, out_shapes
        )
        val_tf_dataset = self.restore_tf_dataset_from_pandas(
            self.val_dataset, out_types, out_shapes
        )

        # препроцессинг
        tf_preprocessed_train = train_tf_dataset = prepare_dataset(
            train_tf_dataset, **self.preprocessing_params, is_training=True
        )
        tf_preprocessed_val = prepare_dataset(
            val_tf_dataset, **self.preprocessing_params, is_training=False
        )

        # сохраняем
        self.preproc_train_dataset = self.as_dataframe_with_proto(
            tf_preprocessed_train, self.json_info
        )
        self.preproc_val_dataset = self.as_dataframe_with_proto(
            tf_preprocessed_val, self.json_info
        )
        self.next(self.train_and_save_model)

    @step
    def train_and_save_model(self):
        """Обучаем модель и сохраняем результат в tflite файле"""
        # restore data
        out_types = (tf.float32, tf.float32)
        out_shapes = ((256, 256, 3), (5,))
        batch_size = self.preprocessing_params["batch_size"]
        num_classes = self.preprocessing_params["num_classes"]

        preproc_train_tf = (
            self.restore_tf_dataset_from_pandas(
                self.preproc_train_dataset, out_types, out_shapes
            )
            .batch(batch_size)
            .prefetch(tf.data.AUTOTUNE)
        )
        preproc_val_tf = (
            self.restore_tf_dataset_from_pandas(
                self.preproc_val_dataset, out_types, out_shapes
            )
            .batch(batch_size)
            .prefetch(tf.data.AUTOTUNE)
        )

        # training params
        learning_rate = 0.002
        label_smoothing_factor = 0.1
        epochs = 30
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        loss_fn = keras.losses.CategoricalCrossentropy(
            label_smoothing=label_smoothing_factor
        )

        # create model
        mobilevit_xxs = create_mobilevit(num_classes=num_classes)
        mobilevit_xxs.compile(optimizer=optimizer, loss=loss_fn, metrics=["accuracy"])

        checkpoint_filepath = "/tmp/checkpoint"
        checkpoint_callback = keras.callbacks.ModelCheckpoint(
            checkpoint_filepath,
            monitor="val_accuracy",
            save_best_only=True,
            save_weights_only=True,
        )

        # fit it
        mobilevit_xxs.fit(
            preproc_train_tf,
            validation_data=preproc_val_tf,
            epochs=epochs,
            callbacks=[checkpoint_callback],
        )
        mobilevit_xxs.load_weights(checkpoint_filepath)
        _, accuracy = mobilevit_xxs.evaluate(preproc_val_tf)

        self.accuracy = round(accuracy * 100, 2)
        mobilevit_xxs.save("mobilevit_xxs")

        # Convert to TFLite. This form of quantization is called
        # post-training dynamic-range quantization in TFLite.
        converter = tf.lite.TFLiteConverter.from_saved_model("mobilevit_xxs")
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,  # Enable TensorFlow Lite ops.
            tf.lite.OpsSet.SELECT_TF_OPS,  # Enable TensorFlow ops.
        ]
        tflite_model = converter.convert()
        open("mobilevit_xxs.tflite", "wb").write(tflite_model)
        self.next(self.end)

    @step
    def end(self):
        print(f"Model saved, Validation accuracy: {self.accuracy}%")


if __name__ == "__main__":
    MobileViTFlow()
