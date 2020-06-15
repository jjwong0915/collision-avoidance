import tensorflow
import os.path as path


class DangerIndexTrainer:
    def __init__(self, model, data):
        self.model = model
        self.data = data

    def train(self, checkpoint_dir, epoch):
        self.model.compile(
            optimizer="adam", loss=tensorflow.keras.losses.mean_squared_error,
        )
        #
        checkpoint = tensorflow.keras.callbacks.ModelCheckpoint(
            filepath=path.join(checkpoint_dir, "{epoch:02d}-{loss:.4f}.h5"),
            save_weights_only=True,
        )
        self.model.fit(
            self.data,
            steps_per_epoch=len(self.data),
            epochs=epoch,
            callbacks=[checkpoint],
        )
