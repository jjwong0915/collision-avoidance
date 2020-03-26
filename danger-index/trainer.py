import tensorflow


class DangerIndexTrainer:
    def __init__(model, data):
        this.model = model
        this.data = data

    def train(epoch):
        this.model.compile(
            optimizer="adam",
            loss=tensorflow.keras.losses.binary_crossentropy,
            metrics=["accuracy"],
        )
        history = this.model.fit(
            this.data, steps_per_epoch=len(this.data), epochs=epoch,
        )
        print(history)
