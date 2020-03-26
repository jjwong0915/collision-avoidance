import dataloader
import model
import trainer

DATA_DIRECTORY = "/home/jjwong0915/Documents/drone/depth_prediction/data/random/"
WIDTH, HEIGHT = 192, 160


def main():
    danger_model = model.DangerIndexModel((WIDTH, HEIGHT, 1))
    danger_dataloader = dataloader.DangerIndexDataloader(
        directory=DATA_DIRECTORY, size=(WIDTH, HEIGHT)
    )
    danger_trainer = trainer.DangerIndexTrainer(
        model=danger_model, data=danger_dataloader.load_data()
    )
    danger_trainer.train()


main()
