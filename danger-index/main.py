import dataloader
import dotenv
import model
import os
import trainer

dotenv.load_dotenv()

EPOCH = int(os.getenv("EPOCH"))
WIDTH, HEIGHT = int(os.getenv("WIDTH")), int(os.getenv("HEIGHT"))
DATA_DIRECTORY = os.getenv("DATA_DIR")
CHECKPOINT_DIRECTORY = os.getenv("CHECKPOINT_DIR")


def main():
    danger_model = model.DangerIndexModel((WIDTH, HEIGHT, 1))
    danger_dataloader = dataloader.DangerIndexDataloader((WIDTH, HEIGHT))
    danger_trainer = trainer.DangerIndexTrainer(
        model=danger_model, data=danger_dataloader.load_data(DATA_DIRECTORY)
    )
    danger_trainer.train(checkpoint_dir=CHECKPOINT_DIRECTORY, epoch=EPOCH)


if __name__ == "__main__":
    main()
