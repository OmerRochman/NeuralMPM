import json
import os
import time
from datetime import datetime

import torch
import wandb

from neuralmpm.models.model_loader import create_model

# TODO Clean


class ModelLogger:
    def __init__(
        self,
        dataset_name: str | None,
        run_config: dict,
        save_interval: int = 10,
        save_only_last: bool = False,
        create_wandb_json: bool = True,
        parent_dir: str = "outputs",
    ):
        if "run_id" not in run_config:
            try:
                self.run_name = wandb.run.name
                self.run_id = wandb.run.id
            except Exception:
                self.run_name = datetime.now().strftime("(%d_%m) %H:%M:%S")
                self.run_id = 0
                # print(f'[Warning]: Wandb not used, setting run name to '
                #      f'{self.run_name}_{self.run_id}')
        else:
            self.run_name = run_config["run_name"]
            self.run_id = run_config["run_id"]

        self.dataset_name = dataset_name

        self.save_interval = save_interval
        self.save_only_last = save_only_last

        self.project_name = run_config.get("project", "experiments")

        self.folder = f"{parent_dir}/{self.project_name}/{self.run_name}_{self.run_id}"
        self.model_folder = f"{self.folder}/models"
        os.makedirs(self.folder, exist_ok=True)
        os.makedirs(self.model_folder, exist_ok=True)

        self.run_config = run_config
        self.model_name = run_config["model"]
        self.model_architecture = run_config["architecture"]

        # if file does not exist
        if not os.path.exists(f"{self.folder}/config.json"):
            with open(f"{self.folder}/config.json", "w") as f:
                json.dump(self.run_config, f, indent=4)

        # TODO: This should be in another function
        # This class is a bit dirty, it should be refactored
        # to allow for easily both checkpointing, saving and loading models
        if create_wandb_json:
            if self.run_id != 0:
                wandb_info = {
                    "run_name": self.run_name,
                    "run_id": self.run_id,
                    "link": f"https://wandb.ai/{wandb.run.entity}/{wandb.run.project}/runs/{self.run_id}",
                }
                with open(f"{self.folder}/wandb.json", "w") as f:
                    json.dump(wandb_info, f, indent=4)

        self.last_time = None
        self.total_time_start = None

        self.best_val = torch.inf

    def start_timer(self):
        self.total_time_start = time.time()
        self.last_time = time.time()

    def save_model(
        self,
        model,
        checkpoint_name: str = None,
        json_dict: dict = None,
    ):
        """
        Save model checkpoint.
        Args:
            json_dict:
            checkpoint_name:
            model: Model to save.
        """

        if self.save_only_last:
            for file in os.listdir(self.model_folder):
                os.remove(f"{self.model_folder}/{file}")

        torch.save(model.state_dict(), f"{self.model_folder}/{checkpoint_name}.ckpt")

        # TODO: save info about when (epoch+time) the model was saved
        # for best only
        # print(f"Model saved at {self.model_folder}/{checkpoint_name}.ckpt.")
        if json_dict:
            elapsed_time_seconds = time.time() - self.total_time_start
            hours = int(elapsed_time_seconds // 3600)
            minutes = int((elapsed_time_seconds % 3600) // 60)
            seconds = int(elapsed_time_seconds % 60)
            json_dict["elapsed_time"] = f"{hours:02}:{minutes:02}:{seconds:02}"
            with open(f"{self.model_folder}/{checkpoint_name}.json", "w") as f:
                json.dump(json_dict, f, indent=4)

    def try_saving(self, model):
        """
        Try saving model checkpoint.
        Args:
            model: Model to save.
        """

        if self.last_time is None:
            raise ValueError("Timer not started.")

        current_time = time.time()
        current_time_diff = current_time - self.last_time
        current_time_diff = int(current_time_diff // 60)

        if current_time_diff >= self.save_interval:
            total_time = int((current_time - self.total_time_start) // 60)

            self.save_model(model, str(total_time))
            self.last_time = current_time

    def load(self, checkpoint="best"):
        model = create_model(self.model_name, self.run_config)

        if checkpoint == "last":
            checkpoint = sorted(
                [
                    int(file.split(".")[0])
                    for file in os.listdir(self.model_folder)
                    if file.endswith(".ckpt")
                ]
            )[-1]
            print("Loading last saved checkpoint:", checkpoint)

        checkpoint_path = os.path.join(self.model_folder, f"{checkpoint}.ckpt")
        model.load_state_dict(torch.load(checkpoint_path))

        return model
