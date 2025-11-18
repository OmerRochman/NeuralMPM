import argparse
import itertools

from dawgz import Job, schedule

from neuralmpm.configs import configs
from neuralmpm.util.slurm import (
    SLURM_ACCOUNT,
    SLURM_TRAINING_CPUS,
    SLURM_TRAINING_PARTITION,
    SLURM_TRAINING_RAM,
    SLURM_TRAINING_TIME,
)


class Experiment:
    def __init__(self, experiment_name):
        self.backend = None
        self.config_dicts = None
        self.experiment_name = experiment_name

        # Load default parameters
        self.experiment_params = configs.DEFAULT

        # Override with experiment-specific parameters
        experiment_params = None
        for key, value in configs.__dict__.items():
            if key == experiment_name:
                experiment_params = value
                break
        if experiment_params is None:
            raise ValueError(f"Experiment {experiment_name} not found.")
        self.experiment_params.update(experiment_params)

        # Generate job configs.
        self.gen_config_dicts()

    def training_job(self, config_dict):
        import torch
        from neuralmpm.pipelines import training

        device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"Experiment {config_dict['run_name']} starting..")

        use_wandb = True

        config_dict["use_wandb"] = use_wandb

        training.setup_train_eval(
            config_dict, progress_bars=self.backend == "async", device=device
        )

    def gen_config_dicts(self):
        exp_type = self.experiment_params.pop("exp_type", "explicit")

        for key, value in self.experiment_params.items():
            if not isinstance(value, tuple):
                self.experiment_params[key] = (value,)

        self.config_dicts = []

        if exp_type == "explicit":
            nb_runs = max([len(v) for v in self.experiment_params.values()])
            varying_params = []
            for key, value in self.experiment_params.items():
                if len(value) == 1:
                    self.experiment_params[key] = value * nb_runs
                else:
                    if isinstance(value[0], dict):
                        formatted_dicts = [
                            " ".join([f"{k[0]}={v}" for k, v in v.items()])
                            for v in value
                        ]
                        varying_params.append(formatted_dicts)
                    else:
                        varying_params.append([f"{key}={v}" for v in value])
            run_names = list(zip(*varying_params))
            run_names = [" ".join(run_name) for run_name in run_names]

            for i in range(nb_runs):
                config_dict = {}
                for key, value in self.experiment_params.items():
                    try:
                        config_dict[key] = value[i]
                    except Exception:
                        print("Error in generating config_dicts!")
                        print(f"{nb_runs=}")
                        print(f"{key=}, {value=}")
                        print(f"{i=}")
                if run_names and "run_name" not in config_dict:
                    config_dict["run_name"] = run_names[i]
                self.config_dicts.append(config_dict)

        else:  # exp_type == 'combi'
            param_combinations = itertools.product(*self.experiment_params.values())

            # TODO Run name

            for params in param_combinations:
                config_dict = dict(zip(self.experiment_params.keys(), params))
                self.config_dicts.append(config_dict)

    def run(self, data, backend="slurm", save_every=10):
        self.backend = backend

        # Set the data of each config_dict
        for config_dict in self.config_dicts:
            if data is not None:
                config_dict["data"] = data
            if save_every is not None:
                config_dict["save_every"] = save_every

        num_jobs = len(self.config_dicts)

        j = Job(
            lambda i: self.training_job(self.config_dicts[i]),
            name=self.experiment_name,
            array=num_jobs,
            partition=SLURM_TRAINING_PARTITION,
            cpus=SLURM_TRAINING_CPUS,
            ram=SLURM_TRAINING_RAM,
            time=SLURM_TRAINING_TIME,
            gpus=1,
        )

        slurm_params = {
            "name": "neuralmpm training",
            "backend": backend,
            "export": "ALL",
            "shell": "/bin/sh",
            "env": ["export WANDB_SILENT=true"],
        }
        if SLURM_ACCOUNT is not None:
            slurm_params["account"] = SLURM_ACCOUNT
        schedule(j, **slurm_params)

        print(f"[{backend}]: Scheduled {num_jobs} job{'s' if num_jobs > 1 else ''}.")


def main():
    parser = argparse.ArgumentParser("Neural MPM Experiment Runner")
    parser.add_argument("experiment", type=str)
    parser.add_argument("-d", "--data", type=str, help="Override experiment's data.")
    parser.add_argument("-l", "--local", action="store_true")
    parser.add_argument("-s", "--save-every", type=int, default=60)

    args = parser.parse_args()
    backend = "async" if args.local else "slurm"
    save_every = args.save_every
    experiment = Experiment(args.experiment)
    experiment.run(args.data, backend=backend, save_every=save_every)


if __name__ == "__main__":
    main()
