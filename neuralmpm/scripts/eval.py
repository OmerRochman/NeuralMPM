# TODO: Several parts should be moved to evaluation.py, only the script part should be here.

import argparse
import json
import os
from glob import glob
import pickle
import shutil

import h5py
from natsort import natsorted
from tqdm import tqdm
from dawgz import job, after, ensure, schedule
from neuralmpm.util.slurm import (
    SLURM_ACCOUNT,
    SLURM_EVAL_COMPUTE_CPUS,
    SLURM_EVAL_COMPUTE_GPUS,
    SLURM_EVAL_COMPUTE_PARTITION,
    SLURM_EVAL_COMPUTE_RAM,
    SLURM_EVAL_COMPUTE_TIME,
    SLURM_EVAL_MERGE_CPUS,
    SLURM_EVAL_MERGE_PARTITION,
    SLURM_EVAL_MERGE_RAM,
    SLURM_EVAL_MERGE_TIME,
)
import numpy as np

from neuralmpm.util.metrics import mse_np, emd


def sim_stats(file, use_emd=False, num_chunks=12, device="cpu"):
    with h5py.File(file, "r") as f:
        gt_traj = f["ground_truth_rollout"][()]
        pred_traj = f["predicted_rollout"][()]
        types = f["types"][()]
        types = np.array([np.where(np.unique(types) == t)[0][0] for t in types])

        if gt_traj.shape[-1] > 3:
            dim = gt_traj.shape[-1] // 2
            gt_traj = gt_traj[:, :, :dim]
            pred_traj = pred_traj[:, :, :dim]
        else:
            dim = gt_traj.shape[-1]

    computed_metrics = {
        "mse": mse_np(gt_traj, pred_traj, types),
    }
    if use_emd:
        mask = types != 0
        computed_metrics["emd"] = emd(
            pred_traj[:, mask], gt_traj[:, mask], num_chunks=num_chunks
        )

    return {
        k: {
            "steps": v.tolist(),
            "mean": v.mean().item(),
            "std": v.std().item(),
        }
        for k, v in computed_metrics.items()
    }


def eval_rollouts(
    run_path: str, dataset: str, use_emd=False, num_chunks=12, backend: str = "local"
):
    if dataset is None:
        with open(os.path.join(run_path, "config.json"), "r") as f:
            config = json.load(f)
            dataset = config["data"].split("/")[-1]

    rollouts_path = os.path.join(run_path, "rollouts", dataset)
    assert os.path.exists(rollouts_path), "Rollouts do not exist."

    metrics_path = os.path.join(run_path, "metrics")
    os.makedirs(metrics_path, exist_ok=True)

    files = glob(rollouts_path + "/*.h5")
    files = natsorted(files)
    num_sims = len(files)

    # In case several evaluations are run in parallel for a same run.
    uid = np.random.randint(0, 1e6)
    tmp_folder = os.path.join(run_path, f"tmp_eval_{uid}")
    os.makedirs(tmp_folder, exist_ok=True)

    @ensure(lambda sim_id: os.path.exists(f"{tmp_folder}/{sim_id}.pkl"))
    @job(
        name="eval_1sim",
        cpus=SLURM_EVAL_COMPUTE_CPUS,
        gpus=SLURM_EVAL_COMPUTE_GPUS,
        partition=SLURM_EVAL_COMPUTE_PARTITION,
        ram=SLURM_EVAL_COMPUTE_RAM,
        time=SLURM_EVAL_COMPUTE_TIME,
        array=num_sims,
    )
    def single_sim_stats(sim_id: int):
        metrics_dict = sim_stats(files[sim_id], use_emd=use_emd, num_chunks=num_chunks)
        pickle.dump(metrics_dict, open(f"{tmp_folder}/{sim_id}.pkl", "wb"))

    @after(single_sim_stats)
    @job(
        name="merge_evals",
        cpus=SLURM_EVAL_MERGE_CPUS,
        ram=SLURM_EVAL_MERGE_RAM,
        time=SLURM_EVAL_MERGE_TIME,
        parition=SLURM_EVAL_MERGE_PARTITION,
    )
    def merge_eval():
        metrics = {
            "mean": {},
            "std": {},
            "runs": {},
        }

        for sim_id in range(num_sims):
            metrics_dict = pickle.load(open(f"{tmp_folder}/{sim_id}.pkl", "rb"))
            metrics["runs"][str(sim_id)] = metrics_dict

        shutil.rmtree(tmp_folder)

        for key in metrics["runs"]["0"].keys():
            values = [v[key]["mean"] for v in metrics["runs"].values()]
            metrics["mean"][key] = sum(values) / len(values)
            metrics["std"][key] = sum(
                [(v - metrics["mean"][key]) ** 2 for v in values]
            ) / len(values)

            metric_mean = metrics["mean"][key] * 1e3
            metric_std = metrics["std"][key] * 1e3

            print(f"{key.upper()}: {metric_mean:.2f} ± {metric_std:.2f}")

        with open(os.path.join(metrics_path, f"{dataset}.json"), "w") as f:
            json.dump(metrics, f, indent=4)

        return metrics["mean"]

    if backend == "local":
        for i in tqdm(range(num_sims)):
            single_sim_stats(i)
        return merge_eval()
    else:
        slurm_params = {
            "name": "neuralmpm eval",
            "backend": "slurm",
            "export": "ALL",
            "shell": "/bin/sh",
        }
        if SLURM_ACCOUNT is not None:
            slurm_params["account"] = SLURM_ACCOUNT
        schedule(merge_eval, **slurm_params)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--run", "-r", type=str, help="Path to the run folder.", required=True
    )
    parser.add_argument(
        "--data",
        "-d",
        type=str,
        help="Dataset name the model was rolled out on.",
        required=False,
    )
    parser.add_argument(
        "--emd",
        "-E",
        action="store_true",
        help="Whether to evaluate rollouts using the Earth Mover's Distance.",
    )
    parser.add_argument(
        "--num-chunks",
        "-nc",
        type=int,
        default=12,
        help="Number of chunks to split the data into for EMD evaluation.",
    )
    parser.add_argument(
        "--distributed",
        "-D",
        action="store_true",
        help="Whether to evaluate rollouts in a distributed manner.",
    )
    args = parser.parse_args()

    eval_rollouts(
        args.run,
        args.data,
        use_emd=args.emd,
        num_chunks=args.num_chunks,
        backend="slurm" if args.distributed else "local",
    )


if __name__ == "__main__":
    main()
