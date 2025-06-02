"""
Computes the statistics of a dataset (mean & std) over a grid of given size,
and saves them to a file in the dataset folder.

Example:
    WaterRamps
     |
     +-- train
     +-- valid
     +-- test
     +-- stats
         |
         +-- train_32x32.h5
         +-- train_64x64.h5
         +-- train_128x128.h5
"""

import argparse
import os
import shutil

import h5py
from neuralmpm.util.slurm import (
    SLURM_ACCOUNT,
    SLURM_STATS_COMPUTE_CPUS,
    SLURM_STATS_COMPUTE_GPUS,
    SLURM_STATS_COMPUTE_PARTITION,
    SLURM_STATS_COMPUTE_RAM,
    SLURM_STATS_COMPUTE_TIME,
    SLURM_STATS_MERGE_CPUS,
    SLURM_STATS_MERGE_PARTITION,
    SLURM_STATS_MERGE_RAM,
    SLURM_STATS_MERGE_TIME,
)
import torch
from natsort import natsorted
from tqdm import tqdm

import neuralmpm.interpolation.interpolation as interp
import neuralmpm.data.data_manager
from neuralmpm.interpolation import kernels
from neuralmpm.interpolation.interpolation import get_voxel_centers, find_size

from dawgz import job, schedule, after, ensure


def sim_stats(
    sim_id, parser, split, grid_coords, size_tensor, dim, seq_length, device="cuda"
):
    sim_data = parser.parse(sim_id, split)
    sim, types = sim_data["sim"].to(device), sim_data["types"].to(device)

    pos = torch.clamp(
        sim[..., :dim],
        min=parser.low_bound.to(device),
        max=parser.high_bound.to(device),
    )
    vel = sim[..., dim:]
    types = types[None, :].expand(seq_length, -1)

    with torch.device(device):
        grid = interp.create_grid_cluster_batch(
            grid_coords.to(device),
            pos,
            vel,
            types,
            parser.get_num_types(),
            kernels.linear,
            low=parser.low_bound.to(device),
            high=parser.high_bound.to(device),
            size=size_tensor.to(device),
            interaction_radius=0.015,  # TODO
        )

        axes = list(range(dim + 1))
        mean = torch.mean(grid, dim=axes)
        mean_squares = torch.mean(grid**2, dim=axes)

    return mean, mean_squares


# TODO merge with below and use async for local
def compute_distributed(
    dataset_path,
    dataset_type="monomat2d",
    grid_size=list[int],
    split="train",
    device="cpu",
):
    """
    Compute the mean and standard deviation of the grids of the entire split
    of a dataset.

    Args:
        dataset_path (str): Path to the dataset.
        dataset_type (str): Type of dataset (monomat2d, ...)
        grid_size (list[int]): Grid size to compute the stats for.
        split (str): Split to compute the mean and standard deviation for.
    """

    parser = neuralmpm.data.data_manager.PARSERS[dataset_type](dataset_path)
    num_channels = parser.get_num_channels()
    seq_length = parser.get_sim_length()

    dim = parser.dim
    grid_size = grid_size[:dim]
    if len(grid_size) == 1:
        grid_size = [grid_size[0]] * dim
    elif len(grid_size) != dim:
        raise ValueError(f"Please refer 1 or {dim} (=dim) grid sizes.")
    grid_size_str = "x".join(map(str, grid_size))

    # TODO Cleaner, should probably have a function that directly computes
    #  this list, converts to a tensor and return it and only use that
    #  maybe even take the parser as argument instead of the bounds?
    sizes = [
        find_size(lb, hb, g)
        for (lb, hb, g) in zip(parser.low_bound, parser.high_bound, grid_size)
    ]
    size_tensor = torch.tensor(sizes)
    grid_coords = get_voxel_centers(
        size_tensor,
        parser.low_bound,
        parser.high_bound,
    )

    sim_data_keys = parser.parse(0, split).keys()

    if "grav" in sim_data_keys:
        num_channels -= dim

    files = os.listdir(os.path.join(dataset_path, split))
    files = natsorted(files)
    num_sims = len(files)

    tmp_folder = f"{dataset_path}/tmp_stats_{grid_size_str}"
    os.makedirs(tmp_folder, exist_ok=True)

    @ensure(lambda sim_id: os.path.exists(f"{tmp_folder}/{sim_id}.pt"))
    @job(
        name="stats_1sim",
        cpus=SLURM_STATS_COMPUTE_CPUS,
        gpus=SLURM_STATS_COMPUTE_GPUS,
        partition=SLURM_STATS_COMPUTE_PARTITION,
        ram=SLURM_STATS_COMPUTE_RAM,
        time=SLURM_STATS_COMPUTE_TIME,
        array=num_sims,
    )
    def single_sim_stats(sim_id: int):
        mean, mean_squares = sim_stats(
            sim_id,
            parser,
            split,
            grid_coords,
            size_tensor,
            dim,
            seq_length,
            device=device,
        )

        stats = {"mean": mean.cpu(), "mean_squares": mean_squares.cpu()}

        torch.save(stats, f"{tmp_folder}/{sim_id}.pt")

    @after(single_sim_stats)
    @job(
        name="merge_stats",
        cpus=SLURM_STATS_MERGE_CPUS,
        partition=SLURM_STATS_MERGE_PARTITION,
        ram=SLURM_STATS_MERGE_RAM,
        time=SLURM_STATS_MERGE_TIME,
    )
    def merge_stats():
        mean = torch.zeros(num_channels)
        mean_squares = torch.zeros(num_channels)
        for i in range(num_sims):
            stats = torch.load(f"{tmp_folder}/{i}.pt")
            mean += stats["mean"]
            mean_squares += stats["mean_squares"]

        shutil.rmtree(tmp_folder)

        mean /= num_sims
        mean_squares /= num_sims
        std = torch.sqrt(mean_squares - mean**2)

        # TODO: Either we concatenate when loading
        # either we fix this num_channels thing
        # num_channels should probably be the channels
        # without gravity, as for a same dataset we might
        # integrate gravity differently than channels?
        if "grav" in sim_data_keys:
            # in Parser class? like parser.process(mean, std)
            mean[4:] = 0
            std[4:] = 1

        print(f"Statistics computed for {dataset_path} ({dataset_type}, {split})")
        print("Mean:", mean.tolist())
        print("Std:", std.tolist())

        try:
            os.makedirs(os.path.join(dataset_path, "stats"), exist_ok=True)
            save_path = os.path.join(
                dataset_path, "stats", f"{split}_{grid_size_str}.h5"
            )

            with h5py.File(save_path, "w") as f:
                f.create_dataset("mean", data=mean.cpu().numpy())
                f.create_dataset("std", data=std.cpu().numpy())
        except PermissionError as e:
            print(f"PermissionError: {e.strerror}")
            save_path = (
                f"{dataset_path.split('/')[-1]}_stats_{split}_{grid_size_str}.h5"
            )
            print(f"Saving locally to {save_path}")
            with h5py.File(save_path, "w") as f:
                f.create_dataset("mean", data=mean.cpu().numpy())
                f.create_dataset("std", data=std.cpu().numpy())

    slurm_params = {
        "name": "neuralmpm dataset stats",
        "backend": "slurm",
        "export": "ALL",
        "shell": "/bin/sh",
    }
    if SLURM_ACCOUNT is not None:
        slurm_params["account"] = SLURM_ACCOUNT
    schedule(merge_stats, **slurm_params)


def compute_dataset_mstd(
    dataset_path,
    dataset_type="monomat2d",
    grid_size=list[int],
    split="train",
    device="cpu",
):
    """
    Compute the mean and standard deviation of the grids of the entire split
    of a dataset.

    Args:
        dataset_path (str): Path to the dataset.
        dataset_type (str): Type of dataset (monomat2d, ...)
        grid_size (list[int]): Grid size to compute the stats for.
        split (str): Split to compute the mean and standard deviation for.
    """

    parser = neuralmpm.data.data_manager.PARSERS[dataset_type](dataset_path)
    num_channels = parser.get_num_channels()
    seq_length = parser.get_sim_length()

    dim = parser.dim
    grid_size = grid_size[:dim]
    if len(grid_size) == 1:
        grid_size = [grid_size[0]] * dim
    elif len(grid_size) != dim:
        raise ValueError(f"Please refer 1 or {dim} (=dim) grid sizes.")

    # TODO Cleaner, should probably have a function that directly computes
    #  this list, converts to a tensor and return it and only use that
    #  maybe even take the parser as argument instead of the bounds?
    sizes = [
        find_size(lb, hb, g)
        for (lb, hb, g) in zip(parser.low_bound, parser.high_bound, grid_size)
    ]
    size_tensor = torch.tensor(sizes).to(device)
    grid_coords = get_voxel_centers(
        size_tensor,
        parser.low_bound.to(device),
        parser.high_bound.to(device),
    ).to(device)

    sim_data_keys = parser.parse(0, split).keys()

    if "grav" in sim_data_keys:
        num_channels -= dim

    mean = torch.zeros(num_channels).to(device)
    mean_squares = torch.zeros(num_channels).to(device)
    axes = list(range(dim + 1))

    for root, _, files in os.walk(os.path.join(dataset_path, split)):
        num_samples = 100  # len(files)

        files = natsorted(files)

        sizes = sizes[0].item()
        for file in tqdm(
            files[:num_samples], desc=f"Computing mean and std for {split}"
        ):
            with torch.device(device):
                if file.endswith(".h5"):
                    sim_id = int(file.split("_")[1].split(".")[0])
                    sim_data = parser.parse(sim_id, split)
                    sim, types = sim_data["sim"], sim_data["types"]

                    pos = torch.clamp(
                        sim[..., :dim],
                        min=parser.low_bound.to(device),
                        max=parser.high_bound.to(device),
                    )
                    vel = sim[..., dim:]
                    types = types[None, :].expand(seq_length, -1)

                    grid = interp.create_grid_cluster_batch(
                        grid_coords,
                        pos,
                        vel,
                        types,
                        parser.get_num_types(),
                        kernels.linear,
                        low=parser.low_bound.to(device),
                        high=parser.high_bound.to(device),
                        size=size_tensor.to(device),
                        interaction_radius=0.015,  # TODO
                    )  # Shape: [T, H, W, C]

                    mean += grid.mean(dim=axes)
                    mean_squares += torch.mean(grid**2, dim=axes)

        mean /= num_samples
        mean_squares /= num_samples
        std = torch.sqrt(mean_squares - mean**2)

        # if dataset_type in ('wbc', 'watergravity'):  # TODO Clean, function
        if "grav" in sim_data_keys:
            # in Parser class? like parser.process(mean, std)
            mean[4:] = 0
            std[4:] = 1

        print(f"Statistics computed for {dataset_path} ({dataset_type}, {split})")
        print("Mean:", mean.tolist())
        print("Std:", std.tolist())

        grid_size_str = "x".join(map(str, grid_size))

        try:
            os.makedirs(os.path.join(dataset_path, "stats"), exist_ok=True)
            save_path = os.path.join(
                dataset_path, "stats", f"{split}_{grid_size_str}.h5"
            )

            with h5py.File(save_path, "w") as f:
                f.create_dataset("mean", data=mean.cpu().numpy())
                f.create_dataset("std", data=std.cpu().numpy())
        except PermissionError as e:
            print(f"PermissionError: {e.strerror}")
            save_path = (
                f"{dataset_path.split('/')[-1]}_stats_{split}_{grid_size_str}.h5"
            )
            print(f"Saving locally to {save_path}")
            with h5py.File(save_path, "w") as f:
                f.create_dataset("mean", data=mean.cpu().numpy())
                f.create_dataset("std", data=std.cpu().numpy())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", "-d", type=str, required=True)
    parser.add_argument("--dataset_type", "-t", type=str, default="monomat2d")
    parser.add_argument("--split", "-S", type=str, default="train")
    parser.add_argument(
        "--grid-size",
        "-g",
        nargs="*",
        help="Grid size.",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--distributed", "-D", action="store_true", help="Use distributed computing."
    )
    args = parser.parse_args()

    if args.distributed:
        compute_distributed(
            args.dataset_path,
            args.dataset_type,
            args.grid_size,
            args.split,
            device="cuda",
        )
    else:
        compute_dataset_mstd(
            args.dataset_path,
            args.dataset_type,
            args.grid_size,
            args.split,
            device="cuda",
        )


if __name__ == "__main__":
    main()
