import argparse
import os
from glob import glob

import h5py
import torch
from natsort import natsorted
from tqdm import tqdm

import neuralmpm.interpolation.interpolation as interp
import neuralmpm.pipelines.simulation as simulate
from neuralmpm.data.data_manager import (
    find_size,
    get_voxel_centers,
    list_to_padded,
    DataManager,
)
from neuralmpm.interpolation import kernels
from neuralmpm.pipelines.evaluation import load_config_and_model

torch.set_float32_matmul_precision("high")


# TODO, probably should be removed and replaced with DataMangaer
# Overall, need to clean a bit the parsers, and have a nice way to easily
# get the sims from each split. I'd like not to hardcore the "sim_{id}.h5"
# format so without using the IDs only.
def get_rollout_sims(files, parser):
    sims = []
    types = []
    gravity = []

    for sim_id in range(parser.get_splits_sizes()["test"]):
        sim_data = parser.parse(sim_id, split="test")

        sim = sim_data["sim"]
        type_ = sim_data["types"]
        grav = sim_data.get("grav", None)

        sims.append(sim)
        types.append(type_)
        gravity.append(grav)

    return sims, types, gravity


def rollout_ic(model_path, ic_path, num_steps=500, ckpt=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device = 'cpu'

    config, model = load_config_and_model(model_path, ckpt)

    if device == "cuda":
        model.cuda()

    data_manager = DataManager(
        config["data"],
        config["dataset_type"],  # TODO
        batch_size=1,
        grid_size=config["grid_size"],
        steps_per_call=config["steps_per_call"],
        autoregressive_steps=config["autoregressive_steps"],
        sims_in_memory=config["sims_in_memory"],
        device=device,
    )
    parser = data_manager.parser
    gmean, gstd = data_manager.get_stats()
    gmean = gmean.to(device)
    gstd = gstd.to(device)
    start = parser.low_bound.to(device)
    end = parser.high_bound.to(device)

    # TODO
    if "grav" in data_manager.parser.parse(0).keys():
        gmean = torch.cat([gmean, torch.zeros(2, device=gmean.device)], dim=0)
        gstd = torch.cat([gstd, torch.ones(2, device=gstd.device)], dim=0)

    print("gmean", gmean.shape)

    with h5py.File(ic_path, "r") as f:
        print(f.keys())
        boundary = torch.tensor(f["boundary"][:]).float()
        particles = torch.tensor(f["particles"][:]).float()

        if boundary.shape[-1] < particles.shape[-1]:
            boundary = torch.cat([boundary, torch.zeros_like(boundary)], dim=-1)

        print("particles", particles.shape)
        print("boundary", boundary.shape)

        sim = torch.cat([particles, boundary], dim=-2)
        print("sim", sim.shape)

        if "types" in f.keys():
            types = torch.tensor(f["types"][:]).float()
        else:
            types = torch.cat(
                [
                    torch.ones(particles.shape[-2]),
                    torch.zeros(boundary.shape[-2]),
                ]
            )

        if types.shape[0] < sim.shape[1]:
            types = torch.cat(
                [types, torch.zeros(sim.shape[1] - types.shape[0], device=types.device)]
            )

    if len(sim.shape) == 2:
        sim = sim[None]
    elif len(sim.shape) == 3:
        sim = sim[:1]

    start = torch.min(sim[0, ..., : parser.dim], dim=0).values
    end = torch.max(sim[0, ..., : parser.dim], dim=0).values
    start, end = start.to(device), end.to(device)
    grid_size = config["grid_size"]
    grid_size = [64, 64]

    print(start, end)
    print(sim.shape, types.shape)
    print("T", types)
    print("T", types.shape)

    sizes = [
        find_size(low, high, grid_axis_size)
        for low, high, grid_axis_size in zip(start, end, grid_size)
    ]
    size_tensor = torch.tensor(sizes).to(device)
    grid_coords = get_voxel_centers(size_tensor, start, end).to(device)

    grids = []

    sim = sim.to(device)
    types = types.to(device)

    pos = sim[..., : parser.dim]
    vel = sim[..., parser.dim :]
    if vel.shape[-1] == 0:
        print("No velocity!")
        vel = (
            torch.zeros_like(pos) - torch.tensor([1.5, -1.0], device=device) * parser.dt
        )
    else:
        vel *= parser.dt
    types = types[None]

    print(pos.shape, vel.shape, types.shape)

    with torch.device(device):
        grid = interp.create_grid_cluster_batch(
            grid_coords,
            pos,
            vel,
            types,
            parser.get_num_types(),
            kernels.linear,
            low=start.to(device),
            high=end.to(device),
            size=size_tensor.to(device),
            interaction_radius=0.015,
        )

        import matplotlib.pyplot as plt

        for i in range(grid.shape[-1]):
            plt.imshow(grid[0, ..., i].cpu())
            plt.savefig(f"grid_{i}.png")
            print(f"Saved grid {i}")

        print("mean", grid.mean())

        """
        if grav is not None:
            grav = grav.to(device)
            grid = torch.cat([
                grid,
                    torch.tile(grav[None, None, None], (*grid.shape[:-1], 1))
            ], dim=-1)
        """

        grid = grid.cpu()

        grids += [grid]

    print(num_steps)

    grids = torch.stack(grids)

    with torch.no_grad():
        num_calls = num_steps // config["steps_per_call"] + 1
        # num_calls = 1500 // config["steps_per_call"] + 1

        fps = []
        init_state = (grids[0], sim)
        batch_types = types

        if device == "cuda":
            init_state = (init_state[0].cuda(), init_state[1].cuda())
            batch_types = batch_types.cuda()
            grid_coords = grid_coords.cuda()
            timer_start = torch.cuda.Event(enable_timing=True)
            timer_end = torch.cuda.Event(enable_timing=True)
            timer_start.record()

        print("ST", size_tensor)

        with torch.device(device):
            pred = simulate.unroll(
                model,
                init_state,
                grid_coords,
                num_calls,
                gmean,
                gstd,
                batch_types,
                parser.get_num_types(),
                start.to(device),
                end.to(device),
                size_tensor.to(device),
                interaction_radius=0.015,
                interp_fn=None,
            )[1][0]

        if device == "cuda":
            timer_end.record()
            torch.cuda.synchronize()
            elapsed_time = timer_start.elapsed_time(timer_end)
            fps.append(pred.shape[1] / (elapsed_time * 1e-3))

        if fps:
            print(f"Average FPS: {sum(fps) / len(fps):.1f}")

        save_path = f"{model_path}/rollouts/{ic_path.split('/')[-1].split('.')[0]}"

        os.makedirs(f"{save_path}", exist_ok=True)

        types = types[0]

        print("wtf", types.shape[0])

        with h5py.File(f"{save_path}/rollout_0.h5", "w") as f:
            f.create_dataset(
                "predicted_rollout", data=pred[:num_steps, : types.shape[0]].cpu()
            )
            print(pred.shape)
            f.create_dataset("types", data=types.cpu())


def rollout(model_path, data_path=None, batch_size=10, ckpt=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Rolling out on", device)

    config, model = load_config_and_model(model_path, ckpt)

    if torch.cuda.is_available():
        model.cuda()

    if data_path is None:
        data_path = config["data"]

    data_manager = DataManager(
        data_path,
        config["dataset_type"],  # TODO
        batch_size=batch_size,
        grid_size=config["grid_size"],
        steps_per_call=config["steps_per_call"],
        autoregressive_steps=config["autoregressive_steps"],
        sims_in_memory=config["sims_in_memory"],
        device=device,
    )
    parser = data_manager.parser

    gmean, gstd = data_manager.get_stats()
    if torch.cuda.is_available():
        gmean = gmean.cuda()
        gstd = gstd.cuda()

    # TODO
    if "grav" in data_manager.parser.parse(0).keys():
        gmean = torch.cat([gmean, torch.zeros(2, device=gmean.device)], dim=0)
        gstd = torch.cat([gstd, torch.ones(2, device=gstd.device)], dim=0)

    files = os.path.join(data_path, "test", "*.h5")

    files = glob(files)
    files = natsorted(files)

    sims, types, grav = get_rollout_sims(files, parser)
    start = parser.low_bound.to(device)
    end = parser.high_bound.to(device)

    sizes = [
        find_size(low, high, grid_axis_size)
        for low, high, grid_axis_size in zip(start, end, config["grid_size"])
    ]
    size_tensor = torch.tensor(sizes).to(device)
    print("devices", size_tensor.device, start.device, end.device)
    grid_coords = get_voxel_centers(size_tensor, start, end)

    if torch.cuda.is_available():
        size_tensor = size_tensor.cuda()
        grid_coords = grid_coords.cuda()

    padded_s0 = list_to_padded([s[0] for s in sims])
    padded_types = list_to_padded(types)

    grids = []

    for sim, typ, grav in zip(sims, types, grav):
        sim = sim.to(device)
        typ = typ.to(device)

        with torch.device(device):
            grid = interp.create_grid_cluster_batch(
                grid_coords,
                sim[:1, ..., : parser.dim],
                sim[:1, ..., parser.dim :],
                torch.tile(typ[None, :], (1, 1)),
                parser.get_num_types(),
                kernels.linear,
                low=parser.low_bound.to(device),
                high=parser.high_bound.to(device),
                size=size_tensor.to(device),
                interaction_radius=0.015,
            )

            if grav is not None:
                grav = grav.to("cuda")
                grid = torch.cat(
                    [grid, torch.tile(grav[None, None, None], (*grid.shape[:-1], 1))],
                    dim=-1,
                )

            grid = grid.cpu()

            grids += [grid]

    grids = torch.stack(grids)

    with torch.no_grad():
        num_calls = sims[0].shape[0] // config["steps_per_call"] + 1
        # num_calls = 1500 // config["steps_per_call"] + 1

        trajectories = []
        fps = []
        for b in tqdm(range(0, len(sims), batch_size)):
            nxt = min(b + batch_size, len(sims))
            init_state = (grids[b:nxt, 0], padded_s0[b:nxt])
            batch_types = padded_types[b:nxt]

            if torch.cuda.is_available():
                init_state = (init_state[0].cuda(), init_state[1].cuda())
                batch_types = batch_types.cuda()
                grid_coords = grid_coords.cuda()
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()

            with torch.device(device):
                _, traj, new_grids = simulate.unroll(
                    model,
                    init_state,
                    grid_coords,
                    num_calls,
                    gmean,
                    gstd,
                    batch_types,
                    parser.get_num_types(),
                    parser.low_bound.to(device),
                    parser.high_bound.to(device),
                    size_tensor.to(device),
                    interaction_radius=0.015,
                    interp_fn=None,
                )

            if torch.cuda.is_available() and len(traj) == batch_size:
                end.record()
                torch.cuda.synchronize()
                elapsed_time = start.elapsed_time(end)
                fps.append(traj.shape[1] / (elapsed_time * 1e-3))
            trajectories.extend([traj[t].cpu() for t in range(traj.shape[0])])

    if fps:
        print(f"Average FPS: {sum(fps) / len(fps):.1f}")

    save_path = f"{model_path}/rollouts/{data_path.split('/')[-1]}"

    os.makedirs(f"{save_path}", exist_ok=True)

    for i, (pred, true, typ) in enumerate(zip(trajectories, sims, types)):
        t = true.shape[0] - 1
        with h5py.File(f"{save_path}/rollout_{i}.h5", "w") as f:
            f.create_dataset("ground_truth_rollout", data=true[1:].cpu())
            f.create_dataset("predicted_rollout", data=pred[:t, : typ.shape[0]].cpu())
            f.create_dataset("types", data=typ.cpu())


def main():
    parser = argparse.ArgumentParser("Neural MPM")
    parser.add_argument("--run", "-r", required=True)
    parser.add_argument("--data-path", "-d", type=str, default=None)
    parser.add_argument("--point-cloud", "-pc", type=str)
    parser.add_argument("--num-steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=8)
    # TODO Other params: grid size, ...
    parser.add_argument("--ckpt", type=str, default=None)
    args = parser.parse_args()

    if args.point_cloud is not None:
        rollout_ic(args.run, args.point_cloud, num_steps=args.num_steps, ckpt=args.ckpt)
    else:
        rollout(args.run, args.data_path, args.batch_size, args.ckpt)


if __name__ == "__main__":
    main()
