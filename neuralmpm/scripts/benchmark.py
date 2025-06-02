import os
import json
import torch

from neuralmpm.pipelines.evaluation import load_model
from neuralmpm.interpolation import interpolation as interp
from neuralmpm.pipelines.simulation import get_step_fn
from neuralmpm.interpolation import kernels


@torch.no_grad()
def benchmark(
    model, steps_per_call, num_particles, total_steps=100, batch_size=1, grid_sizes=None
):
    num_types = 2
    dim = 2

    if grid_sizes is None:
        grid_sizes = [64] * dim

    low = torch.tensor([0.0] * dim).cuda()
    high = torch.tensor([1.0] * dim).cuda()
    sizes = [
        interp.find_size(low, high, grid_axis_size)
        for low, high, grid_axis_size in zip(low, high, grid_sizes)
    ]
    size_tensor = torch.tensor(sizes).cuda()
    grid_coords = interp.get_voxel_centers(size_tensor, low, high).cuda()

    euler_step = get_step_fn(
        grid_coords,
        kernels.linear,
        num_types,
        low,
        high,
        size_tensor,
        interaction_radius=0.015,
    )

    # Particles
    init_particles = torch.rand((batch_size, num_particles, 2 * dim))
    init_particles[..., dim:] *= 10**-3
    init_particles = init_particles.cuda()

    # Types
    num_wall = num_particles // 10
    types = torch.ones((batch_size, num_particles), dtype=torch.int64)
    types[:, :num_wall] = 0
    types = types.cuda()

    device = torch.device("cuda")
    # Grid
    init_grid = interp.create_grid_cluster_batch(
        grid_coords,
        init_particles[..., :dim],
        init_particles[..., dim:],
        types,
        num_types,
        kernels.linear,
        low=low.to(device),
        high=high.to(device),
        size=size_tensor.to(device),
        interaction_radius=0.015,
    ).cuda()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    times = []
    state = (init_grid, init_particles)
    for _ in range(total_steps):
        start.record()

        state, old_particles = state
        grid_preds = model(state)
        grid_velocities = grid_preds
        with torch.device("cuda"):
            full_particles, next_input = euler_step(
                grid_velocities, old_particles, types
            )
        new_particles = full_particles[:, -1]

        state = (next_input, new_particles)

        end.record()
        torch.cuda.synchronize()
        elapsed_time = start.elapsed_time(end)
        times.append(elapsed_time)

    # Take mean of last 10 elements of times
    mean_time = sum(times[-10:]) / 10
    mean_fps = steps_per_call / (mean_time / 1000)

    return mean_time, mean_fps


def main():
    path = "outputs/reprod_check/waterramps_2_3qmsjc31"
    with open(os.path.join(path, "config.json"), "r") as f:
        config_dict = json.load(f)

    model = load_model(path, config_dict)
    model.cuda()

    perf_dict = {}

    # for N in [1e2, 5e2, 1e3, 5e3, 1e4, 5e4, 1e5, 5e5, 1e6, 5e6, 1e7, 3e7]:
    for N in [3.75e7]:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        N = int(N)
        time, fps = benchmark(model, config_dict["steps_per_call"], N)
        peak_memory = torch.cuda.max_memory_allocated() / 1024**2

        print(f"{N} particles: {time:.1f}ms (~{fps:.1f} FPS), used {peak_memory:.1f}MB")
        perf_dict[N] = {"time": time, "fps": fps, "mem": peak_memory}

    with open(os.path.join(path, "perf.json"), "w") as f:
        json.dump(perf_dict, f)


if __name__ == "__main__":
    main()
