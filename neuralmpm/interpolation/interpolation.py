# TODO: merge interpolation & this?
# interpolation might be more general actually
from typing import Callable, List

import torch
import torch.nn.functional as F
from torch import Tensor
from torch_cluster import grid_cluster
from torch_scatter import scatter


def find_size(start, end, num_points):
    """Compute the size of a voxel that would divide into num_points.

    Args:
        start: Start of the interval
        end: End of the interval
        num_points: Number of points to divide the interval into

    Returns:
        The size of a voxel that would divide the interval into num_points

    """
    if start == end:
        raise ValueError(
            "Start and end cannot be equal, as it would make size infinite or "
            "undefined."
        )

    delta = start - end
    if delta < 0:
        delta = -delta  # Ensure delta is positive

    lower_bound = delta / num_points
    upper_bound = (
        delta / (num_points - 1) if num_points != 1 else float("inf")
    )  # Avoid division by zero if N is 1

    # We can choose any size in the interval (lower_bound, upper_bound],
    # Choosing the midpoint for simplicity.
    size = (lower_bound + upper_bound) / 2

    return size


def get_voxel_centers(size, start, end):
    """

    Args:
        size:
        start:
        end:

    Returns:

    """
    num_voxels = ((end - start) / size).long() + 1  # grid size

    centers = [
        torch.arange(start[i] + size[i] / 2, end[i] + size[i] / 2, size[i])
        for i in range(len(num_voxels))
    ]
    voxel_centers = torch.meshgrid(*centers, indexing="ij")
    voxel_centers = torch.stack(voxel_centers, dim=-1)

    return voxel_centers


# TODO handle nD smartly
# TODO: Multiple tensors could be pre-computed and made available somewhere, or provided
# as (again) other arguments.
def p2g_cluster_batch(
    grid_coords,
    positions,
    features,
    batch,
    batch_size,
    intp,
    size,
    low,
    high,
    interaction_radius=0.015,
    normalize=False,
) -> List[Tensor]:
    """

    Args:
        grid_coords:
        positions:
        features:
        batch:
        batch_size:
        intp:
        size:
        low:
        high:
        interaction_radius:
        normalize:

    Returns:

    """
    dim = positions[0].shape[-1]

    if dim == 2:
        X, Y, _ = grid_coords.shape
    else:
        X, Y, Z, _ = grid_coords.shape
    N, M = features.shape  # Assume features are batched similarly: (B, N, M)
    B = int(batch.max())
    dim = positions[0].shape[-1]
    # TODO
    # if isinstance(size, float):
    #    s = torch.tensor([size] * dim + [1], device=device)
    # else:
    #    s = torch.tensor(size + [1], device=device)
    s = torch.cat((size, torch.tensor([1])), dim=-1)
    start = torch.cat((low, torch.tensor([0])), dim=-1)
    end = torch.cat((high, torch.tensor([B])), dim=-1)

    vox = grid_cluster(torch.cat((positions, batch[:, None]), dim=-1), s, start, end)
    flat_grid_coords = torch.tile(
        grid_coords.view(-1, 2), (B + 1, 1)
    )  # Shape: (X*Y, 2)

    if vox.min() < 0:
        exit(-1)

    """
    lit_grid = flat_grid_coords[vox]
    weights = torch.vmap(intp, in_dims=(0, 0, None))(
        lit_grid, positions, interaction_radius
    )
    weights = torch.ones_like(weights)
    weighted_features = weights[:, None] * features
    """
    weighted_features = features

    """
    weighted_features = torch.where(
        weights[:, None] > 0, weighted_features, weighted_features.detach()
    )
    """

    feat = scatter(
        weighted_features,
        vox[:, None],
        dim=0,
        dim_size=flat_grid_coords.shape[0],
        reduce="sum",
    )

    if normalize:
        norm = (
            scatter(
                torch.ones(features.shape[0]),
                vox,
                dim=0,
                dim_size=flat_grid_coords.shape[0],
            )
            + 1e-10
        )
        norm = norm[:, None]
    else:
        norm = 1.0

    final_grid = feat / norm

    if dim == 2:
        final_grid = [
            final_grid[i * X * Y : i * X * Y + X * Y].view(X, Y, M)
            if i in batch
            else torch.zeros((X, Y, M))
            for i in range(batch_size)
        ]
    else:
        final_grid = [
            final_grid[i * X * Y * Z : i * X * Y * Z + X * Y * Z].view(X, Y, Z, M)
            if i in batch
            else torch.zeros((X, Y, Z, M))
            for i in range(batch_size)
        ]

    return final_grid


def create_grid_cluster_batch(
    grid_coords: Tensor,  # Duplicate with low/high/size? Pass more smartly
    particle_pos: Tensor,
    particle_vel: Tensor,
    types: Tensor,
    num_types: int,
    interp_fn: Callable,
    low,
    high,
    size,
    interaction_radius,
):
    """

    Args:
        grid_coords:
        particle_pos:
        particle_vel:
        types:
        num_types: Number of different materials. 0 = wall/borders.
        interp_fn:
        low:
        high:
        size:
        interaction_radius:

    Returns:

    """
    with torch.device(grid_coords[0].device):  # TODO: remove this (check usages)
        batch_size = len(particle_pos)
        dim = particle_pos[0].shape[-1]
        grid_sizes = grid_coords.shape[:dim]

        # BxM tensor containing for each batch element, the number of particles
        # for each of the M types.
        num_parts = torch.zeros(batch_size, num_types, dtype=torch.int32)

        for b in range(batch_size):
            for t in range(num_types):
                num_parts[b, t] = (types[b] == torch.tensor(t)).float().sum()

        # Tensor holding the total count over batches for each type.
        total_counts = num_parts.sum(dim=0)

        particle_normalized_densities = [
            torch.zeros(total_counts[t]) for t in range(num_types)
        ]
        particles_batches = [torch.zeros(total_counts[t]) for t in range(num_types)]
        particle_positions = [
            torch.zeros(total_counts[t].item(), dim) for t in range(num_types)
        ]
        particle_velocities = [
            torch.zeros(total_counts[t].item(), dim) for t in range(num_types)
        ]

        # Previous counts for each type. Shape M
        previous_counts = torch.zeros(num_types, dtype=torch.int32)

        densities = torch.zeros(batch_size, *grid_sizes, num_types)
        velocities = torch.zeros(batch_size, *grid_sizes, (num_types - 1) * dim)

        for b in range(batch_size):
            for t in range(num_types):
                num_parts_bt = num_parts[b, t].item()
                if num_parts_bt > 0:
                    particle_normalized_densities[t][
                        previous_counts[t] : previous_counts[t] + num_parts_bt
                    ] = torch.ones((num_parts_bt,)) / num_parts_bt
                    particles_batches[t][
                        previous_counts[t] : previous_counts[t] + num_parts_bt
                    ] = torch.ones((num_parts_bt,)) * b

                    particle_mask = types[b] == t

                    particle_positions[t][
                        previous_counts[t] : previous_counts[t] + num_parts_bt
                    ] = particle_pos[b, particle_mask]

                    if t > 0:
                        particle_velocities[t][
                            previous_counts[t] : previous_counts[t] + num_parts_bt
                        ] = particle_vel[b, particle_mask]

                    previous_counts[t] += num_parts_bt

        for t in range(num_types):
            if total_counts[t] != 0:
                densities[..., t : t + 1] = torch.stack(
                    p2g_cluster_batch(
                        grid_coords,
                        particle_positions[t],
                        particle_normalized_densities[t][:, None],
                        particles_batches[t],
                        batch_size,
                        interp_fn,
                        size,
                        low,
                        high,
                        interaction_radius=interaction_radius,
                        normalize=False if t > 0 else True,
                    )
                )

                if t > 0:
                    velocities[..., dim * (t - 1) : dim * (t - 1) + dim] = torch.stack(
                        p2g_cluster_batch(
                            grid_coords,
                            particle_positions[t],
                            particle_velocities[t],
                            particles_batches[t],
                            batch_size,
                            interp_fn,
                            size,
                            low,
                            high,
                            interaction_radius=interaction_radius,
                            normalize=True,
                        )
                    )

        # TODO: Currently, it's [vels, density_wall, density_fluids]
        # Might put the wall at the end.

        grid = torch.cat([velocities, densities], dim=-1)

    return grid


def g2p(grid, coords, low, high):
    coords = (coords - low) / (high - low)
    # Normalize coords from [0, 1] to [-1, 1]
    coords = coords * 2 - 1

    # Reverse y axis
    # coords = coords * torch.tensor([1, -1], device=coords.device)

    if len(grid.shape) == 3:
        grid = grid.unsqueeze(0)

    # TODO cleaner
    if coords.shape[-1] == 2:
        grid = grid.permute(0, 3, 1, 2)
        coords = coords[:, :, None, :]
    else:
        grid = grid.permute(0, 4, 1, 2, 3)
        coords = coords[:, :, None, None, :]

    return (
        F.grid_sample(
            grid.real,
            coords.real,
            mode="bilinear",
            align_corners=False,
            padding_mode="zeros",
        )
        .squeeze()
        .transpose(-1, -2)
    ).real
