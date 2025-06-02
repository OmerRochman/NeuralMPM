from typing import Callable

import torch


# TODO: Same arg for all losses for that, maybe a dict.
def particle_position_mse_loss(
    model,
    grids,
    target_grids,
    state,
    targets,
    gmean,
    gstd,
    types,
    num_types,
    step_fn: Callable,
    steps_per_call: int,
    grid_size: int = 64,
    dim: int = 2,
    autoregressive_steps: int = 1,
    debug: int = -1,
):
    loss = 0
    prev_pos = state[..., :dim]
    total_steps = autoregressive_steps * steps_per_call

    norm = torch.count_nonzero(types) * steps_per_call * dim

    # TODO: this
    hack = grids.shape[-1] == 6
    if hack:
        grav = grids[..., -2:]

    num_types = num_types - 1

    for i in range(0, total_steps, steps_per_call):
        points = (grids - gmean) / gstd
        preds = model(points)
        preds = (
            preds * gstd[None, ..., : num_types * dim]
            + gmean[None, ..., : num_types * dim]
        )
        pred_pos, grids = step_fn(
            preds,
            prev_pos,
            types,
        )

        # TODO: this
        if hack:
            grids = torch.cat((grids, grav), dim=-1)

        real_loss = (
            targets[:, i : i + steps_per_call, ..., :dim] - pred_pos[..., :dim]
        ) ** 2
        real_loss = torch.where(types[:, None, :, None] > 0.0, real_loss, 0.0)

        real_loss = real_loss.sum() * 64**2 / norm
        loss += real_loss
        prev_pos = pred_pos[:, -1, ..., :dim]

    return loss


def velocity_loss():
    raise NotImplementedError


def emd_loss():
    raise NotImplementedError
