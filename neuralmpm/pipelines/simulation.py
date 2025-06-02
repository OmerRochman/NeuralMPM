import torch

import neuralmpm.interpolation as interp
from neuralmpm.interpolation import kernels


def get_step_fn(coords, interp_fct, num_types, low, high, size, interaction_radius):
    dim = coords.shape[-1]

    def euler_step(grid_velocities, old_pos, types):
        ys = []
        pos = old_pos[..., :dim]
        #                   0  1  2  3    4         5
        # Grid velocities: (B, X, Y, Z, steps, velocities)

        # Permute
        # TODO better way
        if coords.shape[-1] == 2:
            grid_velocities = grid_velocities.permute(3, 0, 1, 2, 4)
        else:
            grid_velocities = grid_velocities.permute(4, 0, 1, 2, 3, 5)

        pred_step = 1
        for x in grid_velocities:
            interpolated_vel = interp.g2p(x, pos, low, high)
            if len(interpolated_vel.shape) == 2:
                interpolated_vel = interpolated_vel.unsqueeze(0)

            masks = [types == t for t in range(1, num_types)]
            vels = [
                interpolated_vel[..., i * dim : (i + 1) * dim]
                for i in range(num_types - 1)
            ]

            interpolated_vel = torch.stack(
                [mask[..., None] * vel for mask, vel in zip(masks, vels)]
            ).sum(0)

            # TODO
            # grav_vector = torch.tensor([0.0, -9.806]).to(interpolated_vel.device)
            # interpolated_vel = interpolated_vel + grav_vector * 0.0025**2 * 0.5 * pred_step
            pos = pos + interpolated_vel

            pred_step += 1
            ys.append((pos, interpolated_vel))

        # TODO
        # 2D
        # if coords.shape[-1] == 2:
        particles = torch.stack([y[0] for y in ys]).permute(1, 0, 2, 3)
        vel = torch.stack([y[1] for y in ys]).permute(1, 0, 2, 3)
        # else:  # 3D
        #    particles = torch.stack([y[0] for y in ys]).permute(1, 0, 2, 3, 4)
        #    vel = torch.stack([y[1] for y in ys]).permute(1, 0, 2, 3, 4)

        particles = torch.where(
            types[:, None, :, None] > 0.0, particles, old_pos[:, None, :, :dim]
        )
        vel = torch.where(types[:, None, :, None] > 0.0, vel, 0.0)

        particles = torch.clamp(particles, min=low, max=high)

        next_input = interp.create_grid_cluster_batch(
            coords,
            particles[:, -1],
            vel[:, -1],
            types,
            num_types,
            interp_fct,
            low,
            high,
            size,  # TODO
            interaction_radius,
        )

        particles = torch.cat((particles, vel), axis=-1)

        return particles, next_input

    return euler_step


@torch.no_grad()
def unroll(
    model,
    init_state,
    coords,
    num_calls,
    gmean,
    gstd,
    types,
    num_types,
    low,
    high,
    size,
    interaction_radius=0.015,
    interp_fn=kernels.linear,
):
    dim = coords.shape[-1]

    # TODO in args
    num_fluids = num_types - 1

    euler_step = get_step_fn(
        coords,
        interp_fn,
        num_types,
        low,
        high,
        size,
        interaction_radius=interaction_radius,
    )

    def step(state, i):
        (state, old_particles) = state
        state = (state - gmean) / gstd
        grid_preds = model(state)
        grid_velocities = (
            grid_preds * gstd[None, ..., : num_fluids * dim]
            + gmean[None, ..., : num_fluids * dim]
        )
        full_particles, next_input = euler_step(grid_velocities, old_particles, types)
        new_particles = full_particles[:, -1]

        return (next_input, new_particles), (
            grid_preds,
            full_particles,
            next_input,
            new_particles,
        )

    carry = init_state
    ys = []
    for x in torch.arange(num_calls):
        carry, y = step(carry, x)
        # TODO: this
        if init_state[0].shape[-1] == 6:
            grid = torch.cat((carry[0], init_state[0][..., -2:]), axis=-1)
            carry = (grid, carry[1])

        ys.append(y)

    full_grids = torch.stack([y[0] for y in ys])
    full_particles = torch.stack([y[1] for y in ys]).permute(1, 0, 2, 3, 4)
    # TODO cleaner
    if coords.shape[-1] == 2:
        input_grids = torch.stack([y[2] for y in ys]).permute(1, 0, 2, 3, 4)
        full_grids = full_grids.permute(1, 0, -2, 2, 3, -1)
    else:
        input_grids = torch.stack([y[2] for y in ys]).permute(1, 0, 2, 3, 4, 5)
        full_grids = full_grids.permute(1, 0, -2, 2, 3, 4, -1)

    bsize = init_state[0].shape[0]
    full_grids = full_grids.reshape(bsize, -1, *full_grids.shape[-3:])
    full_particles = full_particles.reshape(bsize, -1, *full_particles.shape[-2:])

    return full_grids, full_particles, input_grids
