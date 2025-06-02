from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
import torch
import ott


def mse(pred, true, types=None):
    mask = torch.ones(true.shape[1], dtype=torch.bool)
    if types is not None:
        mask = types != 0

    # TODO
    dim = pred.shape[-1]

    t = true.shape[0] - 1
    err = (pred[:t, mask, :dim] - true[1:, mask, :dim]) ** 2
    err = err.mean(dim=(1, 2))
    return err


def ae(pred, true, types=None):
    mask = torch.ones(true.shape[1], dtype=torch.bool)
    if types is not None:
        mask = types != 0

    # TODO
    dim = pred.shape[-1]

    t = true.shape[0] - 1
    err = torch.abs(pred[:t, mask, :dim] - true[1:, mask, :dim])
    err = err.mean(dim=(1, 2))
    return err


def mse_np(pred, true, types=None):
    mask = np.ones(true.shape[1], dtype=bool)
    if types is not None:
        mask = types != 0

    # TODO
    dim = pred.shape[-1]

    t = true.shape[0] - 1
    err = (pred[:t, mask, :dim] - true[1:, mask, :dim]) ** 2
    err = err.mean(axis=(1, 2))
    return err


def ae_np(pred, true, types=None):
    mask = np.ones(true.shape[1], dtype=bool)
    if types is not None:
        mask = types != 0

    # TODO
    dim = pred.shape[-1]

    t = true.shape[0] - 1
    err = np.abs(pred[:t, mask, :dim] - true[1:, mask, :dim])
    err = err.mean(axis=(1, 2))
    return err


@partial(jnp.vectorize, signature="(N,D),(N,D)->()")
def _emd_fn(pred, ground):
    geom = ott.geometry.pointcloud.PointCloud(
        pred, ground, cost_fn=ott.geometry.costs.Euclidean()
    )
    prob = ott.problems.linear.linear_problem.LinearProblem(geom)
    solve_fn = ott.solvers.linear.sinkhorn.Sinkhorn()
    out = solve_fn(prob)
    return out.reg_ot_cost


def emd(pred, true, num_chunks=4):
    """
    Compute the Earth Mover's Distance between two sets of points.

    Args:
        pred (jnp.ndarray): Predicted points of shape (T, N, D).
        true (jnp.ndarray): Ground truth points of shape (T, N, D).
        num_chunks (int): Number of chunks to split the data into.

    Returns:
        np.ndarray: EMD values of shape (T,).
    """
    if not isinstance(pred, jnp.ndarray):
        pred = jnp.array(pred)
        true = jnp.array(true)

    emd_fn = jax.jit(_emd_fn)
    chunk_size = len(pred) // num_chunks
    emd = [
        emd_fn(
            pred[i * chunk_size : (i + 1) * chunk_size],
            true[i * chunk_size : (i + 1) * chunk_size],
        )
        for i in range(num_chunks)
    ]
    emd = jnp.concatenate(emd, axis=0)

    return np.array(emd)
