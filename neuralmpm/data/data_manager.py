import os
from glob import glob
from math import ceil
from typing import Union, List, Tuple, Sequence, Callable

import h5py
import torch

import neuralmpm.interpolation.kernels as kernels
from neuralmpm.data.parsers import (
    MonoMatParser,
    MultiMatParser,
    WBCSPHParser,
    WaterDropXLParser,
    Mono3DParser,
    WaterGravityParser,
    Dam2DParser,
)
from neuralmpm.interpolation import (
    create_grid_cluster_batch,
    find_size,
    get_voxel_centers,
)
import neuralmpm.scripts.stats as stats_script

PARSERS = {
    "monomat2d": MonoMatParser,
    "multimat": MultiMatParser,
    "wbc": WBCSPHParser,
    "wdxl": WaterDropXLParser,
    "monomat3d": Mono3DParser,
    "watergravity": WaterGravityParser,
    "dam2d": Dam2DParser,
}


def list_to_padded(
    x: Union[List[torch.Tensor], Tuple[torch.Tensor]],
    pad_size: Union[Sequence[int], None] = None,
    pad_value: float = 0.0,
    equisized: bool = False,
) -> torch.Tensor:
    r"""
    Transforms a list of N tensors each of shape (Si_0, Si_1, ... Si_D)
    into:
    - a single tensor of shape (N, pad_size(0), pad_size(1), ..., pad_size(D))
      if pad_size is provided
    - or a tensor of shape (N, max(Si_0), max(Si_1), ..., max(Si_D)) if pad_size is None.

    Args:
      x: list of Tensors
      pad_size: list(int) specifying the size of the padded tensor.
        If `None` (default), the largest size of each dimension
        is set as the `pad_size`.
      pad_value: float value to be used to fill the padded tensor
      equisized: bool indicating whether the items in x are of equal size
        (sometimes this is known and if provided saves computation)

    Returns:
      x_padded: tensor consisting of padded input tensors stored
        over the newly allocated memory.
    """
    if equisized:
        return torch.stack(x, 0)

    if not all(torch.is_tensor(y) for y in x):
        raise ValueError("All items have to be instances of a torch.Tensor.")

    # we set the common number of dimensions to the maximum
    # of the dimensionalities of the tensors in the list
    element_ndim = max(y.ndim for y in x)

    # replace empty 1D tensors with empty tensors with a correct number of dimensions
    x = [
        (y.new_zeros([0] * element_ndim) if (y.ndim == 1 and y.nelement() == 0) else y)
        for y in x
    ]

    if any(y.ndim != x[0].ndim for y in x):
        raise ValueError("All items have to have the same number of dimensions!")

    if pad_size is None:
        pad_dims = [
            max(y.shape[dim] for y in x if len(y) > 0) for dim in range(x[0].ndim)
        ]
    else:
        if any(len(pad_size) != y.ndim for y in x):
            raise ValueError("Pad size must contain target size for all dimensions.")
        pad_dims = pad_size

    N = len(x)
    x_padded = x[0].new_full((N, *pad_dims), pad_value)
    for i, y in enumerate(x):
        if len(y) > 0:
            slices = (i, *(slice(0, y.shape[dim]) for dim in range(y.ndim)))
            x_padded[slices] = y
    return x_padded


def padded_to_list(
    x: torch.Tensor,
    split_size: Union[Sequence[int], Sequence[Sequence[int]], None] = None,
):
    r"""
    Transforms a padded tensor of shape (N, S_1, S_2, ..., S_D) into a list
    of N tensors of shape:
    - (Si_1, Si_2, ..., Si_D) where (Si_1, Si_2, ..., Si_D) is specified in split_size(i)
    - or (S_1, S_2, ..., S_D) if split_size is None
    - or (Si_1, S_2, ..., S_D) if split_size(i) is an integer.

    Args:
      x: tensor
      split_size: optional 1D or 2D list/tuple of ints defining the number of
        items for each tensor.

    Returns:
      x_list: a list of tensors sharing the memory with the input.
    """
    x_list = list(x.unbind(0))

    if split_size is None:
        return x_list

    N = len(split_size)
    if x.shape[0] != N:
        raise ValueError("Split size must be of same length as inputs first dimension")

    for i in range(N):
        if isinstance(split_size[i], int):
            x_list[i] = x_list[i][: split_size[i]]
        else:
            slices = tuple(slice(0, s) for s in split_size[i])  # pyre-ignore
            x_list[i] = x_list[i][slices]
    return x_list


class SmallDataset(torch.utils.data.Dataset):
    def __init__(self, grids, states, targets, types, grid_targets=None):
        self.init_state = (grids[:, 0], states[:, 0])
        self.init_types = types
        self.timesteps = grids.shape[1]
        self.num = grids.shape[0]
        self.types = (
            types[:, None, :]
            .expand(-1, self.timesteps, -1)
            .reshape(-1, types.shape[-1])
        )

        self.grids = grids.reshape(-1, *grids.shape[2:])
        self.states = states.reshape(-1, *states.shape[2:])
        self.targets = targets.reshape(-1, *targets.shape[2:])

        if grid_targets is not None:
            self.grid_targets = grid_targets.reshape(-1, *grid_targets.shape[2:])

    def __len__(self):
        return self.grids.shape[0]

    def __getitem__(self, idx):
        return (
            self.grids[idx],
            self.states[idx],
            self.targets[idx],
            self.types[idx],
            self.grid_targets[idx],
        )

    def get_sims_for_unroll(self):
        return self.init_state, self.init_types


class DataManager:
    def __init__(
        self,
        path: str,
        dataset_type: str,
        batch_size: int = 64,
        grid_size: list[int] = None,
        steps_per_call: int = 1,
        autoregressive_steps: int = 1,
        sims_in_memory: int = 2,
        interp_fn: Callable = kernels.linear,
        num_valid_sims: int = 2,
        skip_m_steps: bool = False,  # Allows to train over blocks of m steps
        device: str = "cpu",
    ):
        if grid_size is None:
            grid_size = [64, 64]

        self.device = torch.device(device)

        self.batch_size = batch_size

        files = os.path.join(path, "train", "*.h5")
        files = glob(files)
        self.files = sorted(files)

        valid_files = os.path.join(path, "valid", "*.h5")
        valid_files = glob(valid_files)
        self.valid_files = sorted(valid_files)[:num_valid_sims]

        self.dataset_type = dataset_type
        self.parser = PARSERS[dataset_type](path)
        self.dim = self.parser.dim

        self.current_sim_idx = 0
        self.sims_in_memory = sims_in_memory
        self.steps_per_call = steps_per_call
        self.skip_m_steps = skip_m_steps
        self.autoregressive_steps = autoregressive_steps
        self.max_t = (
            self.parser.get_sim_length()
            - self.steps_per_call * self.autoregressive_steps
            - 1
        )

        if self.skip_m_steps:
            self.indices = torch.arange(0, self.max_t, self.steps_per_call)
        else:
            self.indices = torch.arange(0, self.max_t)

        self.target_indices = self.indices[:, None] + torch.arange(
            1, self.steps_per_call * self.autoregressive_steps + 1
        )

        self.iters_per_epoch = ceil(
            self.sims_in_memory * self.indices.shape[0] / self.batch_size
        )

        self.sims = None
        self.types = None
        self.gravities = None
        self.grids = None
        self.grid_size = grid_size
        self.gmean = None
        self.gstd = None

        # TODO: handle non-square volumes
        self.size = [
            find_size(low, high, grid_axis_size)
            for low, high, grid_axis_size in zip(
                self.parser.low_bound, self.parser.high_bound, grid_size
            )
        ]
        self.size_tensor = torch.tensor(self.size)

        grid_coords = get_voxel_centers(
            self.size_tensor,
            self.parser.low_bound,
            self.parser.high_bound,
        )

        self.interp_fn = interp_fn
        self.interaction_radius = self.parser.interaction_radius

        self.grid_coords = grid_coords.to(self.device)

    def get_valid_sims(self):
        # TODO: Clean, so many places where we repeat this
        #  create_grid_cluster_batch loop...
        sims = []
        types = []
        grids = []

        for f in self.valid_files:
            # sim, type_, grav = self.load_simulation(f, self.material)
            # TODO remove this and use IDs everywhere (list of existing ids)
            sim_id = int(f.split("_")[-1].split(".")[0])
            sim_data = self.parser.parse(sim_id, "valid")
            sim = sim_data["sim"]
            type_ = sim_data["types"]
            grav = sim_data.get("grav", None)  # TODO, handle any extra data

            sim = sim.to(self.device)
            type_ = type_.to(self.device)
            if grav is not None:
                grav = grav.to(self.device)

            sims.append(sim)
            types.append(type_)

            grid = create_grid_cluster_batch(
                self.grid_coords,
                torch.clamp(  # TODO Clamp function in the parser?
                    sim[..., : self.dim],
                    min=self.parser.low_bound.to(self.device),
                    max=self.parser.high_bound.to(self.device),
                ),
                # TODO handle non-square volumes
                sim[..., self.dim :],
                torch.tile(type_[None, :], (sim.shape[0], 1)),
                self.parser.get_num_types(),
                self.interp_fn,
                low=self.parser.low_bound.to(self.device),
                high=self.parser.high_bound.to(self.device),
                size=self.size_tensor.to(self.device),
                interaction_radius=self.interaction_radius,
            )

            if grav is not None:
                grid = torch.cat(
                    (grid, torch.tile(grav[None, None, None], (*grid.shape[:-1], 1))),
                    dim=-1,
                )

            grids.append(grid[0])

        grids = torch.stack(grids)
        states = list_to_padded([s[0] for s in sims])
        types = list_to_padded(types)

        return (grids, states), types, sims

    def load_sim_buffer(self):
        if self.current_sim_idx + self.sims_in_memory >= len(self.files):
            self.current_sim_idx = 0

        sims = []
        types = []
        gravities = []

        for idx in range(
            self.current_sim_idx, self.current_sim_idx + self.sims_in_memory
        ):
            # idx = 0
            sim = self.files[idx]
            # sim, type_, grav = self.load_simulation(f, self.material)
            # TODO remove this and use IDs everywhere (list of existing ids)
            sim_id = int(sim.split("_")[-1].split(".")[0])
            # sim, type_, grav = self.parser.parse(sim_id, "train")
            sim_data = self.parser.parse(sim_id, "train")

            sim = sim_data["sim"]
            type_ = sim_data["types"]
            grav = sim_data.get("grav", None)  # TODO, handle any extra data

            sims.append(sim)
            types.append(type_)
            gravities.append(grav)

        self.sims = sims
        self.types = types
        self.gravities = gravities
        self.current_sim_idx += self.sims_in_memory

    def build_grids(self):
        """Build the grids over the current buffer of simulations.
        Returns:

        """
        sims = self.sims
        types = self.types
        gravities = self.gravities

        grids = []

        for (
            sim,
            typ,
            grav,
        ) in zip(sims, types, gravities):
            if grav is not None:
                grav = grav.to(self.device)
            sim, typ = sim.to(self.device), typ.to(self.device)
            # TODO: clamp only for WBC

            num_types = self.parser.get_num_types()

            with torch.device(self.device):
                grid = create_grid_cluster_batch(
                    self.grid_coords,
                    torch.clamp(
                        sim[..., : self.dim].to(self.device),
                        min=self.parser.low_bound.to(self.device),
                        max=self.parser.high_bound.to(self.device),
                    ),
                    # TODO handle non-square volumes
                    sim[..., self.dim :],
                    torch.tile(typ[None, :], (sim.shape[0], 1)),
                    num_types=num_types,
                    interp_fn=self.interp_fn,
                    low=self.parser.low_bound.to(self.device),
                    high=self.parser.high_bound.to(self.device),
                    size=self.size_tensor.to(self.device),
                    interaction_radius=self.interaction_radius,
                )
                if grav is not None:
                    grid = torch.cat(
                        (
                            grid,
                            torch.tile(grav[None, None, None], (*grid.shape[:-1], 1)),
                        ),
                        axis=-1,
                    )

                grids.append(grid.cpu())

        self.grids = grids

        axes = tuple(range(self.dim + 1))
        self.gmean = torch.mean(torch.cat(grids, axis=0), dim=axes, keepdim=True)
        self.gstd = torch.std(torch.cat(grids, axis=0), dim=axes, keepdim=True)
        # Avoid division by zero
        self.gstd = torch.where(self.gstd == 0, torch.ones_like(self.gstd), self.gstd)

        self.gmean = self.gmean.to(self.device)
        self.gstd = self.gstd.to(self.device)

        # TODO: this
        if self.gmean.shape[-1] == 6:
            self.gmean[..., -2:] = 0
            self.gstd[..., -2:] = 1

    def load_buffer(self):
        self.load_sim_buffer()
        self.build_grids()

        state_list = []
        target_list = []
        grid_list = []
        grid_target_list = []

        for grid, sim in zip(self.grids, self.sims):
            states = sim[self.indices]
            targets = sim[self.target_indices]
            grids = grid[self.indices]
            grid_targets = grid[self.target_indices]

            state_list.append(states)
            target_list.append(targets)
            grid_list.append(grids)
            grid_target_list.append(grid_targets)

        states = list_to_padded(state_list)
        targets = list_to_padded(target_list)
        # states = torch.stack(state_list)
        # targets = torch.stack(target_list)

        grids = torch.stack(grid_list)
        grid_targets = torch.stack(grid_target_list)

        types = list_to_padded(self.types)
        # types = torch.stack(self.types)

        dataset = SmallDataset(
            grids=grids,
            states=states,
            targets=targets,
            grid_targets=grid_targets,
            types=types,
        )

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            # generator=torch.Generator(device="cuda" if
            # torch.cuda.is_available() else "cpu"),
        )
        return dataloader

    def get_stats(self, split="train"):
        grid_size_str = "x".join([str(g) for g in self.grid_size])
        stats_file_path = os.path.join(
            self.parser.path, "stats", f"{split}_{grid_size_str}.h5"
        )

        if not os.path.exists(stats_file_path):
            print(
                f"Stats file not found at {stats_file_path}. Generating them, this might take time, please pre-generate them on a cluster."
            )
            print("creating on device", self.device)
            stats_script.compute_dataset_mstd(
                str(self.parser.path),
                self.dataset_type,
                self.grid_size,
                split="train",
                device=self.device,
            )

        with h5py.File(stats_file_path, "r") as f:
            mean = torch.tensor(f["mean"])
            std = torch.tensor(f["std"])
            return mean, std
