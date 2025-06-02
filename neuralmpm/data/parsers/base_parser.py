from pathlib import Path

import torch

from neuralmpm.pipelines import visualisation as viz


class Parser:
    def __init__(
        self,
        path,
        dim,
        dt,
        low_coords,
        high_coords,
        interaction_radius=0.015,
    ):
        self.path = Path(path)
        self.dim = dim
        self.dt = dt
        self.interaction_radius = interaction_radius

        self.low_bound = torch.tensor(low_coords)
        self.high_bound = torch.tensor(high_coords)

    def parse(self, sim_id, split="train") -> dict:
        raise NotImplementedError

    def get_sim_length(self):
        raise NotImplementedError

    def get_num_types(self):
        raise NotImplementedError

    def get_original_types(self):
        """
        Returns a list of the original type IDs for the different materials.
        Might be different than [0, 1, ..., num_types-1].
        """
        return list(range(self.get_num_types()))

    def get_colors(self):
        """
        Returns a list of matplotlib colors for each material type.
        Must be of the same size as the number of materials.

        Returns:
            list: List of colors (rgb(a) tuples, hex, or strings)
        """
        return viz.DEFAULT_COLORS[: self.get_num_types()]

    def get_num_channels(self):
        return self.get_num_types() * (1 + self.dim) - self.dim

    def __len__(self):
        raise self.get_splits_sizes()["train"]

    def get_splits_sizes(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        return self.parse(idx, split="train")

    def get_bounds(self):
        return self.low_bound, self.high_bound

    def get_dim(self):
        return self.dim
