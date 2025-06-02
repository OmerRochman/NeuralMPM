r"""Base parser for NeuralMPM-formatted 2D datasets."""

import h5py
import torch
import numpy as np

from neuralmpm.data.parsers.base_parser import Parser


class NMPMParser(Parser):
    def __init__(self, path, dim, dt, low_coords, high_coords):
        super().__init__(
            path=path,
            dim=dim,
            dt=dt,
            low_coords=low_coords,
            high_coords=high_coords,
        )

    def parse(self, sim_id, split="train"):
        sim_path = self.path / split / f"sim_{sim_id}.h5"
        types = None
        metadata = {}
        with h5py.File(sim_path, "r") as f:
            particles = f["particles"][()]
            boundary = f["boundary"][()]
            if "types" in f:
                types = f["types"][()]
                if len(np.unique(types)) == 1:
                    types = None
            for key in f.keys():
                if key not in ["particles", "boundary", "types"]:
                    try:
                        metadata[key] = f[key][()]
                    except Exception:
                        print(f"Could not read metadata {key}")

        particles = torch.tensor(particles[..., : self.dim], dtype=torch.float32)
        boundary = torch.tensor(boundary[..., : self.dim], dtype=torch.float32)
        sim = torch.cat([boundary, particles], dim=1)
        if types is None or self.get_num_types() == 2:
            types = torch.cat(
                [
                    torch.zeros(boundary.shape[1]),
                    torch.ones(particles.shape[1]),
                ],
                dim=0,
            )
        else:
            types = torch.tensor(types)
            if types.shape[0] != sim.shape[1]:
                types = torch.cat(
                    [
                        torch.zeros(boundary.shape[1]),
                        types,
                    ],
                    dim=0,
                )
            types = types.long()
            unique_types = torch.tensor(sorted(self.get_original_types()))
            new_types = torch.arange(len(unique_types))

            for i, t in enumerate(unique_types):
                types[types == t] = new_types[i]

            assert types.max() < len(new_types), (
                f"Types are not consistent! {types.max()} >= {len(new_types)}"
            )

        vel = sim[1:] - sim[:-1]
        sim = torch.cat((sim[1:], vel), axis=-1)

        for key, meta in metadata.items():
            metadata[key] = torch.tensor(meta, dtype=torch.float32)

        sim_data = {
            "sim": sim,
            "types": types,
        }

        sim_data.update(metadata)

        return sim_data
