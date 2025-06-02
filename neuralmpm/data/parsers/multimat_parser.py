r"""Parser for the MultiMaterial dataset from GNS."""

from neuralmpm.data.parsers.nmpm_parser import NMPMParser


class MultiMatParser(NMPMParser):
    def __init__(self, path):
        super().__init__(
            path=path,
            dim=2,
            dt=0.0025,
            low_coords=(0.08, 0.08),
            high_coords=(0.92, 0.92),
        )

    def get_original_types(self):
        return [0, 5, 6, 7]

    def get_num_types(self):
        return 4

    def get_sim_length(self):
        return 1000

    def __len__(self):
        raise 1000  # 1000 training trajectories

    def get_splits_sizes(self):
        return {
            "train": 1000,
            "valid": 100,
            "test": 100,
        }
