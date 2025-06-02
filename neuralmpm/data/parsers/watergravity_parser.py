r"""Parser for the VariableGravity dataset from us (NeuralMPM)."""

from neuralmpm.data.parsers.nmpm_parser import NMPMParser


class WaterGravityParser(NMPMParser):
    def __init__(self, path):
        super().__init__(
            path=path,
            dim=2,
            dt=0.0025,
            low_coords=(-0.0025, -0.0025),
            high_coords=(1.0025, 1.0025),
        )

    def get_num_channels(self):
        return 6

    def get_sim_length(self):
        return 1000

    def get_num_types(self):
        return 2

    def get_splits_sizes(self):
        return {
            "train": 1000,
            "valid": 100,
            "test": 100,
        }
