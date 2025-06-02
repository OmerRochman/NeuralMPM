from neuralmpm.data.parsers.nmpm_parser import NMPMParser


class WaterDropXLParser(NMPMParser):
    def __init__(self, path):
        super().__init__(
            path=path,
            dim=2,
            dt=0.0025,
            low_coords=(0.04, 0.04),
            high_coords=(0.97, 0.97),
        )

    def get_num_types(self):
        return 2

    def get_splits_sizes(self):
        return {
            "train": 1000,
            "valid": 100,
            "test": 100,
        }

    def get_sim_length(self):
        return 1000
