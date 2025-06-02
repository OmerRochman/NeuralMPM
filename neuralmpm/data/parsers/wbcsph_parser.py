from neuralmpm.data.parsers.nmpm_parser import NMPMParser


class WBCSPHParser(NMPMParser):
    def __init__(self, path):
        super().__init__(
            path=path,
            dim=2,
            dt=0.0025,
            low_coords=(0.0025, 0.0025),
            high_coords=(0.9975, 0.9975),
        )

    def get_num_types(self):
        return 2

    def get_num_channels(self):
        return 6

    def get_sim_length(self):
        return 3200

    def __len__(self):
        raise 30

    def get_splits_sizes(self):
        return {
            "train": 30,
            "valid": 9,
            "test": 9,
        }
