"""
Parser for the 2D Dam Break dataset from LagrangeBench.

Ref: https://lagrangebench.readthedocs.io/en/latest/pages/data.html#lagrangebench.data.data.DAM2D
"""

# TODO: Fix
from neuralmpm.data.parsers.nmpm_parser import NMPMParser


class Dam2DParser(NMPMParser):
    def __init__(self, path):
        super().__init__(
            path=path,
            dim=2,
            dt=0.03,
            low_coords=(0.05, 0.05),
            high_coords=(5.44, 2.07),
        )

    def get_sim_length(self):
        return 401

    def get_num_types(self):
        return 2

    def get_splits_sizes(self):
        return {
            "train": 50,
            "valid": 25,
            "test": 25,
        }

    def __len__(self):
        raise 50
