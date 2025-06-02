r"""Parser for monomaterial 3D datasets from GNS (Water, Sand, Goop)."""

from neuralmpm.data.parsers.nmpm_parser import NMPMParser


class Mono3DParser(NMPMParser):
    def __init__(self, path):
        super().__init__(
            path=path,
            dim=3,
            dt=0.005,
            low_coords=(0.08, 0.08, 0.08),
            high_coords=(0.92, 0.92, 0.92),
            interaction_radius=0.035,
        )

        l_path = path.lower()
        if "water" in l_path:
            self.material = "water"
        elif "sand" in l_path:
            self.material = "sand"
        elif "goop" in l_path:
            self.material = "goop"
        else:
            raise ValueError(f"Invalid monomaterial dataset: {path}")

    def get_sim_length(self):
        if self.material == 5:  # Water
            return 800
        elif self.material == 6:  # Sand
            return 350

        return 300  # Goop

    def get_num_types(self):
        return 2  # Wall & fluid

    def __len__(self):
        raise 1000  # 1000 training trajectories
