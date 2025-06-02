"""
Parser for monomaterial 2D datasets from GNS.

This includes:
  - WaterRamps
  - SandRamps
  - Water
  - Sand
  - Goop
  - WaterDrop
"""

from neuralmpm.data.parsers.nmpm_parser import NMPMParser


class MonoMatParser(NMPMParser):
    def __init__(self, path):
        super().__init__(
            path=path,
            dim=2,
            dt=0.0025,
            low_coords=(0.08, 0.08),
            high_coords=(0.92, 0.92),
        )

        l_path = path.split("/")[-1].lower()
        self.dataset_type = l_path
        if "water" in l_path:
            self.material = "water"
        elif "sand" in l_path:
            self.material = "sand"
        elif "goop" in l_path:
            self.material = "goop"
        else:
            raise ValueError(f"Invalid monomaterial dataset: {path}")

    def get_sim_length(self):
        sim_lengths = {
            "water": 1000,
            "sand": 320,
            "goop": 400,
            "sandramps": 400,
            "waterramps": 600,
            "waterdrop": 1000,
        }

        return sim_lengths[self.dataset_type]

    def get_num_types(self):
        return 2  # Wall & fluid

    def get_colors(self):
        type_colors = {
            "water": "blue",
            "sand": "gold",
            "goop": "magenta",
        }
        return ["black", type_colors[self.material]]

    def get_splits_sizes(self):
        if self.dataset_type in ["waterramps", "sandramps"]:
            return {
                "train": 1000,
                "valid": 100,
                "test": 100,
            }

        return {
            "train": 1000,
            "valid": 30,
            "test": 30,
        }


if __name__ == "__main__":
    parser = MonoMatParser("data/Water")
    parser.parse(0)
