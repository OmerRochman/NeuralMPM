DEFAULT = {
    "model": "unet",
    "run_name": "WaterRamps",
    "data": "data/WaterRamps",
    "dataset_type": "monomat2d",
    "epochs": 750,
    "steps_per_call": 8,
    "autoregressive_steps": 4,
    "grid_size": [64, 64],
    "batch_size": 128,
    "passes_over_buffer": 20,  # Number of passes for each buffer.
    "sims_in_memory": 16,  # Number of simulations in one buffer.
    "architecture": {
        "hidden": [64, 64, 64, 64]
    },  # Channels for each block of the "encoder"
    "lr": 1e-3,
    "min_lr": 1e-5,
    "use_schedulers": True,
    "skip_m_steps": True,  # If True, will sample every m steps in the dataset.
    "length": 0.015,  # Deprecated, can be used for different voxelization kernels.
    "grid_noise": 0.0,
    "particle_noise": 0.0,
}

paper = {
    "run_name": (
        "WaterRamps",
        "SandRamps",
        "Goop",
        "MultiMaterial",
        "DamBreak2D",
        "VariableGravity",
    ),
    "data": (
        "data/WaterRamps",
        "data/SandRamps",
        "data/Goop",
        "data/MultiMaterial",
        "data/DamBreak2D",
        "data/VariableGravity",
    ),
    "dataset_type": (
        "monomat2d",
        "monomat2d",
        "monomat2d",
        "multimat",
        "dam2d",
        "watergravity",
    ),
}

# Toy experiment to check installation.
toy = {
    "run_name": "Debug Toy Run",
    "epochs": 10,
    "steps_per_call": 2,
    "autoregressive_steps": 2,
    "grid_size": [16, 16],
    "batch_size": 16,
    "passes_over_buffer": 1,  # Number of passes for each buffer.
    "sims_in_memory": 2,  # Number of simulations in one buffer.
    "architecture": {"hidden": [4, 4]},
}
