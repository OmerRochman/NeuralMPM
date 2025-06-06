r"""Slurm constants"""

# General
SLURM_ACCOUNT = None

# Training
SLURM_TRAINING_PARTITION = "gpu"
SLURM_TRAINING_CPUS = 8
SLURM_TRAINING_RAM = "48GB"
SLURM_TRAINING_TIME = "2-00:00"

# Evaluation chunk computation
SLURM_EVAL_COMPUTE_PARTITION = "gpu"
SLURM_EVAL_COMPUTE_CPUS = 8
SLURM_EVAL_COMPUTE_RAM = "48GB"
SLURM_EVAL_COMPUTE_TIME = "2-00:00"
SLURM_EVAL_COMPUTE_GPUS = (
    1  # Can be turned off for CPU computation of MSE. Use a GPU for EMD!
)
# Evaluation chunk merging
SLURM_EVAL_MERGE_PARTITION = "gpu"
SLURM_EVAL_MERGE_CPUS = 8
SLURM_EVAL_MERGE_RAM = "16GB"
SLURM_EVAL_MERGE_TIME = "05:00"

# Dataset statistics chunk computation
SLURM_STATS_COMPUTE_PARTITION = "gpu"
SLURM_STATS_COMPUTE_CPUS = 8
SLURM_STATS_COMPUTE_RAM = "48GB"
SLURM_STATS_COMPUTE_TIME = "2-00:00"
SLURM_STATS_COMPUTE_GPUS = 1
# Dataset statistics chunk merging
SLURM_STATS_MERGE_PARTITION = "gpu"
SLURM_STATS_MERGE_CPUS = 8
SLURM_STATS_MERGE_RAM = "16GB"
SLURM_STATS_MERGE_TIME = "05:00"
