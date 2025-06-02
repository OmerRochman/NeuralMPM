import argparse

import torch

from neuralmpm.pipelines import training


def main():
    parser = argparse.ArgumentParser("NeuralMPM Training")

    parser.add_argument("--dataset-type", "-dt", type=str, default="monomat2d")
    parser.add_argument("--data", "-d", type=str, help="Path to dataset", required=True)

    parser.add_argument("--nowandb", action="store_true")
    parser.add_argument("--slurm", action="store_true")
    parser.add_argument("--epochs", help="Number of epochs", type=int, default=50)
    parser.add_argument(
        "--name",
        help="Name of the run",
        type=str,
        default=None,
        dest="run_name",
    )
    parser.add_argument(
        "-p", "--project", help="Name of the project", type=str, default="NeuralMPM"
    )

    parser.add_argument(
        "--steps-per-call",
        help="Number of predictions per model call",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--autoregressive-steps",
        help="Number of autoregressive steps during training",
        type=int,
        default=4,
    )
    # --grid-size 64 128
    # arg with multiple possible values
    parser.add_argument(
        "--grid-size",
        nargs="+",
        type=int,
        help="Grid size of the simulation",
        default=[64, 64],
    )
    parser.add_argument("--model", help="Model type", type=str, default="unet")

    parser.add_argument("--batch-size", help="Batch size", type=int, default=128)

    parser.add_argument(
        "--passes-over-buffer",
        help="How many times to repeat a buffer",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--sims-in-memory",
        help="Simulates to load in one buffer",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--architecture",
        nargs="+",
        type=int,
        help="An integer list of architecture parameters",
        default=[64, 64, 64],
    )
    parser.add_argument(
        "--particle-noise", help="Particle noise", type=float, default=0.0003
    )
    parser.add_argument(
        "--grid-noise", help="Grid noise", nargs="+", type=float, default=0.001
    )

    parser.add_argument("--lr", help="Initial learning rate", type=float, default=1e-3)
    parser.add_argument(
        "--min-lr", help="Minimum learning rate", type=float, default=1e-6
    )
    parser.add_argument("--use-schedulers", action="store_true")
    parser.add_argument("--no-skip-steps", action="store_false")
    parser.add_argument("--no-gpu", action="store_true")

    args = parser.parse_args()

    if not args.no_gpu and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    config_dict = vars(args)
    config_dict["use_wandb"] = not parser.parse_args().nowandb
    config_dict["skip_m_steps"] = parser.parse_args().no_skip_steps
    config_dict["architecture"] = {"hidden": config_dict["architecture"]}

    training.setup_train_eval(config_dict, progress_bars=True, device=device)


if __name__ == "__main__":
    main()
