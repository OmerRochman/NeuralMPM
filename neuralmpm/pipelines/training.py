"""
TODO:
  - Add type hints.
  - Add docstrings.
  -
"""

import random
from typing import Callable
import gc

import numpy as np
import torch
import wandb
from tqdm import tqdm

import neuralmpm.pipelines.simulation as simulate
from neuralmpm.models import losses
from neuralmpm.models.model_logger import ModelLogger
from neuralmpm.pipelines.visualisation import animate_comparison
from neuralmpm.data.data_manager import DataManager
from neuralmpm.models import model_loader
from neuralmpm.configs.configs import DEFAULT
from neuralmpm.scripts.rollout import rollout
from neuralmpm.scripts.eval import eval_rollouts


def print_training_info(config_dict):
    print("------------------------------------")
    print("Training NeuralMPM\n")
    print("Configuration:")
    for key, value in config_dict.items():
        key = key.replace("_", " ").capitalize()
        print(f"  {key}: {value}")
    print("------------------------------------\n")


def create_noise_tensors(parser, grid_noise, particle_noise, device="cpu"):
    """
    Utility function to create noise tensors for grid and particles.

    If given a float, the grid noise tensor will make it a list where every
    element has the given value, except for the wall density which is not noised.

    Args:
        parser: DataManager parser.
        grid_noise: Grid noise (list or float)
        particle_noise: Particle noise (float)
        device: Device to move tensors to.
    """

    num_fluids = parser.get_num_types() - 1
    if grid_noise is None:
        # Speed then density. First density is wall, do not add noise.
        grid_noise = [0.0] * num_fluids * parser.dim + [0.0] + [0.0] * num_fluids
    elif len(grid_noise) == 1:
        grid_noise = (
            [grid_noise[0]] * num_fluids * parser.dim
            + [0.0]
            + [grid_noise[0]] * num_fluids
        )

    return grid_noise.to(device), particle_noise.to(device)


def setup_train_eval(config_dict, progress_bars=False, device="cpu"):
    training_setup = setup(config_dict=config_dict, device=device)
    print_training_info(config_dict)
    train(**training_setup, progress_bars=progress_bars, device=device)

    print("Training done. Evaluating...")
    run_path = training_setup["model_logger"].folder
    data_path = config_dict["data"]
    parser = training_setup["datamanager"].parser
    emd_num_chunks = parser.get_sim_length() // 10
    for ckpt in ["best", "last"]:
        try:
            print("Evaluating checkpoint:", ckpt)
            if parser.get_sim_length() > 1000 or parser[0]["sim"].shape[1] > 4000:
                batch_size = 2
            else:
                batch_size = 16
            rollout(run_path, data_path, batch_size, ckpt)
            metrics_mean = eval_rollouts(
                run_path,
                data_path.split("/")[-1],
                use_emd=True,
                num_chunks=emd_num_chunks,
            )

            print("metric mean dict", metrics_mean)

            for metric, value in metrics_mean.items():
                print(f"  {ckpt}_{metric}: {value}")
                if training_setup["run"]:
                    wandb.run.summary[f"{ckpt}_{metric}"] = value
        except Exception:
            print(
                f"Error during evaluation of the {ckpt} checkpoint. Probably trained for too short."
            )

    if training_setup["run"]:
        training_setup["run"].finish()


def setup(config_dict, device="cpu"):
    """ """
    def_config = DEFAULT.copy()
    def_config.update(config_dict)
    config_dict = def_config

    datamanager = DataManager(
        config_dict["data"],
        config_dict["dataset_type"],
        batch_size=config_dict["batch_size"],
        grid_size=config_dict["grid_size"],
        steps_per_call=config_dict["steps_per_call"],
        autoregressive_steps=config_dict["autoregressive_steps"],
        sims_in_memory=config_dict["sims_in_memory"],
        skip_m_steps=config_dict["skip_m_steps"],
        device=device,
    )

    num_total_iterations = (
        config_dict["epochs"]
        * datamanager.iters_per_epoch
        * config_dict["passes_over_buffer"]
    )
    config_dict["total_iterations"] = num_total_iterations

    model = model_loader.from_config(config_dict, datamanager.parser)

    model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=config_dict["lr"])
    schedulers = None
    if config_dict.get("use_schedulers", False):
        warmup_end = 100  # TODO
        cosine_start = 1000
        total_iters = num_total_iterations
        warmup_end = min(warmup_end, cosine_start)
        linear_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1e-2,
            end_factor=1.0,
            total_iters=warmup_end,
        )
        cos_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=total_iters - cosine_start,
            eta_min=config_dict["min_lr"],
            last_epoch=-1,
        )
        schedulers = {
            "linear": linear_scheduler,
            "cosine": cos_scheduler,
            "warmup_end": warmup_end,
            "cosine_start": cosine_start,
            "total_iters": total_iters,
        }

    if config_dict.get("use_wandb", True):
        run = wandb.init(
            entity="neuralmpm",
            project=config_dict.get("project", "NeuralMPM"),
            name=config_dict.get("run_name", None),
            config=config_dict,
        )
        run.watch(model, log="all", log_freq=4)

        config_dict["project"] = wandb.run.project
        config_dict["run_name"] = wandb.run.name
        config_dict["run_id"] = wandb.run.id
    else:
        run = None

    model_logger = ModelLogger(
        config_dict["data"].rstrip("/").split("/")[-1],
        config_dict,
        save_interval=config_dict.get("save_every", 10),
    )

    return {
        "model": model,
        "datamanager": datamanager,
        "optimizer": optimizer,
        "schedulers": schedulers,
        "run": run,
        "model_logger": model_logger,
        "epochs": config_dict["epochs"],
        "passes_over_buffer": config_dict["passes_over_buffer"],
        "particle_noise": torch.tensor(config_dict["particle_noise"]).to(device),
        "grid_noise": torch.tensor(config_dict["grid_noise"]).to(device),
    }


# TODO: Move to the evaluation pipeline.
@torch.no_grad()
def eval_iteration(
    total_step: int,
    datamanager: DataManager,
    model: torch.nn.Module,
    gmean: torch.Tensor,
    gstd: torch.Tensor,
    model_logger: ModelLogger,
    use_wandb: bool,
    device="cpu",
):
    init_state, valid_types, valid_sims = datamanager.get_valid_sims()
    sim_length = datamanager.parser.get_sim_length() - 1

    # TODO: As parameter.
    if sim_length > 1000:
        sim_length = 1000

    num_calls = sim_length // datamanager.steps_per_call + 1

    _, trajectories, _ = simulate.unroll(
        model,
        init_state,
        datamanager.grid_coords,
        num_calls,
        gmean,
        gstd,
        valid_types,
        datamanager.parser.get_num_types(),
        datamanager.parser.low_bound.to(device),
        datamanager.parser.high_bound.to(device),
        datamanager.size_tensor.to(device),
        interaction_radius=datamanager.interaction_radius,
        interp_fn=datamanager.interp_fn,
    )

    time_err = []
    for i in range(len(valid_sims)):
        type_mask = valid_types[i] != 0
        type_mask_vsim = type_mask[: valid_sims[i].shape[1]]
        te = (
            valid_sims[i][1:sim_length, type_mask_vsim, : datamanager.parser.dim]
            - trajectories[i, : sim_length - 1, type_mask, : datamanager.parser.dim]
        ) ** 2
        te = te.mean(axis=(1, 2))
        time_err += [te]

        # TODO: Use metrics.mse

    time_err = sum(time_err) / len(time_err)
    err = time_err.mean()

    plot_idx = torch.randint(0, len(valid_sims), (1,)).item()

    last_sim = valid_sims[plot_idx]
    last_type = valid_types[plot_idx]
    last_traj = trajectories[plot_idx]

    vid = None
    time_err_plt = None

    if err <= model_logger.best_val:
        model_logger.best_val = err

        if model_logger:
            model_logger.save_model(
                model,
                checkpoint_name="best",
                json_dict={
                    "total_step": total_step,
                    "total_error": err.item(),
                },
            )

        data = [
            [x.item(), y.item()]
            for (x, y) in zip(torch.arange(time_err.shape[0]), time_err)
        ]
        table = wandb.Table(data=data, columns=["t", "err"])

        if use_wandb:
            time_err_plt = wandb.plot.line(table, "t", "err", title="Error over time")

            length = sim_length - 1
            vid = animate_comparison(
                last_traj[:length, ..., :2].detach().cpu().numpy(),
                truth=last_sim[1 : sim_length + 1, ..., :2].detach().cpu().numpy(),
                type_=last_type.detach().cpu().numpy(),
                colors=datamanager.parser.get_colors(),
                interval=1,
                save_path=None,
                return_ani=False,
                as_array=True,
                bounds=(
                    (
                        datamanager.parser.low_bound[0].cpu().numpy(),
                        datamanager.parser.high_bound[0].cpu().numpy(),
                    ),
                    (
                        datamanager.parser.low_bound[1].cpu().numpy(),
                        datamanager.parser.high_bound[1].cpu().numpy(),
                    ),
                ),
                fig_base_size=4,
            )

            vid = wandb.Video(vid, fps=16)

    del init_state, valid_types, valid_sims

    return err, time_err_plt, vid


def train(
    model: torch.nn.Module,
    datamanager,
    optimizer: torch.optim.Optimizer,
    schedulers: dict,
    loss_fn: Callable = losses.particle_position_mse_loss,
    run=None,
    grad_accumulation_steps: int = 1,
    passes_over_buffer: int = 3,
    epochs: int = 10,
    model_logger: ModelLogger = None,
    progress_bars: bool = False,
    grid_noise: torch.Tensor = None,
    particle_noise: torch.Tensor = None,
    seed: int = 42,
    device: str = "cpu",
):
    """
    Training pipeline.

    Args:
        model:
        datamanager:
        optimizer:
        schedulers:
        use_wandb:
        particle_noise:
        grid_noise:
        passes_over_buffer:
        epochs:
        model_logger:
        progress_bars:
        loss_fn:

    """
    # TODO make seed a parameter
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    if model_logger is not None:
        model_logger.start_timer()

    total_step = 0

    gpu = device == "cuda"

    step_fn = simulate.get_step_fn(
        datamanager.grid_coords,
        datamanager.interp_fn,
        datamanager.parser.get_num_types(),
        datamanager.parser.low_bound.to(device),
        datamanager.parser.high_bound.to(device),
        datamanager.size_tensor.to(device),
        datamanager.interaction_radius,
    )

    dim = datamanager.parser.dim
    low_bound = datamanager.parser.low_bound.to(device)
    high_bound = datamanager.parser.high_bound.to(device)

    grid_noise = grid_noise.to(device)
    particle_noise = particle_noise.to(device)

    for e in tqdm(range(epochs), desc="Epochs", leave=True, disable=not progress_bars):
        logs = {"epoch": e}

        dataloader = datamanager.load_buffer()
        gmean, gstd = (
            datamanager.gmean,
            datamanager.gstd,
        )  # TODO: use stats from whole dataset during training.

        for pss in tqdm(
            range(passes_over_buffer),
            desc="Buffer Passes",
            leave=False,
            disable=not progress_bars,
        ):
            if gpu:
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)

            grad_acc_step = 0
            for batch in tqdm(
                dataloader, desc="Batches", leave=False, disable=not progress_bars
            ):
                if gpu:
                    start.record()
                grids, states, targets, types, target_grids = batch

                grids, states, targets, types, target_grids = (
                    grids.to(device),
                    states.to(device),
                    targets.to(device),
                    types.to(device),
                    target_grids.to(device),
                )

                if grid_noise is not None and particle_noise is not None:
                    grids = grids + torch.randn_like(grids) * grid_noise
                    noisy_states = states + torch.randn_like(states) * particle_noise
                    states = torch.where(types[..., None] > 0.0, noisy_states, states)

                states[..., :dim] = torch.clamp(
                    states[..., :dim], min=low_bound, max=high_bound
                )

                loss = loss_fn(
                    model,
                    grids,
                    target_grids,
                    states,
                    targets,
                    gmean,
                    gstd,
                    types,
                    datamanager.parser.get_num_types(),
                    step_fn,
                    dim=datamanager.parser.dim,
                    steps_per_call=datamanager.steps_per_call,
                    autoregressive_steps=datamanager.autoregressive_steps,
                    debug=total_step,
                )

                if grad_acc_step == grad_accumulation_steps - 1:
                    optimizer.zero_grad()

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                if grad_acc_step == grad_accumulation_steps - 1:
                    optimizer.step()
                    grad_acc_step = 0
                else:
                    grad_acc_step += 1

                if schedulers is not None:
                    if total_step >= schedulers["cosine_start"]:
                        schedulers["cosine"].step()
                        current_lr = schedulers["cosine"].get_last_lr()
                    elif total_step < schedulers["warmup_end"]:
                        schedulers["linear"].step()
                        current_lr = schedulers["linear"].get_last_lr()

                else:
                    current_lr = optimizer.param_groups[0]["lr"]

                current_lr = (
                    current_lr[0] if isinstance(current_lr, list) else current_lr
                )
                if total_step % 100 == 0:
                    if run:
                        if gpu:
                            end.record()
                            torch.cuda.synchronize()
                            iter_time = start.elapsed_time(end)
                        else:
                            iter_time = 0

                        logs.update(
                            {
                                "loss": loss.item(),
                                "log_loss": torch.log10(loss).item(),
                                "total_step": total_step,
                                "sim_idx": datamanager.current_sim_idx,
                                "lr": current_lr,
                                "time": iter_time / 1000.0,
                            }
                        )
                        run.log(logs, step=total_step)

                    if model_logger is not None:
                        model_logger.try_saving(model)

                total_step += 1

        vid = None
        if e % 5 == 0:
            err, time_err_plt, vid = eval_iteration(
                total_step,
                datamanager,
                model,
                gmean,
                gstd,
                model_logger,
                run is not None,
                device=device,
            )

            if run:
                logs["total_step"] = total_step
                logs["total_error"] = err
                if time_err_plt is not None:
                    logs["error_over_time"] = time_err_plt
                if vid is not None:
                    logs["vid"] = vid

        if run:
            run.log(logs, step=total_step)

        del vid
        torch.cuda.empty_cache()
        gc.collect()
