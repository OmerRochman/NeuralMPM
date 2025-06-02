import json
import os

import matplotlib.pyplot as plt
import torch
from matplotlib.ticker import FuncFormatter

from neuralmpm.data.data_manager import PARSERS
from neuralmpm.models import model_loader


# TODO remove
def get_valid_sims(files, material, load_sim_fun):
    sims = []
    types = []

    for f in files:
        sim, type_, _ = load_sim_fun(f, material)

        sims.append(sim)
        types.append(type_)

    return sims, types


# TODO remove/move elsewhere, need a centralized and clean way to save/load
#  models. Rewrite ModelLogger?
def load_config_and_model(path: str, name: str = None):
    with open(os.path.join(path, "config.json"), "r") as f:
        config_dict = json.load(f)

    model = load_model(path, config_dict, checkpoint_name=name)

    return config_dict, model


# TODO remove, or change at least (for the best by default)
def load_model(path: str, config_dict: dict, checkpoint_name: str = None):
    """

    Args:
        checkpoint_name: Time of the checkpoint to load.
        If None, will load the latest checkpoint.

    Returns:

    """

    if checkpoint_name is None:
        """
        Loads the latest checkpoint saved.
        checkpoint_name = sorted(
            os.listdir(self.model_folder),
            key=lambda x: int(x.split('/')[-1].split(".")[0])
        )[-1]
        """

        checkpoint_name = "best"

    if checkpoint_name == "last":
        checkpoint_name = sorted(
            [
                int(file.split(".")[0])
                for file in os.listdir(os.path.join(path, "models"))
                if file.endswith(".ckpt") and file != "best.ckpt"
            ]
        )[-1]
        print("Loading last saved checkpoint:", checkpoint_name)

    checkpoint_name = os.path.join(path, "models", f"{checkpoint_name}.ckpt")

    """
    if config_dict['model'] == 'unet':
        architecture = config_dict['architecture']['hidden'] + [
            config_dict['steps_per_call']]
        factors = [2] * (len(architecture) - 1)
        model = UNet(architecture, factors, in_channels=4)
        model = torch.compile(model)
    """

    # TODO
    if "dataset_type" in config_dict:
        dataset_type = config_dict["dataset_type"]
        parser = PARSERS[dataset_type](config_dict["data"])
        model = model_loader.from_config(config_dict, parser)
    else:
        model = model_loader.create_model(config_dict["model"], config_dict)

    model.load_state_dict(torch.load(checkpoint_name))

    return model


def plot_error_over_time(err_list):
    mean = sum(err_list) / len(err_list)

    # Calculate variance or standard deviation for the shadow
    std_dev = torch.sqrt(sum([x**2 for x in err_list]) / len(err_list) - mean**2)

    # Convert to CPU if necessary (comment this out if not using PyTorch)
    mean = mean.cpu()
    std_dev = std_dev.cpu()
    # lower = torch.maximum(torch.zeros_like(mean), mean - std_dev).cpu()
    low = torch.inf
    for x in err_list:
        if x.mean() < low:
            lower = x.cpu()
            low = x.mean()

    def format_yaxis(x, pos):
        return f"{x * 1e3:.0f}"

    # Apply formatter to the y-axis

    plt.figure(figsize=(10, 6))
    plt.plot(mean, color="red", label="Mean")  # Plot the mean
    plt.fill_between(
        range(len(mean)),
        lower,
        mean + std_dev,
        color="orange",
        alpha=0.5,
        label="Standard Deviation",
    )
    plt.gca().yaxis.set_major_formatter(FuncFormatter(format_yaxis))

    plt.title("Plot of Mean with Standard Deviation Shadow")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_all_erros(err_list):
    def format_yaxis(x, pos):
        return f"{x * 1e3:.0f}"

    # Apply formatter to the y-axis

    plt.figure(figsize=(10, 6))

    for x in err_list:
        plt.plot(x.cpu())  # Plot the mean
    plt.gca().yaxis.set_major_formatter(FuncFormatter(format_yaxis))

    plt.xlabel("Time")
    plt.ylabel("Error")
    # plt.legend()
    plt.grid(True)
    plt.show()


# ? -> Util
def tensor_list_mean(data):
    mean_dict = {}
    for key, nested_dict in data.items():
        mean_dict[key] = {}
        for subkey, tensor_list in nested_dict.items():
            # Stack the list of tensors and compute the mean
            mean_dict[key][subkey] = torch.stack(tensor_list).mean(dim=0)
    return mean_dict


# ? -> Util
def reduce_results(data):
    reduced = tensor_list_mean(data)
    reduced = torch.utils._pytree.tree_map(lambda x: x.mean().cpu(), reduced)
    return reduced


# Move somewhere else, maybe a secondary package about figures.
def bar_plots(data, subkey, ylabel, titles):
    # Number of plots
    N = len(data)

    # Create a figure with 1 row and N columns
    fig, axs = plt.subplots(1, N, figsize=(10 * N, 5), sharey=True)

    # Ensure axs is iterable
    if N == 1:
        axs = [axs]

    for i, (ax, d, tit) in enumerate(zip(axs, data, titles)):
        # Prepare plot data
        keys = list(d.keys())
        means = [d[key][subkey] for key in keys]  # Directly use scalar values

        # Plotting
        ax.bar(keys, means, color="skyblue")
        ax.set_title(tit)

        # Set y-axis label and formatter only for the first plot
        if i == 0:
            ax.set_ylabel(ylabel)
            ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x * 1e3:.0f}"))

            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            # ax.spines['bottom'].set_visible(False)
            # ax.spines['left'].set_visible(False)
        else:
            ax.yaxis.set_visible(False)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            # ax.spines['bottom'].set_visible(False)
            ax.spines["left"].set_visible(False)

    plt.show()
