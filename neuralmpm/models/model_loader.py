from neuralmpm.models.networks import (
    UNet,
    FNO,
    FFNO,
    # UNO,
    UNet3D,
    CNO2d,
)

ALL = {
    "unet": UNet,
    "fno": FNO,
    "ffno": FFNO,
    # "uno": UNO,
    "unet_3d": UNet3D,
    "cno": CNO2d,
}


# TODO be able to pass all parameters at once using **config_dict


def from_config(config_dict, data_parser):
    return create_model(
        model_type=config_dict["model"],
        architecture=config_dict["architecture"],
        steps_per_call=config_dict["steps_per_call"],
        grid_size=config_dict["grid_size"],
        in_channels=data_parser.get_num_channels(),
        out_channels=(data_parser.get_num_types() - 1) * data_parser.get_dim(),
    )


# TODO use **config_dict in model call to pass all parameters at once
def create_model(
    model_type, architecture, grid_size, steps_per_call, in_channels=4, out_channels=2
):
    """
    Instantiates a model with the given parameters
    and returns it.

    config_dict must contain the architecture with the model parameters as
    well as the required hyperparameters such as grid_size, step_per_call, ...

    Args:
        model_type: str
            Name of the model to instantiate
        config_dict: dict
            Dictionary containing the model parameters and hyperparameters

    Returns:
        model: torch.nn.Module
            The instantiated model
    """

    # TODO: Smarter way to handle num_channels
    #  - Either "hardcode" a formula depending on the model
    #  - have a get_num_channels in Parser
    #  - or even more modular (so we can use same dataset with different
    #  configurations): have some sort of pre_processor somewhere idk

    if model_type == "fno":
        hidden_channels = architecture["hidden"]
        modes = architecture["modes"]
        if modes is None:
            modes = grid_size // 2
        use_mlp = architecture.get("use_mlp", False)
        model = FNO(
            in_channels=4,
            hidden_channels=hidden_channels,
            num_preds=steps_per_call,
            out_channels=2,
            modes=modes,
            use_mlp=use_mlp,
        )
    elif model_type == "ffno":
        hidden_channels = architecture[:-1]
        modes = architecture[-1]
        if modes is None:
            modes = grid_size // 2
        model = FFNO(
            in_channels=4,
            hidden_channels=hidden_channels,
            num_preds=steps_per_call,
            out_channels=2,
            modes=modes,
        )
        # TODO ffno
    elif model_type == "unet":
        architecture = architecture["hidden"] + [steps_per_call]
        factors = [2] * (len(architecture) - 1)
        # TODO Dynamically adjust
        # in_channels = vels + dens = dim * (num_types - 1) + 1 * num_types
        # out_channels = (num_types) * config_dict['dim']  # TODO
        model = UNet(
            architecture, factors, in_channels=in_channels, out_channels=out_channels
        )
        print(f"Using a UNet model ({in_channels} -> {out_channels})")
        # model = torch.compile(model)
    elif model_type == "cno":
        model = CNO2d(
            in_channels=in_channels,
            out_channels=out_channels,
            size=grid_size[0],
            N_layers=architecture["n_layers"],
            steps_per_call=steps_per_call,
            channel_multiplier=architecture["chan_mult"],
            N_res=architecture.get("n_res", 4),
            N_res_neck=architecture.get("n_res_neck", 4),
            mlp_architecture=architecture.get("mlp_architecture", [64, 64, 64]),
            use_bn=False,
        )
        print(f"Using a CNO model ({in_channels} -> {out_channels})")
        # model = torch.compile(model)
    elif model_type == "unet_3d":
        architecture = architecture["hidden"] + [steps_per_call]
        factors = [2] * (len(architecture) - 1)
        model = UNet3D(architecture, factors, in_channels=5)
    elif model_type == "uno":
        """
        Parameters
        ----------

        hidden_channels: initial width of the UNO (after lifting)
            e.g., 128
        uno_out_channels: output channels of each Fourier Layer.
            e.g., [32, 64, 64, 32] for 4 layers
        n_modes: Fourier Modes to use in integral operation of each Fourier Layers
            e.g., [5, 5, 5, 5] for 4 layers
        scalings: Scaling factors for each Fourier Layer
            e.g., [1.0, 0.5, 1.0, 2.0] for 4 layers
        """

        architecture.setdefault("hidden_channels", 128)
        architecture.setdefault("uno_out_channels", [32, 64, 64, 32])
        architecture.setdefault("n_modes", [5, 5, 5, 5])
        architecture.setdefault("scalings", [1.0, 0.5, 1.0, 2.0])

        architecture["n_modes"] = [
            [n_modes, n_modes] if not isinstance(n_modes, list) else n_modes
            for n_modes in architecture["n_modes"]
        ]
        architecture["scalings"] = [
            [scaling, scaling] if not isinstance(scaling, list) else scaling
            for scaling in architecture["scalings"]
        ]

        architecture["uno_n_modes"] = architecture.pop("n_modes")
        architecture["uno_scalings"] = architecture.pop("scalings")

        architecture["n_layers"] = len(architecture["uno_out_channels"])

        # model = UNO(
        #     4, 2, num_preds=steps_per_call, **config_dict[
        #     "architecture"]
        # )
        model = None
    else:
        raise ValueError(
            f"Model {model_type} not recognized.\n Valid Types: {ALL.keys()}"
        )

    return model
