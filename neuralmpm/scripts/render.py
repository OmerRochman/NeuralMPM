"""
TODO: Could be sent as a Slurm array job with dependencies (or even not).

Could even distribute each frame to a CPU job.
"""

import argparse
import json
import os
from glob import glob

import h5py
from natsort import natsorted
from tqdm import tqdm
import numpy as np

import neuralmpm.pipelines.visualisation as viz
from neuralmpm.data.data_manager import PARSERS
from neuralmpm.util.string_utils import ids_to_list


# TODO Split videos and frames
# setup(), render_video(), render_frame() ?
# Separate single h5 rendering, and the tools for rendering a set of h5 files for NeuralMPM.
def render_simulations(
    run_path,  # TODO: How to render from DMCF and GNS?
    dataset,
    sim_ids,
    render_type,
    time_steps,
    gt=False,
    img_type="png",
    engine="v2",
    bounds=None,
    no_walls=False,
):
    """
    Render a set of simulations using a trained NeuralMPM model.

    Args:
        rollouts_folder: Path to the folder containing the rollouts.
        sim_ids: IDs of the simulations to render.
        render_type: Type of rendering to perform. (video or frames)
        time_steps: Time steps to render.
        gt: Whether to render the ground truth.
        img_type: Image type to use for rendering.
        engine: Engine to use for rendering (v1 or v2)
    """
    sim_ids = ids_to_list(sim_ids)
    time_steps = ids_to_list(time_steps)

    # TODO, find a cleaner way, make it consistent with nmpm-rollout
    rollouts_path = os.path.join(run_path, "rollouts", dataset)
    assert os.path.exists(rollouts_path), "Rollouts do not exist."

    renders_path = os.path.join(run_path, "renders", dataset)
    os.makedirs(renders_path, exist_ok=True)

    files = glob(rollouts_path + "/*.h5")
    files = natsorted(files)
    files = [files[i] for i in sim_ids]

    colors = None

    # Check if config.json exists
    # Might not exist for rendering external methods, ...
    if os.path.exists(os.path.join(run_path, "config.json")):
        with open(os.path.join(run_path, "config.json"), "r") as f:
            config_dict = json.load(f)
        dataset_type = config_dict["dataset_type"]
        parser = PARSERS[dataset_type](config_dict["data"])

        if bounds is None:
            bounds = (
                (parser.low_bound[0].cpu().numpy(), parser.high_bound[0].cpu().numpy()),
                (parser.low_bound[1].cpu().numpy(), parser.high_bound[1].cpu().numpy()),
            )

        colors = parser.get_colors()

    assert bounds is not None, (
        'Bounds must be provided. E.g., --bounds "((0, 1) (0, 1))"'
    )

    pbar = tqdm(total=len(sim_ids), desc="Simulations rendered")
    for sim_id, f in zip(sim_ids, files):
        with h5py.File(f, "r") as file:
            if "ground_truth_rollout" in file:
                gt_traj = file["ground_truth_rollout"][()]
            else:
                gt_traj = None
            pred_traj = file["predicted_rollout"][()]
            types = file["types"][()]

        # TODO
        if gt_traj is None:
            gt_traj = pred_traj

        # Trick for the types, might need to allow for custom rendering :-)
        types = np.array([np.where(np.unique(types) == t)[0][0] for t in types])

        length = gt_traj.shape[0] - 1

        if render_type == "frames":
            os.makedirs(renders_path + "/frames", exist_ok=True)
            for i in time_steps:
                if gt:
                    cloud_gt = gt_traj[i, ..., :2]
                    viz.render_cloud(
                        cloud_gt,
                        types=types,
                        save_path=f"{renders_path}/frames/rollout_"
                        f"{sim_id}/gt_{i}.{img_type}",
                        engine=engine,
                        bounds=bounds,
                        no_walls=no_walls,
                        colors=colors,
                    )

                cloud_pred = pred_traj[i, ..., :2]
                viz.render_cloud(
                    cloud_pred,
                    types=types,
                    save_path=f"{renders_path}/frames/rollout_"
                    f"{sim_id}/pred_{i}.{img_type}",
                    engine=engine,
                    bounds=bounds,
                    no_walls=no_walls,
                    colors=colors,
                )
        else:
            if engine == "v2":
                viz.animate_comparison_v2(
                    pred_traj[:length, ..., :2],
                    truth=gt_traj[1:, ..., :2],
                    types=types,
                    interval=1,
                    save_path=f"{renders_path}/videos/rollout_{sim_id}.mp4",
                    return_ani=False,
                    as_array=True,
                    bounds=bounds,
                    colors=colors,
                    fps=75,
                )
            else:
                """
                viz.animate_comparison(
                    pred_traj[:length],
                    truth=gt_traj[1:, ..., :2],
                    type_=types,
                    interval=1,
                    save_path=f"{renders_path}/videos/rollout_{sim_id}.mp4",
                    return_ani=False,
                    as_array=True,
                    bounds=bounds,
                    fps=75,
                    show_progress=True,
                )
                """
                viz.animate_comparison(
                    pred_traj[:length],
                    truth=gt_traj[1:, ..., :2],
                    type_=types,
                    interval=1,
                    save_path=f"{renders_path}/videos/rollout_{sim_id}.mp4",
                    return_ani=False,
                    as_array=True,
                    bounds=bounds,
                    colors=colors,
                    fps=75,
                    show_progress=True,
                )

        pbar.update(1)

    pbar.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("type", type=str, help="video or frames")
    parser.add_argument(
        "--run", "-r", type=str, help="Path to the run folder containing the model."
    )
    parser.add_argument(
        "--data", "-d", type=str, help="Path to the folder containing the rollouts."
    )
    parser.add_argument(
        "--ids", "-i", type=str, help="IDs of the simulations to render."
    )
    parser.add_argument(
        "--engine", "-e", type=str, help="Engine to use for rendering", default="v2"
    )
    parser.add_argument(
        "--bounds", type=str, help="Bounds of the simulation", default=None
    )
    parser.add_argument(
        "--no-walls",
        help="Do not render wall borders",
        action="store_true",
        default=False,
    )

    # For frames
    parser.add_argument(
        "--steps", type=str, help="Time steps to render", default="0,100,200,300"
    )
    parser.add_argument("--gt", type=bool, help="Render ground truth", default=False)
    parser.add_argument("--img-type", type=str, help="Image type", default="png")

    # For videos
    parser.add_argument("--fps", type=int, default=75)
    # TODO Colors argument
    args = parser.parse_args()

    # Eval bounds if provided: "(0, 1), (0, 1)" -> ((0, 1), (0, 1))
    if args.bounds is not None:
        args.bounds = eval(args.bounds)

    render_simulations(
        args.run,
        args.data,
        args.ids,
        args.type,
        args.steps,
        args.gt,
        args.img_type,
        args.engine,
        args.bounds,
        args.no_walls,
        # TODO: extra_args (eg fps for videos, type for imgs)
    )


if __name__ == "__main__":
    main()
