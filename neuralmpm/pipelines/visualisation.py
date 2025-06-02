"""
# TODO: Clean it

Rewrite completely yeah

TODO: Allow single sim video rendering.

TODO: Color
"""

import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import animation
from scipy import stats
from scipy.interpolate import PchipInterpolator
from tqdm import tqdm


DEFAULT_COLORS = (
    "black",  # walls
    "blue",  # water
    "gold",  # sand
    "magenta",  # goop
    "red",  # solids
    "green",  # gas
    "orange",  # fire
)


@torch.no_grad()
def render_sim_simple(simulation, num_particles, vid_size):
    """
    Args:
        simulation: torch.Tensor of shape [T, num_particles, 3]
        num_particles: int (number of particles)
        vid_size:

    Returns: [T, 3, size, size]

    """
    video = torch.zeros((simulation.shape[0], vid_size, vid_size))
    scaled_coords = (simulation[:, :, :2] * vid_size).long()
    scaled_coords[:, :, 1] = vid_size - scaled_coords[:, :, 1]  # Reverse y axis

    # TODO: Find another strategy to hide oob particles instead of clamping
    scaled_coords.clamp_(0, vid_size - 1)

    one, two = torch.tensor(1.0), torch.tensor(2.0)

    def render(vid_t, coords_t):
        vid_t[coords_t[:num_particles, 0], coords_t[:num_particles, 1]] = one
        vid_t[coords_t[num_particles:, 0], coords_t[num_particles:, 1]] = two
        return vid_t

    video = torch.vmap(render)(video, scaled_coords)

    colors = torch.tensor([[0, 0, 0], [248, 229, 89], [134, 74, 249]])

    video = video.unsqueeze(-1)
    video = torch.where(video == 1, colors[1], 0.0) + torch.where(
        video == 2, colors[2], 0.0
    )

    # Transpose for correct video display
    video = video.permute(0, 3, 2, 1)

    video = video.cpu().numpy().astype(np.uint8)

    return video


# TODO get colors from data parser or whatsoever
def animate_comparison(
    simulation,
    truth,
    type_,
    colors=None,
    interval=10,
    save_path=None,
    return_ani=False,
    as_array=False,
    bounds=((0.05, 0.95), (0.05, 0.95)),
    fps=25,
    show_progress=False,
    fig_base_size=6,
):
    # Truncate the simulation and truth to the same length
    simulation = simulation[:, : truth.shape[1]]
    type_ = type_[: truth.shape[1]]

    matplotlib.use("agg")
    fig_size_ratio = (bounds[1][1] - bounds[1][0]) / (bounds[0][1] - bounds[0][0])
    fig_size = (2 * fig_base_size, fig_base_size * fig_size_ratio)
    fig, axs = plt.subplots(1, 2, figsize=fig_size)

    for ax in axs.ravel():
        ax.set_xlim(bounds[0][0], bounds[0][1])
        ax.set_ylim(bounds[1][0], bounds[1][1])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect(1.0)

    wall_start = (type_ != 0.0).sum()

    pred = simulation[:, :wall_start]  # TODO correct types formatting

    material_types = np.unique(type_)  # material_type = type_[0]

    num_steps = pred.shape[0]
    gt_ax, pred_ax = axs

    material_types = [int(mt) for mt in material_types]

    if colors is None:
        colors = DEFAULT_COLORS[: len(material_types)]

    gt_points = {
        particle_type: gt_ax.plot([], [], "o", ms=2, color=colors[particle_type])[0]
        for particle_type in material_types
    }

    pred_points = {
        particle_type: pred_ax.plot([], [], "o", ms=2, color=colors[particle_type])[0]
        for particle_type in material_types
    }

    if show_progress:
        pbar = tqdm(total=num_steps, desc="Frames rendered", position=1, leave=False)

    def update(step_i):
        outputs = []
        gt_ax.set_ylabel("GT")
        pred_ax.set_ylabel("Pred")

        for particle_type, gt_line in gt_points.items():
            if particle_type not in material_types:
                continue
            indices = np.where(type_ == particle_type)[0]
            data = truth[step_i, indices, :2]
            gt_line.set_data(*data.T)

        for particle_type, pred_line in pred_points.items():
            if particle_type not in material_types:
                continue
            indices = np.where(type_ == particle_type)[0]
            data = simulation[step_i, indices, :2]
            pred_line.set_data(*data.T)

        if show_progress:
            pbar.update(1)

        outputs.append([gt_line, pred_line])
        return outputs

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

    ani = animation.FuncAnimation(fig, update, frames=num_steps, interval=interval)

    if save_path:
        if save_path.endswith(".gif"):
            ani.save(save_path, dpi=200, writer=animation.PillowWriter(fps=fps))
        elif save_path.endswith(".mp4"):
            ani.save(save_path, dpi=200, writer=animation.FFMpegWriter(fps=fps))
        plt.close(fig)
        del ani
    else:
        arr = []
        for i in range(num_steps):
            fig.canvas.draw()
            ani._draw_frame(i)
            X = np.array(fig.canvas.renderer._renderer)
            arr.append(X)

        arr = np.stack(arr, axis=0)
        arr = np.transpose(arr, axes=[0, -1, 1, 2])
        plt.close(fig)
        del ani
        return arr


# TODO if GT = False, then plot without the borders (like paper figures)
def animate_comparison_v2(
    simulation,
    truth,
    types,
    colors=None,
    interval=10,
    save_path=None,
    return_ani=False,
    as_array=False,
    bounds=((0.05, 0.95), (0.05, 0.95)),
    use_gradient=True,
    fps=25,
):
    dim = 2  # Can't render 3D here.

    interp = PchipInterpolator([0, 0.2, 0.95, 1.0], [0.3, 0.3, 1.0, 1.0])
    pos, _ = truth[..., :dim], truth[..., dim:]
    num_particles = (types != 0).sum().item()

    # TODO
    COLOR_MAPS = {
        1: ("Blues", 0.725, 0.95),
        2: ("YlOrBr", 0.2, 0.375),
        3: ("BuPu", 0.65, 0.925),
    }

    particles_types = np.unique(types[:num_particles])

    cmaps = []
    for p_type in particles_types:
        p_type = int(p_type.item())
        mat_info = COLOR_MAPS[p_type]

        num_p_type = (types[:num_particles] == p_type).sum().item()
        cmap = plt.get_cmap(mat_info[0])
        cmap = cmap(
            np.linspace(
                mat_info[1] if use_gradient else mat_info[2], mat_info[2], num_p_type
            )
        )
        cmap = np.concatenate([cmap, cmap[: num_p_type - len(cmap)]], axis=0)
        pos_ptype = pos[0, :num_particles][types[:num_particles] == p_type]
        sorted_pos = pos_ptype[:, 1].argsort().argsort()
        cmap = cmap[sorted_pos]
        cmaps.append(cmap)

    cmap = np.concatenate(cmaps, axis=0)

    bottom_left = (bounds[0][0], bounds[1][0])
    top_right = (bounds[0][1], bounds[1][1])
    line_width = 3

    pbar = tqdm(
        total=len(simulation),
        desc="Rendering",
        position=1,
        leave=False,
    )

    def update_plot(frame):
        plt.clf()

        # Truth Plot
        plt.subplot(1, 2, 1)
        kernel = stats.gaussian_kde(
            truth[frame, :num_particles].swapaxes(1, 0), bw_method="scott"
        )
        weights = kernel(truth[frame, :num_particles].swapaxes(1, 0))
        weights = interp(weights)
        new_cmap = cmap.copy()
        new_cmap[..., 3] = weights
        plt.scatter(
            truth[frame, :num_particles, 0],
            truth[frame, :num_particles, 1],
            c=new_cmap,
            edgecolors="none",
        )
        plt.plot(
            [bottom_left[0], bottom_left[0]],
            [bottom_left[1], top_right[1]],
            "black",
            linewidth=line_width,
        )
        plt.plot(
            [top_right[0], top_right[0]],
            [bottom_left[1], top_right[1]],
            "black",
            linewidth=line_width,
        )
        plt.plot(
            [bottom_left[0], top_right[0]],
            [bottom_left[1], bottom_left[1]],
            "black",
            linewidth=line_width,
        )
        plt.plot(
            [bottom_left[0], top_right[0]],
            [top_right[1], top_right[1]],
            "black",
            linewidth=line_width,
        )
        wall_pos = simulation[0, num_particles:, :dim]
        inside_walls = wall_pos[wall_pos[..., 0] > bottom_left[0]]
        inside_walls = inside_walls[inside_walls[..., 0] < top_right[0]]
        inside_walls = inside_walls[inside_walls[..., 1] > bottom_left[1]]
        inside_walls = inside_walls[inside_walls[..., 1] < top_right[1]]
        plt.scatter(
            inside_walls[:, 0],
            inside_walls[:, 1],
            c="black",
            alpha=1.0,
            s=30,
            edgecolors="black",
        )
        plt.axis("off")
        plt.xticks([])
        plt.yticks([])
        plt.xlim(bottom_left[0], top_right[0])
        plt.ylim(bottom_left[1], top_right[1])
        plt.title("Ground-Truth")

        # Simulation Plot
        plt.subplot(1, 2, 2)
        kernel = stats.gaussian_kde(
            simulation[frame, :num_particles].swapaxes(1, 0), bw_method="scott"
        )
        weights = kernel(simulation[frame, :num_particles].swapaxes(1, 0))
        weights = interp(weights)
        new_cmap = cmap.copy()
        new_cmap[..., 3] = weights
        plt.scatter(
            simulation[frame, :num_particles, 0],
            simulation[frame, :num_particles, 1],
            c=new_cmap,
            edgecolors="none",
        )
        plt.plot(
            [bottom_left[0], bottom_left[0]],
            [bottom_left[1], top_right[1]],
            "black",
            linewidth=line_width,
        )
        plt.plot(
            [top_right[0], top_right[0]],
            [bottom_left[1], top_right[1]],
            "black",
            linewidth=line_width,
        )
        plt.plot(
            [bottom_left[0], top_right[0]],
            [bottom_left[1], bottom_left[1]],
            "black",
            linewidth=line_width,
        )
        plt.plot(
            [bottom_left[0], top_right[0]],
            [top_right[1], top_right[1]],
            "black",
            linewidth=line_width,
        )
        wall_pos = simulation[0, num_particles:, :dim]
        inside_walls = wall_pos[wall_pos[..., 0] > bottom_left[0]]
        inside_walls = inside_walls[inside_walls[..., 0] < top_right[0]]
        inside_walls = inside_walls[inside_walls[..., 1] > bottom_left[1]]
        inside_walls = inside_walls[inside_walls[..., 1] < top_right[1]]
        plt.scatter(
            inside_walls[:, 0],
            inside_walls[:, 1],
            c="black",
            alpha=1.0,
            s=30,
            edgecolors="black",
        )
        plt.axis("off")
        plt.xticks([])
        plt.yticks([])
        plt.xlim(bottom_left[0], top_right[0])
        plt.ylim(bottom_left[1], top_right[1])
        plt.title("Predictions")

        pbar.update(1)

    duration = len(simulation)

    base_size = 6
    fig_size_ratio = (bounds[1][1] - bounds[1][0]) / (bounds[0][1] - bounds[0][0])
    fig_size = (2 * base_size, base_size * fig_size_ratio)
    fig = plt.figure(figsize=fig_size)
    ani = animation.FuncAnimation(fig, update_plot, frames=duration, interval=interval)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    if save_path.endswith(".gif"):
        ani.save(save_path, dpi=200, writer=animation.PillowWriter(fps=fps))
    elif save_path.endswith(".mp4"):
        ani.save(save_path, dpi=200, writer=animation.FFMpegWriter(fps=fps))


def render_cloud(
    cloud,
    types,
    save_path,
    colors=None,
    engine="v2",
    use_gradient=True,
    bounds=((0.05, 0.95), (0.05, 0.95)),
    no_walls=False,
):
    dim = 2
    pos = cloud[..., :dim]
    # bottom_left = [0.0962, 0.0962]
    # top_right = [0.9038, 0.9038]
    bottom_left = (bounds[0][0], bounds[1][0])
    top_right = (bounds[0][1], bounds[1][1])
    line_width = 3

    fig_size_ratio = (bounds[1][1] - bounds[1][0]) / (bounds[0][1] - bounds[0][0])
    fig_size = (6, 6 * fig_size_ratio)

    if engine == "v2":
        interp = PchipInterpolator([0, 0.2, 0.95, 1.0], [0.3, 0.3, 1.0, 1.0])

        num_particles = (types != 0).sum().item()

        COLOR_MAPS = {
            1: ("Blues", 0.725, 0.95),
            2: ("YlOrBr", 0.2, 0.375),
            3: ("BuPu", 0.65, 0.925),
        }

        particles_types = np.unique(types[:num_particles])

        cmaps = []
        for p_type in particles_types:
            p_type = int(p_type.item())
            mat_info = COLOR_MAPS[p_type]

            num_p_type = (types[:num_particles] == p_type).sum().item()
            cmap = plt.get_cmap(mat_info[0])
            cmap = cmap(
                np.linspace(
                    mat_info[1] if use_gradient else mat_info[2],
                    mat_info[2],
                    num_p_type,
                )
            )
            cmap = np.concatenate([cmap, cmap[: num_p_type - len(cmap)]], axis=0)
            pos_ptype = pos[:num_particles][types[:num_particles] == p_type]
            sorted_pos = pos_ptype[:, 1].argsort().argsort()
            cmap = cmap[sorted_pos]
            cmaps.append(cmap)

        cmap = np.concatenate(cmaps, axis=0)

        kernel = stats.gaussian_kde(
            pos[:num_particles].swapaxes(1, 0), bw_method="scott"
        )
        weights = kernel(pos[:num_particles].swapaxes(1, 0))
        weights = interp(weights)
        new_cmap = cmap.copy()
        new_cmap[..., 3] = weights

        plt.figure(figsize=fig_size)
        plt.scatter(
            pos[:num_particles, 0],
            pos[:num_particles, 1],
            c=new_cmap,
            edgecolors="none",
        )
        plt.plot(
            [bottom_left[0], bottom_left[0]],
            [bottom_left[1], top_right[1]],
            "black",
            linewidth=line_width,
        )
        plt.plot(
            [top_right[0], top_right[0]],
            [bottom_left[1], top_right[1]],
            "black",
            linewidth=line_width,
        )
        plt.plot(
            [bottom_left[0], top_right[0]],
            [bottom_left[1], bottom_left[1]],
            "black",
            linewidth=line_width,
        )
        plt.plot(
            [bottom_left[0], top_right[0]],
            [top_right[1], top_right[1]],
            "black",
            linewidth=line_width,
        )
        wall_pos = pos[num_particles:, :dim]
        inside_walls = wall_pos[wall_pos[..., 0] > bottom_left[0]]
        inside_walls = inside_walls[inside_walls[..., 0] < top_right[0]]
        inside_walls = inside_walls[inside_walls[..., 1] > bottom_left[1]]
        inside_walls = inside_walls[inside_walls[..., 1] < top_right[1]]
        plt.scatter(
            inside_walls[:, 0],
            inside_walls[:, 1],
            c="black",
            alpha=1.0,
            s=30,
            edgecolors="black",
        )
        plt.axis("off")
        plt.xticks([])
        plt.yticks([])
        plt.xlim(bottom_left[0], top_right[0])
        plt.ylim(bottom_left[1], top_right[1])
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
    elif "v1" in engine:
        plt.figure(figsize=fig_size)
        plt.axis("off")
        plt.xticks([])
        plt.yticks([])
        plt.xlim(bounds[0][0], bounds[0][1])
        plt.ylim(bounds[1][0], bounds[1][1])

        material_types = np.unique(types)  # material_type = type_[0]
        material_types = [int(mt) for mt in material_types]

        if no_walls:
            material_types.pop(0)

        gt_ax = plt.gca()

        if colors is None:
            colors = DEFAULT_COLORS[: len(material_types)]

        points = {
            particle_type: gt_ax.plot([], [], "o", ms=2, color=colors[particle_type])[0]
            for particle_type in material_types
        }

        for particle_type, gt_line in points.items():
            if particle_type not in material_types:
                continue
            indices = np.where(types == particle_type)[0]
            data = pos[indices, :2]
            gt_line.set_data(*data.T)

        if "thinwalls" in engine:
            plt.xlim(bottom_left[0], top_right[0])
            plt.ylim(bottom_left[1], top_right[1])
            plt.plot(
                [bottom_left[0], bottom_left[0]],
                [bottom_left[1], top_right[1]],
                "black",
                linewidth=line_width,
            )
            plt.plot(
                [top_right[0], top_right[0]],
                [bottom_left[1], top_right[1]],
                "black",
                linewidth=line_width,
            )
            plt.plot(
                [bottom_left[0], top_right[0]],
                [bottom_left[1], bottom_left[1]],
                "black",
                linewidth=line_width,
            )
            plt.plot(
                [bottom_left[0], top_right[0]],
                [top_right[1], top_right[1]],
                "black",
                linewidth=line_width,
            )

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0)


def plot_sph(A, B, C=None, r=0.015, bounds=((0.05, 0.95), (0.05, 0.95))):
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    ax.set_xlim(bounds[0][0], bounds[0][1])
    ax.set_ylim(bounds[1][0], bounds[1][1])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect(1.0)
    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)
    # ax.spines['bottom'].set_visible(False)
    # ax.spines['left'].set_visible(False)

    a_line = ax.plot([], [], "o", ms=2, color="black")[0]
    a_line.set_data(*A[..., :2].T)

    b_line = ax.plot([], [], "o", ms=2, color="blue")[0]
    b_line.set_data(*B[..., :2].T)

    if C is not None:
        circle = plt.Circle(C, r, color="r")
        ax.add_patch(circle)

    fig.subplots_adjust(
        left=0.01, bottom=0.01, right=0.99, top=0.99, wspace=0.01, hspace=0.01
    )
    fig.tight_layout()

    plt.show()
