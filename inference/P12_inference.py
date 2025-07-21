import numpy as np
import torch
import os
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import sys
import yaml
from sklearn.decomposition import PCA
import math
import networkx as nx
import matplotlib.patches as patches
from matplotlib.lines import Line2D
import pandas as pd

from model.utils import OneHotEmbedder
from data.loaders.physionet_2012_dataset import PhysioNet2012
from data.collate_fns.GMAN.physionet import distance_collate_fn_P12
from model.GMAN import GMAN
from config import get_config

def horizontal_plot_patient_vs_control_group_scores(
    patient_scores, control_scores, group_names, save_path=None, writer=None, global_step=0
):
    """
    Plots the group contribution scores for patients and controls horizontally.

    Args:
        patient_scores (np.ndarray): The group contribution scores for patients.
        control_scores (np.ndarray): The group contribution scores for controls.
        group_names (list): A list of group names.
        save_path (str, optional): The path to save the plot.
        writer (SummaryWriter, optional): A TensorBoard SummaryWriter to log the plot.
        global_step (int, optional): The global step for TensorBoard logging.
    """
    
    x = np.arange(len(group_names))
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(12, 4))


    ax1.barh(x, patient_scores, color="steelblue")
    ax1.set_title("Group Contributions: Dead")
    ax1.set_yticks(x)
    ax1.set_yticklabels(group_names, ha='right')
    ax1.set_xlabel("Average Contribution")
    ax1.grid(axis='y')


    ax2.barh(x, control_scores, color="salmon")
    ax2.set_title("Group Contributions: Survived")
    ax2.set_yticks(x)
    ax2.set_yticklabels(group_names, ha='right')
    ax2.grid(axis='y')


    fig.tight_layout(rect=(0, 0.03, 1, 0.95))


    for ax in (ax1, ax2):
        ax.margins(y=0)
    fig.canvas.draw()
    yticks = ax1.get_yticklabels()
    maxh = max(label.get_window_extent().height for label in yticks)


    m = 0.2  # inch margin top & bottom
    base_height = (maxh/fig.dpi) * len(group_names) + 2*m


    height = 2.0 * base_height


    orig_width = fig.get_size_inches()[0]
    width = 3.0 * orig_width


    vert_margin = m / height
    fig.subplots_adjust(top=1-vert_margin, bottom=vert_margin)
    fig.set_size_inches(width, height)

    # Save or log
    if save_path:
        fig.savefig(f"{save_path}/groupwise_contributions_by_class.png", dpi=600, bbox_inches='tight')
    if writer:
        writer.add_figure("group_contributions_split", fig, global_step=global_step)

    return fig

def compute_node_contributions_for_graph(gnan, graph_data):
    """
    Computes the node contributions for a single graph.

    Args:
        gnan (nn.Module): The GNAN model.
        graph_data (dict): A dictionary of graph data.

    Returns:
        torch.Tensor: A tensor of node contributions.
    """
    x = graph_data["x_batch"].to(gnan.device)
    dist = graph_data["dist_batch"].to(gnan.device)
    batch_vector = graph_data["batch_vector"].to(gnan.device)


    with torch.no_grad():

        node_embeddings = gnan(x, dist, batch_vector)

        min_val = node_embeddings.min()
        max_val = node_embeddings.max()
        # print(min_val, max_val)
        node_contributions = (node_embeddings - min_val) / (max_val - min_val + 1e-6)

    return node_contributions.cpu()

def draw_single_biomarker_graph(
    model,
    biomarker_name,
    graph_data,
    one_hot_to_color,
    feature_names_dict,
    group_names,
    plot_path,
    x_scaling: float = 1.0,    # scale factor for dist_batch gaps
    min_gap: float = 1.0,      # minimum data‐unit gap between nodes
    base_size: float = 30,     # minimum node size
    node_size_scale: float = 500  # scales normalized contribution → size
):
    """
    Draws a single biomarker graph.

    Args:
        model (nn.Module): The GMAN model.
        biomarker_name (str): The name of the biomarker.
        graph_data (dict): A dictionary of graph data.
        one_hot_to_color (dict): A dictionary mapping one-hot encodings to colors.
        feature_names_dict (dict): A dictionary mapping one-hot encodings to feature names.
        group_names (list): A list of group names.
        plot_path (str): The path to save the plot.
        x_scaling (float, optional): The scaling factor for the distance between nodes.
        min_gap (float, optional): The minimum gap between nodes.
        base_size (float, optional): The base size for the nodes.
        node_size_scale (float, optional): The scaling factor for the node sizes.
    """

    gnan = model.gnans[model.biomarker_to_gnan[biomarker_name]]
    data_for_contrib = {
        "x_batch":    graph_data["x_batch"].to(model.device),
        "dist_batch": graph_data["dist_batch"].to(model.device),
        "batch_vector": graph_data["batch_vector"].to(model.device),
    }
    contribs = compute_node_contributions_for_graph(gnan, data_for_contrib) \
               .sum(-1).cpu().numpy()


    dist = graph_data["dist_batch"].cpu().numpy()
    n = dist.shape[0]
    times = [0.0]
    for i in range(1, n):
        raw_gap = float(dist[i, i-1]) * x_scaling
        gap = max(raw_gap, min_gap)
        times.append(times[-1] + gap)


    G = nx.DiGraph()
    G.add_nodes_from(range(n))
    if n > 1:
        G.add_edges_from((i, i+1) for i in range(n-1))


    pos = {i: (times[i], 0.0) for i in range(n)}


    x_np = graph_data["x_batch"].cpu().numpy()
    onehot_dim = len(group_names)
    node_keys = []
    node_colors = []
    for i in range(n):
        onehot = x_np[i, -onehot_dim:].astype(int).tolist()
        key = ",".join(map(str, onehot))
        node_keys.append(key)
        node_colors.append(one_hot_to_color.get(key, "#999999"))

    mn, mx = contribs.min(), contribs.max()
    norm = (contribs - mn) / (mx - mn + 1e-6)
    sizes = base_size + (norm**0.5) * node_size_scale


    fig, ax = plt.subplots(figsize=(10, 3), dpi=300)
    nx.draw_networkx_nodes(G, pos,
        node_color=node_colors,
        node_size=sizes,
        alpha=0.9,
        ax=ax
    )
    nx.draw_networkx_edges(G, pos,
        edge_color="black",
        arrows=True,
        arrowstyle='-|>',
        arrowsize=12,
        width=2,
        ax=ax
    )


    unique_keys = []
    for k in node_keys:
        if k not in unique_keys:
            unique_keys.append(k)
    state_handles = [
        Line2D([0], [0], marker='o', color='w',
               markerfacecolor=one_hot_to_color[k],
               markersize=8,
               label=feature_names_dict.get(k, k))
        for k in unique_keys
    ]
    legend1 = ax.legend(handles=state_handles,
                        loc='upper left', title="State")


    bio_handle = Line2D([0], [0], linestyle='',
                        marker='', color='none',
                        label=biomarker_name)
    ax.add_artist(legend1)
    ax.legend(handles=[bio_handle],
              loc='upper right', title="Biomarker")


    ax.set_axis_off()
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')

    return plt

def plot_all_biomarkers_as_combined_graph(
    model,
    individual_graphs,
    colour_map,
    feature_names_dict,
    group_names,
    biomarker_name_to_code,
    plot_path="outputs/combined_node_importance.png",
    x_spacing=5.0,
    y_spacing=3.0,
    gli=0
):
    """
    Plots all biomarkers as a combined graph.

    Args:
        model (nn.Module): The GMAN model.
        individual_graphs (dict): A dictionary of individual graphs.
        colour_map (dict): A dictionary mapping biomarker names to colors.
        feature_names_dict (dict): A dictionary mapping biomarker names to feature names.
        group_names (list): A list of group names.
        biomarker_name_to_code (dict): A dictionary mapping biomarker names to codes.
        plot_path (str, optional): The path to save the plot.
        x_spacing (float, optional): The spacing between nodes on the x-axis.
        y_spacing (float, optional): The spacing between nodes on the y-axis.
        gli (int, optional): The global step for saving the plot.
    """
    out_dir = os.path.dirname(plot_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    G = nx.DiGraph()
    node_colors = []
    node_sizes = []
    node_labels = {}
    pos = {}
    global_node_idx = 0
    all_x_positions = []
    all_durations = []
    used_biomarkers = set()

    contrib_dict = {}
    all_contribs = []

    for idx, biomarker_name in enumerate(group_names):
        code = biomarker_name_to_code.get(biomarker_name)
        if code is None or code not in individual_graphs:
            continue

        gnan = model.gnans[model.biomarker_to_gnan[code]]
        data = individual_graphs[code]
        graph_kwargs = {
            "x_batch": data["x_batch"].to(model.device),
            "dist_batch": data["dist_batch"].to(model.device),
            "batch_vector": data["batch_vector"].to(model.device),
        }

        contribs = compute_node_contributions_for_graph(gnan, graph_kwargs).sum(-1).cpu().numpy()
        contrib_dict[code] = contribs
        all_contribs.extend(contribs)
        used_biomarkers.add(biomarker_name)

    global_min = np.min(all_contribs)
    global_max = np.max(all_contribs)


    for idx, biomarker_name in enumerate(group_names):
        code = biomarker_name_to_code.get(biomarker_name)
        if code is None or code not in individual_graphs:
            continue

        data = individual_graphs[code]
        contribs = contrib_dict[code]
        norm = (contribs - global_min) / (global_max - global_min + 1e-6)

        dist_matrix = data["dist_batch"].cpu().numpy()
        n = data["x_batch"].shape[0]
        dist_matrix = recover_original_time_deltas(dist_matrix)

        timestamps = [0.0]
        for i in range(1, n):
            timestamps.append(float(dist_matrix[i, 0]))

        total_duration = timestamps[-1]
        all_durations.append(total_duration)
        x_positions = [-(total_duration - t) * x_spacing for t in timestamps]
        all_x_positions.extend(x_positions)

        y = -idx * y_spacing
        node_indices = []
        color = colour_map.get(biomarker_name, "#999999")
        label = feature_names_dict.get(biomarker_name, biomarker_name)

        for i, x_pos in enumerate(x_positions):
            G.add_node(global_node_idx)
            pos[global_node_idx] = (float(x_pos), float(y))
            node_colors.append(color)
            # size = 30 + (float(norm[i]) ** 0.5) * 500
            size = 10 + float(contribs[i]) * 30
            node_sizes.append(size)
            node_labels[global_node_idx] = label
            node_indices.append(global_node_idx)
            global_node_idx += 1

        for u, v in zip(node_indices[:-1], node_indices[1:]):
            G.add_edge(u, v)


    plt.figure(figsize=(16, 10))
    ax = plt.gca()
    nx.draw(
        G, pos,
        node_color=node_colors,
        node_size=node_sizes,
        edge_color="gray",
        with_labels=False,
        alpha=0.85,
        arrows=True,
        arrowstyle='-|>',
        arrowsize=10
    )


    handles = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=colour_map.get(name, "#999999"),
               markersize=10, label=feature_names_dict.get(name, name))
        for name in sorted(used_biomarkers)
    ]
    n_legend_cols = math.ceil(len(handles) / 2)
    plt.legend(
        handles=handles,
        loc='upper center',
        bbox_to_anchor=(0.5, 1.12),
        ncol=n_legend_cols,
        frameon=False,
        fontsize=12,
        handletextpad=0.5,
        columnspacing=1.0
    )

    # Timeline
    if all_x_positions and all_durations:
        timeline_y = min(y for _, y in pos.values()) - y_spacing
        x_min, x_max = min(all_x_positions), max(all_x_positions)
        duration = max(all_durations)
        tick_labels = [f"{int(t)}" for t in np.linspace(-duration, 0, 5)]
        tick_xs = np.linspace(x_min, x_max, 5)

        ax.annotate(
            '', xy=(x_max + 5, timeline_y), xytext=(x_min, timeline_y),
            arrowprops=dict(arrowstyle='->', lw=2.0, color='black')
        )

        for tick, label in zip(tick_xs, tick_labels):
            ax.vlines(tick, timeline_y - 0.3, timeline_y + 0.3, colors='black')
            ax.text(tick, timeline_y - 0.6, label, ha='center', va='top', fontsize=10)

    # plt.title("Combined Node Importance Across Biomarker Graphs")
    plt.axis("off")
    plt.tight_layout()
    plt.subplots_adjust(right=0.75)
    plt.savefig(f"{plot_path}/{gli}.png", dpi=300, bbox_inches="tight")
    plt.savefig(f"{plot_path}/{gli}.svg", dpi=300, bbox_inches="tight")
    return plt

def recover_original_time_deltas(distances):
    """
    Recovers the original time deltas from the distance matrix.

    Args:
        distances (np.ndarray): The distance matrix.

    Returns:
        np.ndarray: The time deltas.
    """
    with np.errstate(divide='ignore'):
        inverse = 1 / distances
        time_deltas = inverse - 0.1
        time_deltas[np.isinf(inverse)] = 0.0
    return time_deltas
    
def plot_groupwise_prediction_curves(
    model,
    data_batch,
    batch_size,
    group_indices,
    group_names=None,
    n_points=100,
    writer=None,
    global_step=0,
    save_path=None
):
    """
    Plots the prediction sensitivity curves for each biomarker group.

    Args:
        model (nn.Module): The GMAN model.
        data_batch (dict): A batch of data.
        batch_size (int): The size of the batch.
        group_indices (list): A list of group indices to plot.
        group_names (list, optional): A list of group names.
        n_points (int, optional): The number of points to use for the prediction curves.
        writer (SummaryWriter, optional): A TensorBoard SummaryWriter to log the plot.
        global_step (int, optional): The global step for TensorBoard logging.
        save_path (str, optional): The path to save the plot.
    """
    model.eval()


    print("Computing baseline prediction...")
    with torch.no_grad():
        print(batch_size)

        baseline_output = model(data_batch, batch_size)
        if isinstance(baseline_output, tuple):
            baseline_output = baseline_output[0]
        baseline_output = baseline_output.detach().cpu().numpy().mean()

    x_range_dict = {}
    delta_preds = {}

    print("Computing groupwise prediction curves...")
    for group_idx in tqdm(group_indices):
        biom_group = model.biomarker_groups[group_idx]


        group_raw_inputs = []
        for biom in biom_group:
            if biom not in data_batch:
                continue
            x = data_batch[biom]["x_batch"]
            group_raw_inputs.append(x)

        if not group_raw_inputs:
            print(f"Group {group_idx} has no data.")
            continue


        group_raw_inputs = torch.cat(group_raw_inputs, dim=0).detach().cpu().numpy()

        if group_raw_inputs.shape[0] < 2:
            print(f"Group {group_idx} has too few samples for PCA.")
            continue

        print("Running PCA on group raw inputs...")

        pca = PCA(n_components=1)
        x_proj = pca.fit_transform(group_raw_inputs)
        mean = x_proj.mean()
        std = x_proj.std()
        x_range = np.linspace(mean - 2 * std, mean + 2 * std, n_points)
        recon_inputs = pca.inverse_transform(x_range.reshape(-1, 1))
        x_range_dict[group_idx] = x_range


        print(f"Computing predictions for group {group_idx}...")
        group_deltas = []
        for sweep_vec in recon_inputs:
            modified_batch = {}

            # Deep copy original batch
            for biom, data_dict in data_batch.items():
                modified_batch[biom] = {}
                for key, val in data_dict.items():
                    modified_batch[biom][key] = val.clone()

            sweep_tensor = torch.tensor(sweep_vec, dtype=torch.float32)

            for biom in biom_group:
                if biom in modified_batch:
                    batch_vec = modified_batch[biom]["batch_vector"]
                    # Inject sweep value as raw GNAN input
                    modified_batch[biom]["x_batch"] = sweep_tensor.repeat(batch_vec.shape[0], 1)

            with torch.no_grad():
                pred = model(modified_batch, batch_size)
                if isinstance(pred, tuple):
                    pred = pred[0]
                pred = pred.detach().cpu().numpy().mean()
                delta = pred - baseline_output
                group_deltas.append(delta)

        delta_preds[group_idx] = group_deltas

    print("Finished computing groupwise prediction curves.")

    fig = plt.figure(figsize=(12, 6))
    gs = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[0.4, 2.5])
    ax_title = fig.add_subplot(gs[0])
    ax = fig.add_subplot(gs[1])


    ax_title.set_title("Prediction Sensitivity to Group Input (Δ from baseline)", pad=20, fontsize=14)
    ax_title.axis("off")  # Hide axes for title area


    legend_handles = []
    legend_labels = []
    for group_idx in group_indices:
        if group_idx in delta_preds:
            label = group_names[group_idx] if group_names else f"Group {group_idx}"
            x_vals = x_range_dict[group_idx]
            y_vals = delta_preds[group_idx]
            line, = ax.plot(x_vals, y_vals, label=label)
            legend_handles.append(line)
            legend_labels.append(label)


    ncols = math.ceil(len(legend_labels) / 3)
    fig.legend(
        handles=legend_handles,
        labels=legend_labels,
        loc='upper center',
        bbox_to_anchor=(0.5, 1.05),
        ncol=ncols,
        frameon=False,
        fontsize=8
    )

    fig.tight_layout(rect=(0, 0, 1, 1))

    # Plot axis
    ax.set_xlabel("PCA-1 of Group Input (raw features)")
    ax.set_ylabel("Δ Prediction")
    ax.axhline(0, color='gray', linestyle='--')
    ax.grid(True)


    # if save_path:
    all_x_vals = np.concatenate([x_range_dict[g] for g in delta_preds if g in x_range_dict])


    mean = all_x_vals.mean()
    std = all_x_vals.std()
    ax.set_xlim(mean - 3 * std, mean + 3 * std)

    fig.savefig(f"{save_path}/groupwise_prediction_sensitivity_{global_step}.png")

    plt.close(fig)

def main():
    exp_config = get_config()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    biomarker_groups = exp_config.biomarker_groups
    group_names = [biom[0] for biom in biomarker_groups]
    biomarker_name_to_code = dict(zip(group_names, group_names))
    cmap = plt.get_cmap('tab20')
    colour_map = {name: cmap(i % cmap.N) for i, name in enumerate(group_names)}
    feature_names_dict = {name: name for name in group_names}
    one_hot_to_color = {}
    for idx, name in enumerate(group_names):
        onehot = [1 if i==idx else 0 for i in range(len(group_names))]
        key = ",".join(map(str, onehot))
        one_hot_to_color[key]    = colour_map[name]
        feature_names_dict[key]  = name

    # Load Data
    folder = exp_config.tmp_data_dir
    neg_files = []
    pos_files = []

    for fname in os.listdir(folder):
        if not fname.endswith('.psv'):
            continue
        path = os.path.join(folder, fname)

        surv = pd.read_csv(path, sep='|', usecols=['Survival'])['Survival']

        if surv.eq(1).any():
            pos_files.append(os.path.join('tmp', fname))
        else:
            neg_files.append(os.path.join('tmp', fname))

    one_hot_embedder = OneHotEmbedder(input_dim=exp_config.num_biom, output_dim=exp_config.num_biom_embed)
    pos_dataset = PhysioNet2012(files=pos_files, config=exp_config, biom_one_hot_embedder=one_hot_embedder)
    neg_dataset = PhysioNet2012(files=neg_files, config=exp_config, biom_one_hot_embedder=one_hot_embedder)
    
    patient_dataloader = DataLoader(pos_dataset, batch_size=exp_config.batch_size, num_workers=16, shuffle=False, collate_fn=distance_collate_fn_P12)
    control_dataloader = DataLoader(neg_dataset, batch_size=exp_config.batch_size, num_workers=16, shuffle=False, collate_fn=distance_collate_fn_P12)

    # Load Model
    model = GMAN(config=exp_config).to(device)
    model.load_state_dict(torch.load(exp_config.model_path, map_location=device)["model_state_dict"])

    # Run Inference and Visualization
    patient_dataloader_single = DataLoader(pos_dataset, batch_size=1, num_workers=0, shuffle=True, collate_fn=distance_collate_fn_P12)
    for i in range(100):
        while True:
            data_patient = next(iter(patient_dataloader_single))
            individual_graphs = data_patient[0]
            total_nodes = sum(len(g["batch_vector"]) for g in individual_graphs.values())
            if total_nodes >= 10 * 3 and total_nodes <= 10 * 8:
                break

        plot_all_biomarkers_as_combined_graph(
            model=model,
            individual_graphs=individual_graphs,
            colour_map=colour_map,
            feature_names_dict=feature_names_dict,
            group_names=group_names,
            biomarker_name_to_code=biomarker_name_to_code,
            plot_path=exp_config.plot_path,
            gli=i,
        )

    patient_dataloader_batched = DataLoader(pos_dataset, batch_size=32, num_workers=0, shuffle=True, collate_fn=distance_collate_fn_P12)
    for i, data_patient in enumerate(tqdm(patient_dataloader_batched)):
        plot_groupwise_prediction_curves(
            model=model,
            data_batch=data_patient[0],
            batch_size=32,
            group_indices=range(len(exp_config.biomarker_groups)),
            group_names=exp_config.group_names,
            n_points=100,
            writer=None,
            global_step=i,
            save_path=exp_config.plot_path
        )

if __name__ == '__main__':
    main() 