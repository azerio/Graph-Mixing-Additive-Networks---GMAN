import os
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D
import torch
import math
import pandas as pd


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
        _, _, node_embeddings = gnan(x, dist, batch_vector, return_node_embeddings=True)
        node_contributions = node_embeddings.sum(dim=1)  # shape: [num_nodes]
    return node_contributions.cpu()

def plot_all_biomarkers_as_combined_graph(
    model,
    individual_graphs,
    colour_map,
    feature_names_dict,
    group_names,
    biomarker_name_to_code,
    plot_path
):
    """
    Plots the node importances for all biomarkers as a single combined graph.

    Args:
        model (nn.Module): The GMAN model.
        individual_graphs (dict): A dictionary of individual graphs.
        colour_map (dict): A dictionary mapping biomarker names to colors.
        feature_names_dict (dict): A dictionary mapping biomarker names to feature names.
        group_names (list): A list of group names.
        biomarker_name_to_code (dict): A dictionary mapping biomarker names to codes.
        plot_path (str): The path to save the plot.
    """
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)

    G = nx.DiGraph()
    node_colors = []
    node_sizes = []
    node_labels = {}
    global_node_idx = 0

    prediag_date = individual_graphs[list(individual_graphs.keys())[0]]["prediag_date"]

    all_contributions = []
    all_contributions_by_biomarker = {}

    # Compute node contributions for each biomarker
    for biomarker_name in group_names:
        if biomarker_name not in biomarker_name_to_code:
            continue
        biomarker_code = biomarker_name_to_code[biomarker_name]
        if biomarker_code not in individual_graphs:
            continue

        gnan_idx = model.biomarker_to_gnan[biomarker_code]
        gnan = model.gnans[gnan_idx]
        graph_data = individual_graphs[biomarker_code]

        graph_data_for_gnan = {
            "x_batch": graph_data["x_batch"].to(model.device),
            "dist_batch": graph_data["dist_batch"].to(model.device),
            "batch_vector": graph_data["batch_vector"].to(model.device)
        }

        node_contribs = compute_node_contributions_for_graph(gnan, graph_data_for_gnan)
        all_contributions.append(node_contribs)
        all_contributions_by_biomarker[biomarker_name] = node_contribs

    all_contribs_flat = torch.cat(all_contributions)
    global_min = all_contribs_flat.min()
    global_max = all_contribs_flat.max()

    present_biomarkers = set()

    # Create the graph
    for biomarker_name in group_names:
        if biomarker_name not in biomarker_name_to_code:
            continue

        biomarker_code = biomarker_name_to_code[biomarker_name]
        if biomarker_code not in individual_graphs:
            continue

        graph_data = individual_graphs[biomarker_code]
        x = graph_data["x_batch"].cpu()

        node_contribs = all_contributions_by_biomarker[biomarker_name]
        q_low = torch.quantile(all_contribs_flat, 0.05)
        q_high = torch.quantile(all_contribs_flat, 0.95)
        normalized = (node_contribs - q_low) / (q_high - q_low + 1e-6)

        normalized = normalized.clamp(0, 1)

        color = colour_map.get(biomarker_name, "#999999")
        label = feature_names_dict.get(biomarker_name, "unknown")

        present_biomarkers.add(biomarker_name)

        local_nodes = []
        for i in range(x.size(0)):
            G.add_node(global_node_idx)
            node_colors.append(color)

            base_size = 10
            # scale_factor = 5
            importance = float(abs(node_contribs[i].item()))
            scaled_size = base_size + importance
            node_sizes.append(scaled_size)

            node_labels[global_node_idx] = label
            local_nodes.append(global_node_idx)
            global_node_idx += 1

        for u, v in zip(local_nodes[:-1], local_nodes[1:]):
            G.add_edge(u, v)
            
    # Position the nodes
    pos = {}
    y_spacing = 3.0
    biomarker_index = 0
    node_cursor = 0
    top_padding = 1.0

    for biomarker_name in group_names:
        if biomarker_name not in biomarker_name_to_code:
            continue
        biomarker_code = biomarker_name_to_code[biomarker_name]
        if biomarker_code not in individual_graphs:
            continue

        graph_data = individual_graphs[biomarker_code]
        x_batch = graph_data["x_batch"]
        sampling_dates = graph_data["sampling_dates"]

        time_from_diag = [
            (pd.to_datetime(d) - pd.to_datetime(prediag_date)).days / 365.25
            for d in sampling_dates
        ]

        y = -(biomarker_index + top_padding) * y_spacing
        for x_pos in time_from_diag:
            pos[node_cursor] = (x_pos, y)
            node_cursor += 1

        biomarker_index += 1

    # Plot the graph
    plt.figure(figsize=(14, 10))
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

    ax = plt.gca()

    # Draw the timeline
    arrow_y = -len(group_names) * y_spacing
    arrow_xmin = min([p[0] for p in pos.values()])
    arrow_xmax = max([p[0] for p in pos.values()]) + 0.5

    ax.text(
        arrow_xmax, arrow_y + 0.3,
        "0y – Diagnosis Date",
        fontsize=12, rotation=45,
        ha='left', va='bottom'
    )

    arrow = patches.FancyArrowPatch(
        (arrow_xmin, arrow_y), (arrow_xmax, arrow_y),
        arrowstyle='->',
        mutation_scale=15,
        linewidth=1.5,
        color='black'
    )
    ax.add_patch(arrow)

    tick_y = arrow_y - 0.5
    label_y = arrow_y - 1.2

    for biomarker_name in group_names:
        if biomarker_name not in biomarker_name_to_code:
            continue
        biomarker_code = biomarker_name_to_code[biomarker_name]
        if biomarker_code not in individual_graphs:
            continue

        sampling_dates = individual_graphs[biomarker_code]["sampling_dates"]
        rel_times = [
            (pd.to_datetime(d) - pd.to_datetime(prediag_date)).days / 365.25
            for d in sampling_dates
        ]

        tick_y = arrow_y - 0.5
        label_y = arrow_y - 1.2

        timeline_start = arrow_xmin
        timeline_end = arrow_xmax
        tick_positions = torch.linspace(timeline_start, timeline_end, steps=5)

        for tick_x in tick_positions:
            ax.plot([tick_x, tick_x], [arrow_y, tick_y], color='black', linewidth=0.5)

            years_to_diag = tick_x.item() - 0  # timeline is already in years relative to diagnosis
            ax.text(
                tick_x.item(), label_y,
                f"{years_to_diag:.0f}y",
                ha='center', va='top',
                fontsize=12
            )

    # Add the legend
    legend_handles = [
        Line2D([0], [0], marker='o', color='w',
               markerfacecolor=colour_map[b], markersize=10,
               label=feature_names_dict.get(b, "unknown"))
        for b in present_biomarkers
    ]
    n_legend_cols = math.ceil(len(legend_handles) / 2)
    plt.legend(
        handles=legend_handles,
        loc='upper center',
        bbox_to_anchor=(0.5, 1.12),
        ncol=n_legend_cols,
        frameon=False,
        fontsize=12,
        handletextpad=0.5,
        columnspacing=1.0
    )

    plt.axis("off")
    plt.savefig(f"{plot_path}.png", dpi=300, bbox_inches="tight")
    plt.savefig(f"{plot_path}.svg", format="svg", bbox_inches="tight")
    plt.close()