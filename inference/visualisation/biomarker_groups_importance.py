from sklearn.decomposition import PCA
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def extract_group_inputs(model, data_batch, batch_size):
    """
    Extracts the inputs to the biomarker group deep sets.

    Args:
        model (nn.Module): The GMAN model.
        data_batch (dict): A batch of data.
        batch_size (int): The size of the batch.

    Returns:
        dict: A dictionary where keys are group indices and values are the corresponding group inputs.
    """
    group_inputs = {str(i): [] for i in range(len(model.biomarker_groups))}
    biom_idx_map = {}
    hidden_channels = model.hidden_channels

    # Get the outputs of the GNANs for each biomarker
    outputs = torch.zeros(size=(model.max_num_GNANs, batch_size, hidden_channels))
    for ind, (k, b) in enumerate(data_batch.items()):
        biom_idx_map[k] = ind
        x_batch = b['x_batch']
        dist_batch = b['dist_batch']
        batch_vector = b['batch_vector']
        gnan_output = model.gnans[ind](x_batch, dist_batch, batch_vector)[0]

        # Pad the output if it's smaller than the expected size
        expected_size = outputs[ind].shape[0]
        actual_size = gnan_output.shape[0]
        if actual_size < expected_size:
            pad_size = expected_size - actual_size
            pad_tensor = torch.zeros((pad_size, gnan_output.shape[1]))
            gnan_output = torch.cat([gnan_output, pad_tensor], dim=0)

        outputs[ind] = gnan_output

    outputs = outputs.permute(1, 0, 2)

    # Sum the outputs of the biomarkers in each group
    for group_idx, group in enumerate(model.biomarker_groups):
        group_out = []
        for biomarker in group:
            if biomarker in biom_idx_map:
                biom_idx = biom_idx_map[biomarker]
                group_out.append(outputs[:, biom_idx])

        if group_out and len(group_out) > 1:
            group_sum = torch.stack(group_out, dim=0).sum(dim=0)
            group_inputs[str(group_idx)].append(group_sum.detach().cpu().numpy())
        elif group_out and len(group_out) == 1:
            group_inputs[str(group_idx)].append(group_out[0].detach().cpu().numpy())

    # Concatenate the group inputs
    for k in group_inputs:
        if group_inputs[k]:
            group_inputs[k] = np.vstack(group_inputs[k])
        else:
            group_inputs[k] = np.array([])

    return group_inputs

def plot_patient_vs_control_group_scores(
    patient_scores, control_scores, group_names, save_path=None, writer=None, global_step=0,
    patient_color='steelblue', control_color='salmon', patient_title='Group Contributions: Patients',
    control_title='Group Contributions: Controls', main_title='Group-Level Contributions by Class'
):
    """
    Plots the group contribution scores for patients and controls.

    Args:
        patient_scores (np.ndarray): The group contribution scores for patients.
        control_scores (np.ndarray): The group contribution scores for controls.
        group_names (list): A list of group names.
        save_path (str, optional): The path to save the plot.
        writer (SummaryWriter, optional): A TensorBoard SummaryWriter to log the plot.
        global_step (int, optional): The global step for TensorBoard logging.
        patient_color (str, optional): The color for the patient bars.
        control_color (str, optional): The color for the control bars.
        patient_title (str, optional): The title for the patient subplot.
        control_title (str, optional): The title for the control subplot.
        main_title (str, optional): The main title for the plot.
    """
    x = np.arange(len(group_names))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4), sharey=True)


    bars1 = ax1.bar(x, patient_scores, color=patient_color)
    ax1.set_title(patient_title)
    ax1.set_xticks(x)
    ax1.set_xticklabels(group_names, rotation=30, ha='right')
    ax1.set_ylabel("Average Contribution")
    ax1.grid(axis='y')


    bars2 = ax2.bar(x, control_scores, color=control_color)
    ax2.set_title(control_title)
    ax2.set_xticks(x)
    ax2.set_xticklabels(group_names, rotation=30, ha='right')
    ax2.grid(axis='y')

    fig.suptitle(main_title, fontsize=14)
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))

    if save_path:
        fig.savefig(f"{save_path}/groupwise_contributions_by_class.png")

    if writer:
        writer.add_figure("group_contributions_split", fig, global_step=global_step)

    plt.close(fig)

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

    # Get the baseline prediction
    with torch.no_grad():
        baseline_output = model(data_batch, batch_size)
        if isinstance(baseline_output, tuple):
            baseline_output = baseline_output[0]
        baseline_output = baseline_output.detach().cpu().numpy().mean()

    x_range_dict = {}
    delta_preds = {}

    # For each group, sweep over a range of input values and record the change in prediction
    for group_idx in tqdm(group_indices):
        biom_group = model.biomarker_groups[group_idx]

        # Get the raw inputs for the group
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

        # Use PCA to project the inputs to a 1D space
        pca = PCA(n_components=1)
        x_proj = pca.fit_transform(group_raw_inputs)
        mean = x_proj.mean()
        std = x_proj.std()
        x_range = np.linspace(mean - 2 * std, mean + 2 * std, n_points)
        recon_inputs = pca.inverse_transform(x_range.reshape(-1, 1)) 
        x_range_dict[group_idx] = x_range

        # Sweep over the reconstructed inputs and record the change in prediction
        group_deltas = []
        for sweep_vec in recon_inputs:
            modified_batch = {}


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

    # Plot the results
    fig, ax = plt.subplots(figsize=(8, 4))
    for group_idx in group_indices:
        if group_idx in delta_preds:
            label = group_names[group_idx] if group_names else f"Group {group_idx}"
            ax.plot(x_range_dict[group_idx], delta_preds[group_idx], label=label)

    ax.set_title("Prediction Sensitivity to Group Input (Δ from baseline)")
    ax.set_xlabel("PCA-1 of Group Input (raw features)")
    ax.set_ylabel("Δ Prediction")
    ax.axhline(0, color='gray', linestyle='--')
    ax.legend()
    ax.grid(True)
    fig.tight_layout()

    if writer:
        writer.add_figure("groupwise_prediction_sensitivity", fig, global_step=global_step)

    fig.savefig(f"{save_path}/groupwise_prediction_sensitivity_{global_step}.png")

    plt.close(fig)
