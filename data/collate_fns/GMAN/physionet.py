import itertools
import torch

def distance_collate_fn_physionet(batch):
    keys = list(itertools.chain(*[list(x.keys()) for (x, _) in batch]))
    collated_batch = {key: [] for key in keys}
    out = {key: {} for key in keys}

    y_batch = []

    for s_idx, (s, graph_label) in enumerate(batch):
        y_batch.append(graph_label)
        for k, v in s.items():
            collated_batch[k].append(v)

    for k, v in collated_batch.items():
        num_nodes_list = []
        x_batch = []
        dists = []
        batch_vector = []

        for x in v:
            num_nodes_list.append(x[0].shape[0])
            x_batch.append(x[0])
            dists.append(x[1])

        x_batch = torch.cat(x_batch, dim=0)

        total_nodes = sum(num_nodes_list)
        dist_batch = torch.full((total_nodes, total_nodes), fill_value=-1, dtype=torch.float)

        start_idx = 0
        for dist_mat in dists:
            n = dist_mat.shape[0]
            dist_batch[start_idx:start_idx + n, start_idx:start_idx + n] = dist_mat
            start_idx += n

        for graph_idx, n_nodes in enumerate(num_nodes_list):
            batch_vector.append(torch.full((n_nodes,), fill_value=graph_idx, dtype=torch.long))
        batch_vector = torch.cat(batch_vector, dim=0)

        out[k]["x_batch"] = x_batch
        out[k]["dist_batch"] = dist_batch
        out[k]["batch_vector"] = batch_vector

    y_batch = torch.cat(y_batch, dim=0)
    return out, y_batch