import torch

def distance_collate_fn_fakenews(batch):
    """
    Collate for FakeNews: batch of (s_dict, graph_label), s_dict={'propagate': node_dict}.
    Splits components into 'single' (size=1) and 'not single' (size>1).
    Returns out dict with two keys, and y_batch.
    """
    single_items = []
    multi_items = []
    y_list = []
    graph_node_counts = []
    for g_idx, (s_dict, graph_label) in enumerate(batch):
        y_list.append(graph_label)
        node_dict = s_dict['propagate']
        # for each component (root or son-subtree)
        for x, dist in node_dict.values():
            if x.shape[0] == 1:
                single_items.append((x, dist, g_idx))
            else:
                multi_items.append((x, dist, g_idx))
    def batch_components(items):
        # items: list of (x, dist, graph_idx)
        counts = [x.shape[0] for x, _, _ in items]
        x_batch = torch.cat([x for x, _, _ in items], dim=0)
        total = x_batch.shape[0]
        dist_batch = torch.full((total, total), -1.0, dtype=torch.float)
        idx = 0
        for x, dist, _ in items:
            n = x.shape[0]
            dist_batch[idx:idx+n, idx:idx+n] = dist
            idx += n
        batch_vector = torch.cat([
            torch.full((x.shape[0],), g_idx, dtype=torch.long)
            for x, _, g_idx in items
        ], dim=0)
        return {'x_batch': x_batch, 'dist_batch': dist_batch, 'batch_vector': batch_vector}
    out = {
        'single': batch_components(single_items),
        'not_single': batch_components(multi_items)
    }
    y_batch = torch.cat(y_list, dim=0)
    return out, y_batch