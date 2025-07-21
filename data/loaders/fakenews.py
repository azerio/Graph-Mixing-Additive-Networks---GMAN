import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

class FakeNewsTwitterDataset(Dataset):
    """
    Each sample is a tuple (s_dict, label) for one graph.
    You can optionally restrict to a subset of root_ids via `roots`.
    """
    def __init__(self, data, config, roots=None):
        """
        Initializes the FakeNewsTwitterDataset.

        Args:
            data (dict): A dictionary of graph data.
            config (ConfigDict): A configuration object containing dataset parameters.
            roots (list, optional): A list of root IDs to filter the dataset.
        """
        self.config = config
        self.graph_key = config.graph_key

        # If a subset of roots specified, filter
        if roots is not None:
            data = {r: data[r] for r in roots if r in data}
            
        self.samples = [
            ({self.graph_key: {nid: (x, dist) for nid, (x, dist, _) in graph.items()}}, next(iter(graph.values()))[2])
            for graph in tqdm(data.values(), desc="Building samples", total=len(data))
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]