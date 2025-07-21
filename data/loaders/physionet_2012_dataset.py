import torch
import torch.utils.data as data
import pandas as pd
import numpy as np
import pickle
from uuid import uuid4

from utils import OneHotEncoder

import os

PROJECT_ROOT = "."


class PhysioNet2012(data.Dataset):
    """
    A PyTorch Dataset for the PhysioNet 2012 challenge.
    """
    def __init__(
        self,
        files,
        config,
        biom_one_hot_embedder,
        save_path=None,
    ):
        """
        Initializes the PhysioNet2012 dataset.

        Args:
            files (list): A list of file paths to the dataset files.
            config (ConfigDict): A configuration object containing dataset parameters.
            biom_one_hot_embedder (OneHotEmbedder): A one-hot embedder for the biomarkers.
            save_path (str, optional): The path to save the processed data.
        """
        self.files = files
        self.config = config
        self.biomarker_features = config.biomarker_features
        self.static_features = config.static_features
        self.time_variable = config.time_variable
        self.biom_encoder = OneHotEncoder(self.biomarker_features)
        self.biom_one_hot_embedder = biom_one_hot_embedder
        self.save_path = save_path

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        return self._build_set_of_graphs(file_path)
            
    def _read_and_preprocess_data(self, file_path):
        """
        Reads and preprocesses a single data file.

        Args:
            file_path (str): The path to the data file.

        Returns:
            tuple: A tuple containing the processed dataframe, graph label, static features, and biomarkers.
        """
        abs_file_path = os.path.join(self.config.project_root, file_path)
        dataframe = pd.read_csv(abs_file_path, sep="|")
        dataframe = dataframe.sort_values(self.time_variable)
        dataframe = dataframe.dropna(axis=1, how='all')

        graph_label = dataframe['Survival'].iloc[-1]
        static_features = dataframe[self.static_features].iloc[0]
        biomarkers = [col for col in dataframe.columns if col in self.biomarker_features]

        return dataframe, graph_label, static_features, biomarkers

    def _build_node_features_and_distances(self, dataframe, biomarker, static_features):
        """
        Builds node features and distance matrices for a single biomarker.

        Args:
            dataframe (pd.DataFrame): The input dataframe.
            biomarker (str): The name of the biomarker.
            static_features (pd.Series): The static features for the graph.

        Returns:
            tuple: A tuple containing the node features and the distance matrix.
        """
        biom_data = dataframe[biomarker].dropna()
        biomarker_encoded = self.biom_one_hot_embedder(torch.tensor(self.biom_encoder.encode(biomarker).tolist()).to(torch.float)).tolist()

        node_features = torch.tensor(
            [[x] + static_features.tolist() + biomarker_encoded for x in biom_data],
            dtype=torch.float
        )

        time_dim = (dataframe[self.time_variable][biom_data.index]).to_numpy()
        node_distances = (time_dim[:, None] - time_dim[None, :]).astype(int)
        node_distances = np.tril(node_distances, k=0).astype(np.float32)
        node_distances[node_distances == 0] = np.inf
        np.fill_diagonal(node_distances, 0)
        node_distances += 0.1
        node_distances = 1 / node_distances

        return node_features, torch.tensor(node_distances, dtype=torch.float)

            
    def _build_set_of_graphs(self, file_path):
        """
        Builds a set of graphs from a single data file.

        Args:
            file_path (str): The path to the data file.

        Returns:
            tuple: A tuple containing the graph dictionary and the graph label.
        """
        dataframe, graph_label, static_features, biomarkers = self._read_and_preprocess_data(file_path)
        graph_dict = {}

        for biomarker in biomarkers:
            node_features, node_distances = self._build_node_features_and_distances(dataframe, biomarker, static_features)
            graph_dict[biomarker] = (node_features, node_distances, torch.tensor([graph_label], dtype=torch.float))

            if self.save_path:
                text_label = 'survived' if graph_label == 1 else 'no_survived'
                with open(f"{self.save_path}/{text_label}/{uuid4()}.pkl", 'wb') as f:
                    pickle.dump(graph_dict, f)
        
        return graph_dict, torch.tensor([graph_label], dtype=torch.float)

