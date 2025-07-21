import torch
import torch.nn as nn
from collections import defaultdict

class EXTGNAN(nn.Module):
    """
    EXTGNAN (Graph-based Neural Additive Network) is a module that processes graph-structured data.
    It applies a series of transformations to input features and distance matrices to produce node embeddings.
    """
    def __init__(self, config, is_graph_task, batch_size):
        """
        Initializes the EXTGNAN module.

        Args:
            config (ConfigDict): A configuration object containing model hyperparameters.
            is_graph_task (bool): A flag indicating whether the task is a graph-level prediction task.
            batch_size (int): The size of the input batches.
        """
        super().__init__()

        self.device = config.device
        self.out_channels = config.out_channels
        self.hidden_channels = config.hidden_channels
        self.num_layers = config.n_layers
        self.bias = config.bias
        self.init_std = config.init_std
        self.dropout = config.dropout
        self.rho_per_feature = config.rho_per_feature
        self.normalize_rho = config.normalize_rho
        self.is_graph_task = is_graph_task
        self.readout_n_layers = config.readout_n_layers
        self.feature_groups = config.feature_groups
        self.batch_size = batch_size
        self.fs = nn.ModuleList()
        self.return_laplacian = config.return_laplacian

        for group_index, group in enumerate(self.feature_groups):
            curr_f = self._create_layers(
                in_channels=group, 
                out_channels=self.out_channels,
                hidden_channels=self.hidden_channels,
                bias=self.bias,
                num_layers=self.num_layers,
            )
            
            self.fs.append(nn.Sequential(*curr_f).to(self.device))

        m_bias = True
        if is_graph_task:
            m_bias = False

        rho_layers = []
        in_dim = 1
        for _ in range(self.num_layers):
            rho_layers.append(nn.Linear(in_dim, self.hidden_channels, bias=self.bias))
            # rho_layers.append(nn.BatchNorm1d(self.hidden_channels))
            # rho_layers.append(nn.GroupNorm(num_groups=1, num_channels=hidden_channels))
            rho_layers.append(nn.LeakyReLU())
            rho_layers.append(nn.Dropout(self.dropout))
            in_dim = self.hidden_channels
        rho_layers.append(nn.Linear(self.hidden_channels, self.out_channels, bias=self.bias))
        self.rho = nn.Sequential(*rho_layers).to(self.device)

        self._init_params()

    def _init_params(self):
        """Initializes the model's parameters using Kaiming normal initialization for weights and zeros for biases."""
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() >= 2:
                nn.init.kaiming_normal_(param, mode='fan_in', nonlinearity='leaky_relu')
            elif 'bias' in name:
                nn.init.constant_(param, 0)

    def _create_layers(self, in_channels, out_channels, hidden_channels, bias, num_layers):
        """
        Creates a list of layers for a feed-forward network.

        Args:
            in_channels (int): The number of input channels.
            out_channels (int): The number of output channels.
            hidden_channels (int): The number of hidden channels.
            bias (bool): A flag indicating whether to use bias terms.
            num_layers (int): The number of layers in the network.

        Returns:
            list: A list of PyTorch layers.
        """
        if num_layers == 1:
                curr_f = [nn.Linear(in_channels, out_channels, bias=bias)]
        else:

            curr_f = [
                nn.Linear(in_channels, hidden_channels, bias=bias),
                # nn.BatchNorm1d(hidden_channels),
                # nn.GroupNorm(num_groups=self.batch_size, num_channels=hidden_channels),
                nn.LayerNorm(hidden_channels),
                nn.ReLU(), 
                nn.Dropout(p=self.dropout)
            ]

            for j in range(1, num_layers - 1):
                curr_f.append(
                    nn.Linear(hidden_channels, hidden_channels, bias=bias)
                )
                # curr_f.append(nn.BatchNorm1d(hidden_channels))
                # curr_f.append(nn.GroupNorm(num_groups=self.batch_size, num_channels=hidden_channels))
                curr_f.append(nn.LayerNorm(hidden_channels))
                curr_f.append(nn.ReLU())
                curr_f.append(nn.Dropout(p=self.dropout))
            curr_f.append(nn.Linear(hidden_channels, out_channels, bias=bias))
        return curr_f

    def compute_laplacian_from_learned_dist(self, dist_matrix, sigma=1.0, normalized=True, device='cpu'):
        """
        Computes the graph Laplacian from a learned distance matrix.

        Args:
            dist_matrix (torch.Tensor): A tensor representing the learned distance matrix.
            sigma (float): The sigma value for the Gaussian kernel.
            normalized (bool): A flag indicating whether to compute the normalized Laplacian.
            device (str): The device to which the tensors should be moved.

        Returns:
            tuple: A tuple containing the Laplacian matrix and its eigenvalues.
        """
        A = torch.exp(- (dist_matrix ** 2) / sigma**2) # Gaussian kernel (affinity)
        # A = torch.tril(A)  # only lower triangle

        A = 0.5 * (A + A.T) # try with symmetric A

        D = torch.diag(A.sum(dim=1))

        if normalized:
            d = D.diag().clamp(min=1e-3)
            d_inv_sqrt = torch.diag(1.0 / torch.sqrt(D.diag() + 1e-6))
            L = torch.eye(A.size(0)).to(device) - d_inv_sqrt @ A @ d_inv_sqrt
        else:
            L = D - A
        
        eigvals = torch.linalg.eigvalsh(L.cpu()).to(device)

        return L, eigvals

    def forward(self, x_batch, dist_batch, batch_vector, return_node_embeddings=False):
        """
        Performs the forward pass of the EXTGNAN module.

        Args:
            x_batch (torch.Tensor): A tensor of input node features.
            dist_batch (torch.Tensor): A tensor of distance matrices.
            batch_vector (torch.Tensor): A tensor mapping each node to its graph in the batch.
            return_node_embeddings (bool): A flag indicating whether to return node embeddings.

        Returns:
            torch.Tensor or tuple: The output of the model, which is either a tensor of graph embeddings
                                   or a tuple containing graph embeddings and Laplacian data.
        """
        N, _ = x_batch.shape
        fx = torch.empty(N, len(self.feature_groups), self.out_channels).to(self.device)
        start_idx = 0

        for group_index, group_size in enumerate(self.feature_groups):  
            feature_cols = x_batch[:, start_idx : start_idx + group_size]


            if group_size == 1:
                feature_cols = feature_cols.view(-1, 1)

            feature_cols = self.fs[group_index](feature_cols)

            fx[:, group_index] = feature_cols
            start_idx += group_size

        fx_perm = torch.permute(fx, (2, 0, 1))

        dist_embed = self.rho(dist_batch.flatten().view(-1, 1))
        dist_embed = dist_embed.view(N, N, self.out_channels)
        mask = (dist_batch >= 0)
        dist_embed[~mask] = 0.0
        m_dist = dist_embed.permute(2, 0, 1)
        
        mf = torch.matmul(m_dist, fx_perm)
        mf = mf.sum(dim=2)
        mf = mf.permute(1, 0)

        if self.is_graph_task:
            num_graphs = batch_vector.max().item() + 1
            out_graph = torch.zeros(num_graphs, mf.size(1), device=mf.device)
            graph_index = batch_vector.view(-1, 1).expand(-1, mf.size(1))
            out_graph.scatter_add_(0, graph_index, mf)

            if self.return_laplacian:
                laplacian_data = {}

                with torch.no_grad():
                    unique_graph_ids = batch_vector.unique()
                    for graph_id in unique_graph_ids:
                        node_mask = (batch_vector == graph_id)

                        learned_dist_scalar = dist_embed.mean(dim=-1)  # shape: [N, N]
                        learned_dist_scalar[~mask] = 0.0

                        # if node_mask.sum() < 2:
                        #     continue # skip graphs with fewer than 2 nodes

                        sub_dist_matrix = learned_dist_scalar[node_mask][:, node_mask]
                        L, eigvals = self.compute_laplacian_from_learned_dist(
                            sub_dist_matrix,
                            sigma=1.0,
                            normalized=True,
                            device=self.device
                        )
                        laplacian_data[int(graph_id)] = {
                            "laplacian": L,
                            "eigenvalues": eigvals,
                        }

                        if return_node_embeddings:
                            return out_graph, laplacian_data, mf

                        return out_graph, laplacian_data

            return out_graph

        return mf

class DeepSet(nn.Module):
    """
    A DeepSet module for processing sets of items.
    It consists of a permutation-invariant layer (phi) and a readout layer.
    """
    def __init__(self, in_dim, out_dim, hidden_dim, n_layers, device):
        """
        Initializes the DeepSet module.

        Args:
            in_dim (int): The dimension of the input features.
            out_dim (int): The dimension of the output features.
            hidden_dim (int): The dimension of the hidden layers.
            n_layers (int): The number of layers in the phi network.
            device (str): The device to which the tensors should be moved.
        """
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.device = device

        layers = []
        for i in range(n_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())

        self.phi = nn.Sequential(*layers).to(device)
        self.readout = nn.Linear(hidden_dim, out_dim).to(device)

    def forward(self, x):
        """
        Performs the forward pass of the DeepSet module.

        Args:
            x (torch.Tensor): A tensor of input items.

        Returns:
            torch.Tensor: The output of the DeepSet module.
        """
        x = self.phi(x)
        x = torch.mean(x, dim=0)
        x = self.readout(x)
        return x


class GMAN(nn.Module):
    """
    GMAN (Graph-Mixing Additive Network) is a model that uses multiple EXTGNANs to process different groups of biomarkers.
    It combines the outputs of the EXTGNANs using a DeepSet to produce a final prediction.
    """
    def __init__(
            self,
            config
    ):
        """
        Initializes the GMAN module.

        Args:
            config (ConfigDict): A configuration object containing model hyperparameters.
        """
        super().__init__()

        self.device = config.device
        self.out_channels = config.out_channels
        self.hidden_channels = config.hidden_channels
        self.n_layers = config.n_layers
        self.bias = config.get('bias', True)
        self.dropout = config.dropout
        self.rho_per_feature = config.get('rho_per_feature', True)
        self.normalize_rho = config.normalize_rho
        self.fs = nn.ModuleList()
        self.is_graph_task = config.is_graph_task
        self.readout_n_layers = config.get('readout_n_layers', 1)
        self.max_num_GNANs = config.max_num_GNANs
        self.feature_groups = config.feature_groups
        self.batch_size = config.batch_size
        self.biomarker_groups = config.biomarker_groups
        self.n_biom_group_layers = config.get('n_biom_group_layers', 3)
        self.return_laplacian = config.get('return_laplacian', False)
        self.deepset_n_layers = config.get('deepset_n_layers', 2)
        self.gnan_mode = config.gnan_mode
        self.config = config


        if self.gnan_mode == "single":
            self.gnans = nn.ModuleList([
                self.create_gnan().to(self.device)
            ])
            self.biomarker_to_gnan = defaultdict(lambda: 0)

        elif self.gnan_mode == "per_group":
            self.gnans = nn.ModuleList([
                self.create_gnan().to(self.device) for _ in self.biomarker_groups
            ])
            self.biomarker_to_gnan = {}
            for group_idx, group in enumerate(self.biomarker_groups):
                for biom in group:
                    self.biomarker_to_gnan[biom] = group_idx

        elif self.gnan_mode == "per_biomarker":
            all_biomarkers = [b for group in self.biomarker_groups for b in group]
            self.gnans = nn.ModuleList([
                self.create_gnan().to(self.device) for _ in all_biomarkers
            ])
            self.biomarker_to_gnan = {biom: i for i, biom in enumerate(all_biomarkers)}

        else:
            raise ValueError(f"Invalid gnan_mode: {self.gnan_mode}")

        self.readout = nn.Linear(self.max_num_GNANs, self.out_channels, bias=self.bias).to(self.device)

        self.group_deep_sets = nn.ModuleDict({
            str(i): DeepSet(in_dim=self.hidden_channels, out_dim=self.hidden_channels, hidden_dim=self.hidden_channels, n_layers=self.n_biom_group_layers, device=self.device).to(self.device) for i in range(len(self.biomarker_groups))
        })
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() >= 2:
                nn.init.kaiming_normal_(param, mode='fan_in', nonlinearity='leaky_relu')
            elif 'bias' in name:
                nn.init.constant_(param, 0)

    def create_gnan(self):
        """
        Creates an EXTGNAN module.

        Returns:
            EXTGNAN: An EXTGNAN module.
        """
        return EXTGNAN(
            config=self.config,
            is_graph_task=self.is_graph_task,
            batch_size=self.batch_size,
        )

    def create_mlp(self, num_layers):
        """
        Creates a multi-layer perceptron (MLP).

        Args:
            num_layers (int): The number of layers in the MLP.

        Returns:
            nn.Sequential: A PyTorch sequential container representing the MLP.
        """
        layers = []
        out_dim = self.hidden_channels
        for _ in range(num_layers):
            layers.append(nn.Linear(self.hidden_channels, self.hidden_channels))
            layers.append(nn.LayerNorm(self.hidden_channels))
            layers.append(nn.LeakyReLU())
            layers.append(nn.Dropout(self.dropout))
        layers.append(nn.Linear(self.hidden_channels, out_dim, bias=True))
        return nn.Sequential(*layers)

    def forward(self, inputs, batch_dim, return_group_outputs=False):
        """
        Performs the forward pass of the GMAN module.

        Args:
            inputs (dict): A dictionary of input data, where keys are biomarker names and values are dictionaries
                           of input tensors.
            batch_dim (int): The batch dimension.
            return_group_outputs (bool): A flag indicating whether to return the outputs of the biomarker groups.

        Returns:
            torch.Tensor or tuple: The output of the model, which is either a tensor of predictions
                                   or a tuple containing group outputs and Laplacian data.
        """
        assert (len(list(inputs.keys())) <= self.max_num_GNANs)
        biom_idx_map = {}
        outputs = torch.zeros(size=(self.max_num_GNANs, batch_dim, self.hidden_channels)).to(self.device)
        laplacian_outputs = {}

        for ind, (k, b) in enumerate(inputs.items()):
            biom_idx_map[k] = ind
            gnan_idx = self.biomarker_to_gnan[k]

            x_batch = b['x_batch'].to(self.device)
            dist_batch = b['dist_batch'].to(self.device)
            batch_vector = b['batch_vector'].to(self.device)

            if self.return_laplacian:
                out = self.gnans[gnan_idx](x_batch, dist_batch, batch_vector)

                gnan_output, lap_data = out
                laplacian_outputs[k] = lap_data
            else:
                gnan_output = self.gnans[gnan_idx](x_batch, dist_batch, batch_vector)
            expected_size = outputs[ind].shape[0]
            actual_size = gnan_output.shape[0]

            if actual_size < expected_size:
                pad_size = expected_size - actual_size
                pad_tensor = torch.zeros((pad_size, gnan_output.shape[1]), device=gnan_output.device)
                gnan_output = torch.cat([gnan_output, pad_tensor], dim=0)

            outputs[ind] = gnan_output

        groups_output = []
        outputs = outputs.permute(1, 0, 2)

        for group_idx, group in enumerate(self.biomarker_groups):
            group_out = []
            for biomarker in group:
                if biomarker in biom_idx_map:
                    biom_idx = biom_idx_map[biomarker]
                    group_out.append(outputs[:, biom_idx])

            if group_out and len(group_out) > 1:
                group_out = torch.stack(group_out, dim=0)
                group_out = self.group_deep_sets[str(group_idx)](group_out)
                groups_output.append(group_out)
            elif group_out and len(group_out) == 1:
                groups_output.append(group_out[0])

        groups_output = torch.stack(groups_output, dim=0)

        groups_output = groups_output.permute(1, 0, 2)
        outputs = groups_output.sum(dim=-1)

        if return_group_outputs:
            return groups_output, laplacian_outputs

        if self.return_laplacian:
            return outputs.sum(dim=-1), laplacian_outputs

        return outputs.sum(dim=-1)
