import torch
import torch.nn as nn
from model.GMAN import EXTGNAN, DeepSet

# GMAN with two GNANs and original hidden collapse
class GMAN(nn.Module):
    """
    GMAN (Graph-Mixing Additive Network) for the FakeNews dataset.
    This model uses two separate EXTGNANs to process 'single' and 'not_single' components of the input graphs.
    """
    def __init__(
        self,
        config
    ):
        """
        Initializes the GMAN module for FakeNews.

        Args:
            config (ConfigDict): A configuration object containing model hyperparameters.
        """
        super().__init__()
        self.device = config.device
        self.config = config

        # two independent GNANs
        self.gnans = nn.ModuleDict({
            'single': EXTGNAN(
                config=config,
                is_graph_task=config.is_graph_task,
                batch_size=config.batch_size
            ).to(self.device),
            'not_single': EXTGNAN(
                config=config,
                is_graph_task=config.is_graph_task,
                batch_size=config.batch_size
            ).to(self.device),
        })

        # DeepSet for 'single' group
        self.deep_set = DeepSet(
            in_dim=config.hidden_channels,
            out_dim=config.hidden_channels,
            hidden_dim=config.hidden_channels,
            n_layers=config.get('n_biom_group_layers', 3),
            device=self.device
        ).to(self.device)

        # init weights
        for name, p in self.named_parameters():
            if 'weight' in name and p.dim() >= 2:
                nn.init.kaiming_normal_(p, nonlinearity='leaky_relu')
            elif 'bias' in name:
                nn.init.constant_(p, 0)

    def forward(self, inputs, batch_dim, return_group_outputs = False):
        """
        Performs the forward pass of the GMAN module.

        Args:
            inputs (dict): A dictionary of input data, with keys 'single' and 'not_single'.
            batch_dim (int): The batch dimension.
            return_group_outputs (bool): A flag indicating whether to return the outputs of the biomarker groups.

        Returns:
            torch.Tensor or tuple: The output of the model.
        """
        # "single" path → GNAN then DeepSet
        s = self.gnans['single'](
            inputs['single']['x_batch'].to(self.device),
            inputs['single']['dist_batch'].to(self.device),
            inputs['single']['batch_vector'].to(self.device),
        )
        s = self.deep_set(s.unsqueeze(1))  # [batch, hidden]

        # "not_single" path → GNAN only
        ns = self.gnans['not_single'](
            inputs['not_single']['x_batch'].to(self.device),
            inputs['not_single']['dist_batch'].to(self.device),
            inputs['not_single']['batch_vector'].to(self.device),
        )  # [batch, hidden]

        group_outputs = torch.stack([s, ns], dim=1)  # [batch, 2, hidden]
        if return_group_outputs:
            return group_outputs

        combined = s + ns                    # [batch, hidden]
        return combined.sum(dim=-1)         # [batch]