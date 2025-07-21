import torch
import torch.nn as nn
import math
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts


class OneHotEmbedder(nn.Module):
    """
    A one-hot embedder that uses a linear layer to create one-hot-like embeddings.
    The weights of the linear layer are frozen.
    """
    def __init__(self, input_dim=50, output_dim=8):
        """
        Initializes the OneHotEmbedder module.

        Args:
            input_dim (int): The dimension of the input features.
            output_dim (int): The dimension of the output embeddings.
        """
        super().__init__()
        self.embedding = nn.Linear(input_dim, output_dim, bias=False)
        self.embedding.weight.requires_grad = False  # Freeze weights

    def forward(self, x):
        """
        Performs the forward pass of the OneHotEmbedder module.

        Args:
            x (torch.Tensor): A tensor of input features.

        Returns:
            torch.Tensor: The one-hot-like embeddings.
        """
        with torch.no_grad():
            return self.embedding(x)


class StabilizedBCEWithLogitsLoss(nn.Module):
    """
    A stabilized version of Binary Cross-Entropy with Logits Loss.
    It adds a penalty for large output magnitudes to prevent instability during training.
    """
    def __init__(self, stability_factor=10.0, print_components=False):
        """
        Initializes the StabilizedBCEWithLogitsLoss module.

        Args:
            stability_factor (float): The threshold for the magnitude penalty.
            print_components (bool): A flag indicating whether to print the loss components.
        """
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.stability_factor = stability_factor
        self.print_components = print_components
        
    def forward(self, outputs, labels):
        """
        Performs the forward pass of the StabilizedBCEWithLogitsLoss module.

        Args:
            outputs (torch.Tensor): The model's predictions.
            labels (torch.Tensor): The ground truth labels.

        Returns:
            torch.Tensor: The total loss.
        """
        bce_loss = self.bce(outputs, labels)
        
        # The magnitude penalty is the ReLU of the absolute output minus the stability factor.
        # This penalizes outputs with a magnitude greater than the stability_factor.
        magnitude_penalty = torch.mean(
            torch.relu(torch.abs(outputs) - self.stability_factor)
        )
        
        # Add debugging prints
        if torch.isnan(bce_loss).any():
            print(f"NaN detected in BCE loss. Outputs: {outputs}, Labels: {labels}")
        
        total_loss = bce_loss + 0.1 * magnitude_penalty
        
        if self.print_components:
            with torch.no_grad():
                print(f"BCE Loss: {bce_loss.item():.4f}")
                print(f"Magnitude Penalty: {(0.1 * magnitude_penalty).item():.4f}")
            
        return total_loss


class SymmetricStabilizedBCEWithLogitsLoss(nn.Module):
    """
    A symmetric, stabilized version of Binary Cross-Entropy with Logits Loss.
    It uses a symmetric penalty for large output magnitudes and supports positive class weighting.
    """
    def __init__(self, stability_factor=5.0, magnitude_weight=0.5, print_components=False, pos_weight=None):
        """
        Initializes the SymmetricStabilizedBCEWithLogitsLoss module.

        Args:
            stability_factor (float): The threshold for the magnitude penalty.
            magnitude_weight (float): The weight of the magnitude penalty.
            print_components (bool): A flag indicating whether to print the loss components.
            pos_weight (float, optional): A weight for the positive class.
        """
        super().__init__()

        if pos_weight is not None:
            self.bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))
        else:
            self.bce = nn.BCEWithLogitsLoss()
        self.stability_factor = stability_factor
        self.magnitude_weight = magnitude_weight
        self.print_components = print_components
        
    def forward(self, outputs, labels):
        """
        Performs the forward pass of the SymmetricStabilizedBCEWithLogitsLoss module.

        Args:
            outputs (torch.Tensor): The model's predictions.
            labels (torch.Tensor): The ground truth labels.

        Returns:
            torch.Tensor: The total loss.
        """
        # Standard BCE loss with optional weighting
        bce_loss = self.bce(outputs, labels)
        
        # Compute the symmetric magnitude penalty
        magnitudes = torch.abs(outputs)
        penalty_mask = magnitudes > self.stability_factor

        # The penalty is a smoothed version of the absolute difference between the magnitude and the stability factor.
        delta = 1.0  # Smoothing factor
        magnitude_penalty = torch.where(
            penalty_mask,
            delta * (torch.abs(magnitudes - self.stability_factor) - 0.5 * delta),
            torch.zeros_like(magnitudes)
        )

        # Calculate weighted penalty    
        total_penalty = self.magnitude_weight * torch.mean(magnitude_penalty)
        
        # Print components for monitoring
        if self.print_components:
            with torch.no_grad():
                print(f"BCE Component: {bce_loss.item():.4f}")
                print(f"Magnitude Penalty: {total_penalty.item():.4f}")
                if magnitudes.max() > self.stability_factor:
                    print(f"Warning: Output magnitude {magnitudes.max().item():.4f} exceeds stability factor {self.stability_factor}")
        
        return bce_loss + total_penalty



class CosineAnnealingWarmRestartsDecay(CosineAnnealingWarmRestarts):
    """
    A custom learning rate scheduler that combines cosine annealing with warm restarts and learning rate decay.
    The learning rate is decayed at the end of each restart cycle.
    """
    def __init__(
        self, optimizer, T_0, T_mult=1, eta_min=0, last_epoch=-1, verbose=False, decay=1
    ):
        """
        Initializes the CosineAnnealingWarmRestartsDecay scheduler.

        Args:
            optimizer (Optimizer): The optimizer.
            T_0 (int): The number of iterations for the first restart.
            T_mult (int): A factor that increases T_i after a restart.
            eta_min (float): The minimum learning rate.
            last_epoch (int): The index of the last epoch.
            verbose (bool): If True, prints a message on every update.
            decay (float): The factor by which the learning rate is decayed after each restart.
        """
        super().__init__(
            optimizer,
            T_0,
            T_mult=T_mult,
            eta_min=eta_min,
            last_epoch=last_epoch,
            verbose=verbose,
        )
        self.decay = decay
        self.initial_lrs = self.base_lrs

    def step(self, epoch=None):
        if epoch == None:
            if self.T_cur + 1 == self.T_i:
                if self.verbose:
                    print("multiplying base_lrs by {:.4f}".format(self.decay))
                self.base_lrs = [base_lr * self.decay for base_lr in self.base_lrs]
        else:
            if epoch < 0:
                raise ValueError(
                    "Expected non-negative epoch, but got {}".format(epoch)
                )
            if epoch >= self.T_0:
                if self.T_mult == 1:
                    n = int(epoch / self.T_0)
                else:
                    n = int(
                        math.log(
                            (epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult
                        )
                    )
            else:
                n = 0

            self.base_lrs = [
                initial_lrs * (self.decay**n) for initial_lrs in self.initial_lrs
            ]

        super().step(epoch)