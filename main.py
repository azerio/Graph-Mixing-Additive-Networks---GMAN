import argparse
import os
import random
import numpy as np
import torch
import wandb
from datetime import datetime
import pickle
from torch.utils.data import DataLoader

from data.loaders.physionet_2012_dataset import PhysioNet2012
from data.loaders.fakenews import FakeNewsTwitterDataset
from model.utils import OneHotEmbedder, SymmetricStabilizedBCEWithLogitsLoss
from config import get_config
from model.GMAN import GMAN as PhysionetGMAN
from model.GMANFakeNews import GMAN as FakeNewsGMAN
from data.collate_fns.GMAN.physionet import distance_collate_fn_physionet
from data.collate_fns.GMAN.fakenews import distance_collate_fn_fakenews
from model.GMAN_trainer import train_epoch, test_epoch

def setup_seed(seed):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_dataloader(exp_config, data_split):
    """Get dataloaders for the specified dataset."""
    if exp_config.dataset == 'physionet':
        one_hot_embedder = OneHotEmbedder(input_dim=exp_config.num_biom, output_dim=exp_config.num_biom_embed)
        train_dataset = PhysioNet2012(files=data_split['train_files'], config=exp_config, biom_one_hot_embedder=one_hot_embedder)
        val_dataset = PhysioNet2012(files=data_split['val_files'], config=exp_config, biom_one_hot_embedder=one_hot_embedder)
        test_dataset = PhysioNet2012(files=data_split['test_files'], config=exp_config, biom_one_hot_embedder=one_hot_embedder)
        collate_fn = distance_collate_fn_physionet
    elif exp_config.dataset == 'fakenews':
        processed_path = 'FakeNewsData/data/gossipcop_graphs.pt'
        data = torch.load(processed_path)
        train_dataset = FakeNewsTwitterDataset(data=data, config=exp_config, roots=data_split['train_files'])
        val_dataset = FakeNewsTwitterDataset(data=data, config=exp_config, roots=data_split['val_files'])
        test_dataset = FakeNewsTwitterDataset(data=data, config=exp_config, roots=data_split['test_files'])
        collate_fn = distance_collate_fn_fakenews
    else:
        raise ValueError(f"Unknown dataset: {exp_config.dataset}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=exp_config.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=exp_config.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=exp_config.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )
    return train_loader, val_loader, test_loader

def get_model(exp_config, device):
    """Get the model for the specified dataset."""
    if exp_config.dataset == 'physionet':
        model = PhysionetGMAN(
            config=exp_config
        ).to(device)
    elif exp_config.dataset == 'fakenews':
        model = FakeNewsGMAN(
            config=exp_config
        ).to(device)
    else:
        raise ValueError(f"Unknown dataset: {exp_config.dataset}")
    return model

def main():
    exp_config = get_config()
    
    print(exp_config)

    setup_seed(exp_config.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if exp_config.dataset == 'physionet':
        split_path = os.path.join("P12_data_splits", "split_1.pkl")
    elif exp_config.dataset == 'fakenews':
        split_path = "FakeNewsData/splits/split.pkl"

    with open(split_path, "rb") as f:
        data_split = pickle.load(f)
    
    print("Loaded pre-computed splits:")
    print(f"Number of training samples: {len(data_split['train_files'])}")
    print(f"Number of validation samples: {len(data_split['val_files'])}")
    print(f"Number of test samples: {len(data_split['test_files'])}")

    if exp_config.dataset == 'physionet':
        pos_weight = 6.0
        print("Weighted_loss:", pos_weight)
    
    train_loader, val_loader, test_loader = get_dataloader(exp_config, data_split)
    exp_config.device = device
    model = get_model(exp_config, device)

    unique_run_name = f"{exp_config.dataset}_layers-{exp_config.n_layers}_hidden-{exp_config.hidden_channels}_lr-{exp_config.lr}_dropout-{exp_config.dropout}_seed-{exp_config.seed}"
    
    if exp_config.wandb:
        config_dict = exp_config.copy()
        config_dict['device'] = torch.cuda.get_device_name() if torch.cuda.is_available() else 'cpu'
        config_dict['model'] = model.__class__.__name__
        wandb.init(
            project=f"GMAN-{exp_config.dataset.capitalize()}",
            config=config_dict,
            settings=wandb.Settings(start_method='thread'),
            name=unique_run_name,
            reinit=True
        )

    optimizer = torch.optim.Adam(model.parameters(), lr=exp_config.lr, weight_decay=exp_config.wd)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=20, min_lr=1e-8
    )
    
    last_epoch = 0
    
    print(f"Started training on {device}")
    print(f"Running experiment: {exp_config.exp_name}")
    
    best_val_loss = float('inf')
    best_val_auc = float('-inf')
    best_val_auprc = float('-inf')

    for epoch in range(exp_config.epochs):
        if exp_config.dataset == 'physionet':
            loss_fn = SymmetricStabilizedBCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            loss_fn = SymmetricStabilizedBCEWithLogitsLoss()

        
        train_loss, train_acc, train_auc, train_auprc  = train_epoch(
            epoch=last_epoch + epoch,
            model=model,
            dloader=train_loader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            writer=None
        )
        print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Train Accuracy={train_acc:.4f}, Train AUC={train_auc:.4f}, Train AUPRC={train_auprc:.4f}")
        
        val_loss, val_acc, val_auc, val_auprc = test_epoch(
            epoch=last_epoch + epoch,
            model=model,
            dloader=val_loader,
            loss_fn=loss_fn,
            device=device,
            writer=None
        )
        print(f"Epoch {epoch}: Val Loss={val_loss:.4f}, Val Accuracy={val_acc:.4f}, Val AUC={val_auc:.4f}, Val AUPRC={val_auprc:.4f}")

        if exp_config.dataset == 'fakenews':
            scheduler.step(val_loss)
        
        test_loss, test_acc, test_auc, test_auprc = test_epoch(
            epoch=last_epoch + epoch,
            model=model,
            dloader=test_loader,
            loss_fn=loss_fn,
            device=device,
            writer=None
        )
        print(f"Epoch {epoch}: Test Loss={test_loss:.4f}, Test Accuracy={test_acc:.4f}, Test AUC={test_auc:.4f}, Test AUPRC={test_auprc:.4f}")
        
        if exp_config.wandb:
            wandb.log({
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "train_auc": train_auc,
                "train_auprc": train_auprc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_auc": val_auc,
                "val_auprc": val_auprc,
                "test_loss": test_loss,
                "test_acc": test_acc,
                "test_auc": test_auc,
                "test_auprc": test_auprc,
            })
        
        checkpoint_dir = f"{exp_config.model_checkpoints_dir}/{unique_run_name}"
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_name = os.path.join(checkpoint_dir, f'best_params_by_val_loss.pth')
            torch.save(model.state_dict(), checkpoint_name)
            if exp_config.wandb:
                wandb.save(checkpoint_name)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            checkpoint_name = os.path.join(checkpoint_dir, f'best_params_by_val_auc.pth')
            torch.save(model.state_dict(), checkpoint_name)
            if exp_config.wandb:
                wandb.save(checkpoint_name)

        if val_auprc > best_val_auprc:
            best_val_auprc = val_auprc
            checkpoint_name = os.path.join(checkpoint_dir, f'best_params_by_val_auprc.pth')
            torch.save(model.state_dict(), checkpoint_name)
            if exp_config.wandb:
                wandb.save(checkpoint_name)
        
        checkpoint_name = os.path.join(checkpoint_dir, f'last_epoch.pth')
        torch.save(model.state_dict(), checkpoint_name)
        if exp_config.wandb:
            wandb.save(checkpoint_name)

    test_loss, test_acc, test_auc, test_auprc = test_epoch(
        epoch=last_epoch + exp_config.epochs,
        model=model,
        dloader=test_loader,
        loss_fn=loss_fn,
        device=device,
        writer=None
    )
    print(f"Final Test Loss={test_loss:.4f}, Final Test Accuracy={test_acc:.4f}, Final Test AUC={test_auc:.4f}, Final Test AUPRC={test_auprc:.4f}")
    if exp_config.wandb:
        wandb.log({
            "final_test_loss": test_loss,
            "final_test_acc": test_acc,
            "final_test_auc": test_auc,
            "final_test_auprc": test_auprc
        })
        wandb.finish()

if __name__ == '__main__':
    main() 