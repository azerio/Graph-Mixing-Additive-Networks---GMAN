import torch
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix, ConfusionMatrixDisplay
from tqdm import tqdm
import matplotlib.pyplot as plt
import traceback


def run_epoch(
    epoch,
    model,
    dloader,
    loss_fn,
    optimizer,
    scheduler,
    device,
    running_loss,
    n_samples,
    all_probas,
    all_y_batch,
    running_acc,
    writer,
    is_train,
    cleanup=True,
    train_after_oom=False,
    log_interval=50
):
    logging_dir = "Train" if is_train else "Test"
    print()
    print(f"{logging_dir} run")

    with torch.autograd.set_detect_anomaly(True):
        for i, (data_batch, y_batch) in enumerate(tqdm(dloader)):

            try:
                y_batch = y_batch.to(device)
                outputs = model(data_batch, y_batch.shape[0])

                if isinstance(outputs, tuple):
                    outputs = outputs[0]

                loss = loss_fn(outputs.view(-1), y_batch)

                if is_train:
                    optimizer.zero_grad()
                    loss.backward()

                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  
                    optimizer.step()
                    scheduler.step(loss)

                running_loss += loss.item()
                n_samples += len(y_batch)

                preds = torch.sigmoid(outputs).view(-1) > 0.5
                running_acc += (preds == y_batch).sum().item()

                all_probas.append(torch.sigmoid(outputs).detach().cpu().numpy())
                all_y_batch.append(y_batch.detach().cpu().numpy())

                if writer is not None:
                    writer.add_scalar(f"{logging_dir} Batch Loss", loss.item(), epoch * len(dloader) + i)                

                if (i + 1) % log_interval == 0:
                    intermediate_probas = np.concatenate(all_probas)
                    intermediate_y_batch = np.concatenate(all_y_batch)
                    intermediate_auc = roc_auc_score(intermediate_y_batch, intermediate_probas)
                    intermediate_acc = running_acc / n_samples
            except Exception as e:
                print(f"Skipping batch {i} due to error: {e}")
                traceback.print_exc()
                continue



        # Aggregate AUC computation
        all_probas = np.concatenate(all_probas)
        all_y_batch = np.concatenate(all_y_batch)

        auc = roc_auc_score(all_y_batch, all_probas)
        auprc = average_precision_score(all_y_batch, all_probas)
        avg_loss = running_loss / len(dloader)
        avg_acc = running_acc / n_samples

        # Compute confusion matrix
        all_preds = (all_probas > 0.5).astype(int)
        cm = confusion_matrix(all_y_batch, all_preds)
        cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

        # Plot confusion matrix
        fig, ax = plt.subplots(figsize=(6, 6))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm_percentage, display_labels=[0, 1])
        disp.plot(ax=ax, cmap="Blues", values_format=".2f")
        plt.title(f"{logging_dir} Confusion Matrix - Epoch {epoch}")
        plt.tight_layout()

        if writer is not None:
            writer.add_scalar(f"{logging_dir} Loss/Epoch", avg_loss, epoch)
            writer.add_scalar(f"{logging_dir} Accuracy/Epoch", avg_acc, epoch)
            writer.add_scalar(f"{logging_dir} AUC/Epoch", auc, epoch)
            writer.add_figure(f"{logging_dir} Confusion Matrix/Epoch {epoch}", fig, global_step=epoch)
            writer.add_scalar(f"{logging_dir} AUPRC/Epoch", auprc, epoch)


        plt.close(fig)


    return avg_loss, avg_acc, auc, auprc

def train_epoch(epoch, model, dloader, loss_fn, optimizer, scheduler, device, train_after_oom=False, writer=None):
    model.train()
    running_loss = 0.0
    n_samples = 0
    all_probas = []
    all_y_batch = []
    running_acc = 0.0

    avg_loss, avg_acc, auc, auprc = run_epoch(
        epoch,
        model,
        dloader,
        loss_fn,
        optimizer,
        scheduler,
        device,
        running_loss,
        n_samples,
        all_probas,
        all_y_batch,
        running_acc,
        writer,
        is_train=True,
    )

    return avg_loss, avg_acc, auc, auprc

def test_epoch(epoch, model, dloader, loss_fn, device, train_after_oom=False, writer=None):
    with torch.no_grad():
        running_loss = 0.0
        n_samples = 0
        all_probas = []
        all_y_batch = []
        running_acc = 0.0

        model.eval()
    
        avg_loss, avg_acc, auc, auprc = run_epoch(
            epoch,
            model,
            dloader,
            loss_fn,
            None,
            None,
            device,
            running_loss,
            n_samples,
            all_probas,
            all_y_batch,
            running_acc,
            writer,
            is_train=False
        )

        return avg_loss, avg_acc, auc, auprc