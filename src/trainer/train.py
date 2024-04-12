import wandb
from src.trainer.metrics import calculate_metrics
from src.trainer.predict import predict, train_epoch
from src.trainer.eval import eval
from tqdm.auto import tqdm
import pandas as pd
import torch
import os


def train(
        model,
        model_name,
        dataset_name,
        train_dataloader,
        optimizer,
        epochs,
        val_dataloader,
        test_dataloader,
        labels,
        problem_type,
        log_wandb,
        checkpoint_path="model_checkpoints"
):
    # Ensure the checkpoint directory exists
    os.makedirs(checkpoint_path, exist_ok=True)

    tq = tqdm(range(epochs))

    for epoch in tq:
        model.train()
        train_y_true, train_y_pred, train_loss = train_epoch(
            model, train_dataloader, optimizer, problem_type
        )

        model.eval()
        val_y_true, val_y_pred, val_loss = predict(model, val_dataloader, problem_type)

        report_dict = calculate_metrics(val_y_true, val_y_pred, labels, problem_type)

        tq.set_description(f"train_loss: {train_loss:.4f}, val_loss: {val_loss:.4f}")

        # Create a descriptive filename for the checkpoint
        descriptive_filename = f"{model_name}_{dataset_name}_epoch_{epoch + 1}_val_loss_{val_loss:.4f}.pt"
        checkpoint_filename = os.path.join(checkpoint_path, descriptive_filename)
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
            'report_dict': report_dict
        }, checkpoint_filename)

    # Evaluate the model after all epochs are completed
    df = eval(model, test_dataloader, labels, problem_type)

    return df