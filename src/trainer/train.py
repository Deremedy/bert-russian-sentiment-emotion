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
        tokenizer,
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
        # Assume train_epoch function handles training logic
        train_y_true, train_y_pred, train_loss = train_epoch(
            model, train_dataloader, optimizer, problem_type
        )

        model.eval()
        # Assume predict function handles validation logic
        val_y_true, val_y_pred, val_loss = predict(model, val_dataloader, problem_type)

        # Optionally include calculate_metrics if needed for your application
        tq.set_description(f"train_loss: {train_loss:.4f}, val_loss: {val_loss:.4f}")

        # Save model and tokenizer using Hugging Face's `save_pretrained`
        model_directory = os.path.join(
            checkpoint_path,
            f"{model_name}-{dataset_name}-epoch_{epoch + 1}-val_loss_{val_loss:.4f}"
        )
        os.makedirs(model_directory, exist_ok=True)
        model.save_pretrained(model_directory)
        tokenizer.save_pretrained(model_directory)

    # Evaluate the model after all epochs are completed
    df = eval(model, test_dataloader, labels, problem_type)
    return df