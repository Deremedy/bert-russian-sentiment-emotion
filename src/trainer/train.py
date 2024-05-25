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
        # Train the model
        train_y_true, train_y_pred, train_loss = train_epoch(
            model, train_dataloader, optimizer, problem_type
        )

        model.eval()
        # Validate the model
        val_y_true, val_y_pred, val_loss = predict(model, val_dataloader, problem_type)

        # Update tqdm description with losses
        tq.set_description(f"train_loss: {train_loss:.4f}, val_loss: {val_loss:.4f}")

        # Save model and tokenizer using Hugging Face's `save_pretrained`
        filename = f"{model_name}-{dataset_name}-epoch_{epoch + 1}-val_loss_{val_loss:.4f}"
        # TODO: remove disabled saving
        # model_directory = os.path.join(
        #     checkpoint_path,
        #     filename
        # )
        # os.makedirs(model_directory, exist_ok=True)
        # model.save_pretrained(model_directory)
        # tokenizer.save_pretrained(model_directory)

        # Evaluate the model
        test_y_true, test_y_pred, test_loss = predict(model, test_dataloader, problem_type)
        report_dict = calculate_metrics(test_y_true, test_y_pred, labels, problem_type)

        # Save evaluation results to the same directory as the model
        df = pd.DataFrame(report_dict).round(2)
        print(df)
        evaluation_filepath = os.path.join(checkpoint_path, filename + "_evaluation_results.csv")
        df.to_csv(evaluation_filepath)

    # Evaluate the model after all epochs are completed
    df = eval(model, test_dataloader, labels, problem_type)
    return df