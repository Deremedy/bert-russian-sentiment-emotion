import wandb
from src.trainer.train import train
from src.trainer.eval import eval
from src.model.models import get_model
import os
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate, call
from src.utils.utils import save_model, push_to_hub
import random
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# turn off bert warnings
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# printing full errors
os.environ["HYDRA_FULL_ERROR"] = "1"

# login to services
# wandb.login()
# notebook_login()

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    set_seed(42)

    if cfg.task == "train":
        tokenizer, model = get_model(
            cfg.model.encoder,
            cfg.dataset.labels,
            cfg.dataset.num_labels,
            cfg.trainer.problem_type,
            cfg.task,
        )

        train_dataloader, val_dataloader, test_dataloader = call(
            cfg.dataset.dataloader, tokenizer=tokenizer
        )

        if cfg.log_wandb:
            wandb.init(
                project=f"{cfg.project_name}-{cfg.model.name}-{cfg.dataset.name}",
                config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
            )
        optimizer = instantiate(cfg.optimizer, params=model.parameters())

        model.cuda()

        train(
            model=model,
            tokenizer=tokenizer,
            model_name=cfg.model.name,
            dataset_name=cfg.dataset.name,
            train_dataloader=train_dataloader,
            optimizer=optimizer,
            epochs=cfg.trainer.num_epochs,
            val_dataloader=val_dataloader,
            test_dataloader=test_dataloader,
            labels=cfg.dataset.labels,
            problem_type=cfg.trainer.problem_type,
            log_wandb=cfg.log_wandb,
        )

        # if cfg.log_wandb:
        #     wandb.finish()

        # save_model(
        #     model,
        #     tokenizer,
        #     f"models/{cfg.model.name}-{cfg.dataset.name}-ep={cfg.trainer.num_epochs}-lr={cfg.trainer.lr}",
        # )

    elif cfg.task == "eval":
        def load_model(tokenizer_dir, model_dir):
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
            model = AutoModelForSequenceClassification.from_pretrained(model_dir)
            return tokenizer, model

        # Path to the saved model directory
        saved_model_dir = f"model_checkpoints/{cfg.trained_model_name}"

        # Load model and tokenizer from the saved directory
        tokenizer, model = load_model(saved_model_dir, saved_model_dir)

        # Assuming you have a function to prepare data loaders
        train_dataloader, val_dataloader, test_dataloader = call(
            cfg.dataset.dataloader, tokenizer=tokenizer
        )

        # Assuming you have a function to evaluate the model
        results = eval(
            model=model,
            test_dataloader=test_dataloader,
            labels=cfg.dataset.labels,
            problem_type=cfg.trainer.problem_type,
            trained_model_name=cfg.trained_model_name,
        )
        print("Evaluation results:", results)

if __name__ == "__main__":
    main()
