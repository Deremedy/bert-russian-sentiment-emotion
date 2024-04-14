from src.trainer.metrics import calculate_metrics, calculate_f1_score_old
from src.trainer.predict import predict
import pandas as pd
import os


def get_unique_filename(base_filename, directory="model_evaluation_runs"):
    """
    Generates a unique filename by appending a counter to the base filename
    if a file with the same name already exists in the specified directory.
    """
    # Ensure the directory exists
    os.makedirs(directory, exist_ok=True)

    # Prepare initial filename
    full_path = os.path.join(directory, f"{base_filename}.csv")
    counter = 1

    # If the file exists, find a unique name by appending a counter
    while os.path.exists(full_path):
        full_path = os.path.join(directory, f"{base_filename}_{counter}.csv")
        counter += 1

    return full_path


def eval(model, test_dataloader, labels, problem_type, return_debug=False, trained_model_name=None):
    model.eval()

    test_y_true, test_y_pred, test_loss = predict(model, test_dataloader, problem_type)
    report_dict = calculate_metrics(test_y_true, test_y_pred, labels, problem_type)

    df = pd.DataFrame(report_dict)

    # if problem_type == "multi_label_classification":
    #     f1s, f1 = calculate_f1_score_old(test_y_true, test_y_pred, "micro", len(labels))
    #     df.loc["wrong f1 micro"] = f1s + [None] + [f1] + [None, None]

    #     f1s, f1 = calculate_f1_score_old(test_y_true, test_y_pred, "macro", len(labels))
    #     df.loc["wrong f1 macro"] = f1s + [None] + [f1] + [None, None]

    df = df.round(2)

    print(df)
    run_report_filepath = get_unique_filename(f"{trained_model_name}")
    df.to_csv(run_report_filepath)

    if return_debug:
        return test_y_true, test_y_pred, df
    else:
        return df
