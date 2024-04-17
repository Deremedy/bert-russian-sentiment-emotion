import pandas as pd

def load_and_process_csv(file_path):
    # Load the CSV file into a DataFrame
    data = pd.read_csv(file_path)

    # Transpose the DataFrame
    data_transposed = data.set_index('Unnamed: 0').T
    data_transposed.columns = ['precision', 'recall', 'f1-score', 'support', 'auc-roc']

    # Rearrange columns
    data_transposed = data_transposed[['precision', 'recall', 'f1-score', 'auc-roc', 'support']]

    return data_transposed

def format_as_markdown_table(dataframe):
    # Create the header of the table
    header = "|              |Precision|Recall|F1-Score|AUC-ROC|Support|\n"
    header += "|--------------|---------|------|--------|-------|-------|\n"

    # Format each row as a markdown table row
    markdown_table = header
    for index, row in dataframe.iterrows():
        row_str = f"|{index:14}|{row['precision']:>9.2f}|{row['recall']:>6.2f}|{row['f1-score']:>8.2f}|{row['auc-roc']:>7.2f}|{int(row['support']):6}|\n"
        markdown_table += row_str

    return markdown_table

# Usage
file_path = 'lvbert-lv-go-emotions-token-lv-epoch_5-val_loss_0.1007.csv'
data_transposed = load_and_process_csv(file_path)
markdown_table = format_as_markdown_table(data_transposed)
print(markdown_table)
