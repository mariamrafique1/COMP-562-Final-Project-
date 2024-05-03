import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support, accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Model Definition
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# save metrics to the directory
images_dir = Path('LSTM_metrics')
images_dir.mkdir(exist_ok=True)

# Load and prepare data
def load_and_prepare_data(file_path):
    df = pd.read_excel(file_path, header=1)
    df.columns = df.columns.str.strip()
    X = df.drop(['Timestamp', 'Normal/Attack'], axis=1)
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df['Normal/Attack'])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = X_scaled.reshape(X_scaled.shape[0], 1, X_scaled.shape[1])
    return torch.tensor(X_scaled).float(), torch.tensor(y).long()

# Training
def train_lstm(model, data_tensor, labels, epochs, batch_size):
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=0.001)
    model.train()
    for epoch in range(epochs):
        for i in range(0, data_tensor.size(0), batch_size):
            batch_data = data_tensor[i:min(i + batch_size, data_tensor.size(0))]
            batch_labels = labels[i:min(i + batch_size, labels.size(0))]
            outputs = model(batch_data)
            loss = criterion(outputs, batch_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# Plot and save metrics
def plot_and_save_metrics(metrics, model_name='LSTMModel'):
    categories = ['Precision', 'Recall', 'F1 Score']
    values_0 = [metrics['Precision_0'], metrics['Recall_0'], metrics['F1_score_0']]
    values_1 = [metrics['Precision_1'], metrics['Recall_1'], metrics['F1_score_1']]
    combined_values = [(v0 + v1) / 2 for v0, v1 in zip(values_0, values_1)]

    fig, ax = plt.subplots()
    x = np.arange(len(categories))  
    width = 0.35  

    rects1 = ax.bar(x - width/2, values_0, width, label='Class 0 (Attack)', color='lightblue')
    rects2 = ax.bar(x + width/2, values_1, width, label='Class 1 (Normal)', color='lightgreen')

    ax.set_ylabel('Scores')
    ax.set_title('Performance Metrics by class')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()

    fig.tight_layout()
    plt.savefig(images_dir / f'{model_name}_performance_metrics_by_class.png')
    plt.close()

    
    plt.figure(figsize=(8, 6))
    plt.bar(categories, combined_values, color='grey')
    plt.title('Combined Metrics')
    plt.ylabel('Average Score')
    plt.savefig(images_dir / f'{model_name}_combined_metrics.png')
    plt.close()


    cm = metrics['Confusion Matrix']
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(images_dir / f'{model_name}_confusion_matrix.png')
    plt.close()

   
    print("\nClassification Report:")
    print(metrics['Classification Report'])
    report_df = pd.DataFrame(metrics['Classification Report']).transpose()
    report_df.to_csv(images_dir / f'{model_name}_classification_report.csv')


if __name__ == '__main__':
    file_path = 'SWaT_Dataset_Attack_v0.xlsx'
    X_tensor, y_tensor = load_and_prepare_data(file_path)
    X_train, X_test, y_train, y_test = train_test_split(X_tensor.numpy(), y_tensor.numpy(), test_size=0.2, random_state=42)

    X_train_tensor = torch.tensor(X_train).float()
    y_train_tensor = torch.tensor(y_train).long()
    X_test_tensor = torch.tensor(X_test).float()
    y_test_tensor = torch.tensor(y_test).long()

    model = LSTMModel(input_size=51, hidden_size=64, num_layers=1, num_classes=2)
    train_lstm(model, X_train_tensor, y_train_tensor, epochs=50, batch_size=64)

   
    model.eval()
    with torch.no_grad():
        outputs = model(X_test_tensor)
        _, predicted = torch.max(outputs.data, 1)
        precision, recall, fscore, _ = precision_recall_fscore_support(y_test_tensor.numpy(), predicted.numpy(), average=None, labels=[0, 1])
        accuracy = accuracy_score(y_test_tensor.numpy(), predicted.numpy())
        cm = confusion_matrix(y_test_tensor.numpy(), predicted.numpy())
        cr = classification_report(y_test_tensor.numpy(), predicted.numpy(), output_dict=True)

        metrics = {
            'Precision_0': precision[0],
            'Recall_0': recall[0],
            'F1_score_0': fscore[0],
            'Precision_1': precision[1],
            'Recall_1': recall[1],
            'F1_score_1': fscore[1],
            'Accuracy': accuracy,
            'Confusion Matrix': cm,
            'Classification Report': cr
        }

        plot_and_save_metrics(metrics)
