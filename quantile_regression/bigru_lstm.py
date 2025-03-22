import torch 
from torch import nn
import os, sys
import pandas as pd
sys.path.append(os.getcwd())
dataset_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../dataset"))
sys.path.append(dataset_path)
from classification_model import PriceClassifier
from torch.utils.data import DataLoader
from dataset import SequenceFinancialDataset
class BiGRU_LSTM(nn.Module):
    def __init__(self, input_size, hidden_bigru_size, hidden_lstm1_size, hidden_lstm2_size, output_size, dropout=0.2):
        super(BiGRU_LSTM, self).__init__()
        self.output_size = output_size
        self.bigru = nn.GRU(input_size=input_size,
                            hidden_size=hidden_bigru_size,
                            batch_first=True,
                            dropout=dropout,
                            bidirectional=True)
        self.lstm1 = nn.LSTM(input_size=hidden_bigru_size*2,
                            hidden_size=hidden_lstm1_size,
                            batch_first=True,
                            dropout=dropout)
        self.lstm2 = nn.LSTM(input_size=hidden_lstm1_size,
                            hidden_size=hidden_lstm2_size,
                            num_layers=2,
                            batch_first=True,
                            dropout=dropout)
        self.fc = nn.Linear(in_features=hidden_lstm2_size, out_features=output_size)
    def forward(self, x):
        out, _ = self.bigru(x)
        out, _ = self.lstm1(out)
        out, _ = self.lstm2(out)
        return self.fc(out)

if __name__=="__main__":
    df = pd.read_csv("data/all_data.csv")
    seq_length = 1000
    dataset = SequenceFinancialDataset(df, seq_length=seq_length)
    #MODEL ARCHITECTURE
    input_size = dataset.X_long_term.shape[1]
    hidden_bigru_size=100
    hidden_lstm1_size=100 
    hidden_lstm2_size=50
    output_size=3
    model = BiGRU_LSTM(input_size, hidden_bigru_size, hidden_lstm1_size, hidden_lstm2_size, output_size)
    batch_size = 32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.to(device)
    for x_short, x_long, y, masks, percentages, labels, close_price in train_loader:
        x_long.to(device)
        out = model(x_long)
        print(out.size())
        break