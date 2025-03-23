import torch 
from torch import nn
import os, sys
import pandas as pd
sys.path.append(os.getcwd())
dataset_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../dataset"))
sys.path.append(dataset_path)
from sklearn.metrics import precision_score, recall_score
from classification_model import PriceClassifier
from torch.utils.data import DataLoader
from dataset import SequenceFinancialDataset, get_train_val_test
from tqdm import tqdm
import joblib
class BiGRU_LSTM(nn.Module):
    def __init__(self, input_size, hidden_bigru_size, hidden_lstm1_size, hidden_lstm2_size, output_size, dropout=0.2):
        super(BiGRU_LSTM, self).__init__()
        self.output_size = output_size
        self.bigru = nn.GRU(input_size=input_size,
                            hidden_size=hidden_bigru_size,
                            batch_first=True,
                            bidirectional=True)
        self.lstm1 = nn.LSTM(input_size=hidden_bigru_size*2,
                            hidden_size=hidden_lstm1_size,
                            batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.lstm2 = nn.LSTM(input_size=hidden_lstm1_size,
                            hidden_size=hidden_lstm2_size,
                            num_layers=2,
                            batch_first=True,
                            dropout=dropout)
        self.fc = nn.Linear(in_features=hidden_lstm1_size, out_features=output_size)
        # self.relu = nn.ReLU()
    def forward(self, x): 
        # print(torch.isnan(x).sum())  # Count NaNs
        # print(x.max(), x.min())  # Check extreme values
        out, _ = self.bigru(x)
        out = self.dropout(out)
        out, _ = self.lstm1(out)
        out = self.dropout(out)
        # out, _ = self.lstm2(out)
        return self.fc(out)

if __name__=="__main__":
    seq_length = 1000
    dataset, train_loader, val_loader, test_loader, _, _, _, = get_train_val_test()
    #MODEL ARCHITECTURE
    input_size = next(iter(train_loader))[1].size(1)
    hidden_bigru_size=100
    hidden_lstm1_size=100 
    hidden_lstm2_size=50
    output_size=3
    model = BiGRU_LSTM(input_size, hidden_bigru_size, hidden_lstm1_size, hidden_lstm2_size, output_size)
    batch_size = 128
    num_epochs = 200
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = DataLoader(train_loader, batch_size=batch_size, shuffle=False, drop_last=False)
    val_loader = DataLoader(val_loader, batch_size=batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_loader, batch_size=batch_size, shuffle=False, drop_last=False)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model.to(device)
    progress_bar = tqdm(range(num_epochs))
    for idx in range(17):
        model.train()
        print(f"Training Model for label target_{idx}")
        running_loss = 0.0
        total_batch = len(train_loader)
        for epoch in progress_bar:
            for batch_idx, (x_short, x_long, y, masks, percentages, labels, close_price) in enumerate(train_loader):
                x_long = x_long.to(device)
                labels = labels.to(device)
                # print(f"Epoch {epoch} Batch {batch_idx}/{total_batch}")
                out = model(x_long)
                loss = criterion(out[:, -1], labels[:, -1, idx]+1)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
                optimizer.step()
                running_loss += loss.item()
            epoch_loss = running_loss/(batch_idx+1)
            running_loss=0
            progress_bar.set_postfix(loss=f"{epoch_loss:.4f}")
            if epoch%5 == 0:
                #Saving Model each 5 epochs
                os.makedirs("exps", exist_ok=True)
                joblib.dump(model, f"exps/bigru_lstm_epoch{epoch}.pkl")
        model.eval()
        with torch.no_grad():
            print("Validating...")
            total_true = 0
            num_samples = 0
            all_preds = []
            all_labels = []
            for x_short, x_long, y, masks, percentages, labels, close_price in val_loader:
                x_long = x_long.to(device)
                labels = labels.to(device)
                out = model(x_long)
                loss = criterion(out[:, -1], labels[:, -1, idx]+1)
                preds = torch.argmax(out[:, -1], dim=1)
                total_true +=(preds==labels[:, -1, idx]+1).sum().item()
                num_samples += len(labels)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
            # pre = precision_score(preds, labels[:, -1, idx]+1)
            # recall = recall_score(preds)
            print(f"(Validate) Accuracy {total_true/num_samples:.2f}")
            print(f"(Validate) Precision {precision_score(all_labels, all_preds):.2f}")
            print(f"(Validate) Recall {recall_score(all_labels, all_preds):.2f}")

                # : Precision {pre:.2f} : Recall {recall:.2f}")

