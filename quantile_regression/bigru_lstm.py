import torch 
from torch import nn
import os, sys
import pandas as pd
sys.path.append(os.getcwd())
dataset_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../dataset"))
sys.path.append(dataset_path)
from sklearn.metrics import precision_score, recall_score, classification_report
from classification_model import PriceClassifier
from torch.utils.data import DataLoader
from dataset import SequenceFinancialDataset, get_train_val_test
from tqdm import tqdm
import joblib

class BiGRU_LSTM_Clasiifier():
    def __init__(self, from_pretrained: bool=False , output_size: int=1):
        super().__init__()
        self.from_pretrained = from_pretrained
        self.num_targets = output_size
        self.models = []
        self.feature_names = None
        self.past_x_long = None # seq_lenng (ex: 1000) last x_long datapoint from the past 
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    def fit(self, train_dataset: SequenceFinancialDataset, 
                    val_dataset: SequenceFinancialDataset, 
                    test_dataset: SequenceFinancialDataset, model_name:str='bigru_lstm',
                    checkpoint_path = None):
        #CONFIG MODEL'S ARCHITECHURE
        input_size = next(iter(train_dataset))[1].size(1)
        hidden_bigru_size=100
        hidden_lstm1_size=100 
        hidden_lstm2_size=50
        num_class=3 # Output size
        #PREPARE DATASET
        batch_size=128
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
        # val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
        # test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
        #CONFIG HYPER PARAMS
        num_epochs = 200
        seq_length = 1000
        progress_bar = tqdm(range(num_epochs))
        self.past_x_long = torch.stack([train_dataset[idx][1] for idx in range(seq_length)]) #Load last x_long for inference
        if model_name.lower()=='bigru_lstm':
            model = BiGRU_LSTM(input_size=input_size,
                               hidden_bigru_size=hidden_bigru_size,
                               hidden_lstm1_size=hidden_lstm1_size,
                               hidden_lstm2_size=hidden_lstm2_size,
                               output_size=num_class)
        else: 
            raise ValueError("model not exist!!!")    
        # if self.from_pretrained:
        #     checkpoint = torch.load(checkpoint_path)
        #     model.load_state_dict(checkpoint['weights'])
        model.to(self.device)
        criterion = nn.CrossEntropyLoss()
        model.train()
        for idx in range(self.num_targets):
            print(f"Training Model for target_{idx}")
            running_loss = 0.0
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            for epoch in progress_bar:
                for batch_idx, (x_short, x_long, y, masks, percentages, labels, close_price) in enumerate(train_loader):
                    x_long = x_long.to(self.device)
                    labels = labels.to(self.device)
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
                    torch.save({"weights": model.state_dict(),
                                "x_past_long": self.past_x_long}, f"exps/bigru_lstm_epoch{epoch}_target{idx}.pt")
            self.models.append({"weights": model.state_dict(),
                                "x_past_long": self.past_x_long})
        torch.save(self.models, f"exps/bigru_lstm_all_model.pt")
        print("Training Completed")
    
    def evaluate(self, test_dataset, model_name:str="bigru_lstm", checkpoint_path:str="exps/bigru_lstm_all_model.pt"):
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
        self.load_model(checkpoint_path=checkpoint_path)
        if model_name.lower()=="bigru_lstm":
            input_size = next(iter(test_dataset))[1].size(1)
            hidden_bigru_size=100
            hidden_lstm1_size=100 
            hidden_lstm2_size=50
            num_class=3 # Output size
            model = BiGRU_LSTM(model = BiGRU_LSTM(input_size=input_size,
                               hidden_bigru_size=hidden_bigru_size,
                               hidden_lstm1_size=hidden_lstm1_size,
                               hidden_lstm2_size=hidden_lstm2_size,
                               output_size=num_class))
        else: 
            raise ValueError("model not exist!!!")
        for idx in self.num_targets:
            model.load_state_dict(self.models[idx]['weights'])
            model.to(self.device)
            model.eval()
            with torch.no_grad():
                print("Testing...")
                total_true = 0
                num_samples = 0
                total_loss = 0
                all_preds = []
                all_labels = []
                for batch_idx, (x_short, x_long, y, masks, percentages, labels, close_price) in enumerate(test_loader):
                    x_long = x_long.to(self.device)
                    labels = labels.to(self.device)
                    out = model(x_long)
                    loss = criterion(out[:, -1], labels[:, -1, idx]+1)
                    preds = torch.argmax(out[:, -1], dim=1)
                    total_true+=(preds==labels[:, -1, idx]+1).sum().item()
                    num_samples+=len(labels)
                    total_loss+=loss.item()
                    all_preds.extend(preds.cpu().numpy()-1)
                    all_labels.extend(labels[:, -1, idx].cpu().numpy())
                print(f"Total Loss: {total_loss/batch_idx+1}")
                print(f"Accuracy: {total_true/num_samples:.2f}")
                print(classification_report(all_labels, all_preds))
                print(f"Precision (Macro): {precision_score(all_labels, all_preds, average='macro'):.2f}")
                print(f"Recall (Macro): {recall_score(all_labels, all_preds, average='macro'):.2f}")
                print(f"Precision (Micro): {precision_score(all_labels, all_preds, average='micro'):.2f}")
                print(f"Recall (Micro): {recall_score(all_labels, all_preds, average='micro'):.2f}")        

    def forward(self, x_short: torch.Tensor ,x_long: torch.Tensor, model_name:str="bigru_lstm"):
        ### WARNING: Currently, the model on ly can inference for target 0, the code will be update in the future to help model can inference num_targets
        """"
        args: 
        x_short: new x short data,
        x_long: new x long data
        """
        self.load_model()
        if model_name.lower()=="bigru_lstm":
            input_size = next(iter(test_dataset))[1].size(1)
            hidden_bigru_size=100
            hidden_lstm1_size=100 
            hidden_lstm2_size=50
            num_class=3 # Output size
            model = BiGRU_LSTM(model = BiGRU_LSTM(input_size=input_size,
                            hidden_bigru_size=hidden_bigru_size,
                            hidden_lstm1_size=hidden_lstm1_size,
                            hidden_lstm2_size=hidden_lstm2_size,
                            output_size=num_class))
        model.load_state_dict(self.models[0]['weights'])
        num_new_data = x_long.size(0) # Get index form trainset to start slicing
        # last_x_long = train_dataset[len(train_dataset)][1] # idex-1 to get last item in df,  index 1 is x_long
        sliced_x_long = self.past_x_long[:len(self.past_x_long)-num_new_data]
        input_data = torch.concat([sliced_x_long, x_long], dim=0).unsqueeze(0)
        outs = model(input_data)
        prediction = torch.argmax(outs[:, -1, :], dim=-1)
        return prediction
    
    def load_model(self, checkpoint_path: str="exps/bigru_lstm_all_model.pt"):
        self.models = torch.load(checkpoint_path)
        self.past_x_long = self.models[0]['x_past_long']

class BiGRU_LSTM(nn.Module):
    def __init__(self, input_size, hidden_bigru_size, hidden_lstm1_size, hidden_lstm2_size, output_size, dropout=0.2, num_targets=1):
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
        self.num_targets = num_targets #(num_ouputs)
        self.models = [] #save model for each target
        self.last_x_long = None
    
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
    dataset, train_dataset, val_dataset, test_dataset, _, _, _, = get_train_val_test()
    #MODEL ARCHITECTURE
    input_size = next(iter(train_dataset))[1].size(1)
    hidden_bigru_size=100
    hidden_lstm1_size=100 
    hidden_lstm2_size=50
    output_size=3
    model = BiGRU_LSTM(input_size, hidden_bigru_size, hidden_lstm1_size, hidden_lstm2_size, output_size)
    batch_size = 128
    num_epochs = 200
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model.to(device)
    progress_bar = tqdm(range(num_epochs))
    # for idx in range(17):
    #     model.train()
    #     print(f"Training Model for label target_{idx}")
    #     running_loss = 0.0
    #     total_batch = len(train_loader)
    #     for epoch in progress_bar:
    #         for batch_idx, (x_short, x_long, y, masks, percentages, labels, close_price) in enumerate(train_loader):
    #             x_long = x_long.to(device)
    #             labels = labels.to(device)
    #             # print(f"Epoch {epoch} Batch {batch_idx}/{total_batch}")
    #             out = model(x_long)
    #             loss = criterion(out[:, -1], labels[:, -1, idx]+1)
    #             optimizer.zero_grad()
    #             loss.backward()
    #             torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
    #             optimizer.step()
    #             running_loss += loss.item()
    #         epoch_loss = running_loss/(batch_idx+1)
    #         running_loss=0
    #         progress_bar.set_postfix(loss=f"{epoch_loss:.4f}")
    #         if epoch%5 == 0:
    #             #Saving Model each 5 epochs
    #             os.makedirs("exps", exist_ok=True)
    #             joblib.dump(model, f"exps/bigru_lstm_epoch{epoch}.pkl")
    checkpoint = joblib.load('bigru_lstm_epoch195.pkl')
    print(checkpoint)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    with torch.no_grad():
        print("Testing...")
        total_true = 0
        num_samples = 0
        all_preds = []
        all_labels = []
        for x_short, x_long, y, masks, percentages, labels, close_price in test_loader:
            x_long = x_long.to(device)
            labels = labels.to(device)
            out = model(x_long)
            loss = criterion(out[:, -1], labels[:, -1, 0]+1)
            preds = torch.argmax(out[:, -1], dim=1)
            total_true +=(preds==labels[:, -1, 0]+1).sum().item()
            num_samples += len(labels)
            all_preds.extend(preds.cpu().numpy()-1)
            all_labels.extend(labels[:, -1, 0].cpu().numpy())
        # pre = precision_score(preds, labels[:, -1, idx]+1)
        # recall = recall_score(preds)
        print(f"(Validate) Accuracy {total_true/num_samples:.2f}")
        print(classification_report(all_labels, all_preds))
        print(f"(Validate) Precision {precision_score(all_labels, all_preds, average='macro'):.2f}")
        print(f"(Validate) Recall {recall_score(all_labels, all_preds, average='macro'):.2f}")

                # : Precision {pre:.2f} : Recall {recall:.2f}")

