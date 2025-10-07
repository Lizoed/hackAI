import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import random
import scipy
from scipy.signal import butter, filtfilt
import os
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

filter_ = butter(5, [0.6 / 30 * 2, 10 / 30 * 2], btype='bandpass')

def process_ppg(ppg):
    ppg = scipy.signal.filtfilt(filter_[0], filter_[1], ppg)
    ppg = (ppg - ppg.mean()) / ppg.std()
    return ppg

class HDF5FolderDataset(Dataset):
    def __init__(self, csv_path, root_dir, begin=0, end=1, is_training=True):
        full_metadata = pd.read_csv(csv_path)
        full_metadata = full_metadata[full_metadata['num_chunks'] > 0]
        full_metadata['data_hdf5_path'] = full_metadata['data_hdf5_path'].str.replace('\\', '/')

        start_index = int(begin * len(full_metadata))
        end_index = int(end * len(full_metadata))

        self.metadata = full_metadata.iloc[start_index:end_index].reset_index(drop=True)

        self.root_dir = root_dir
        self.param_keys = ['dbp_mean_experiment', 'sbp_mean_experiment']
        self.is_training = is_training

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        hdf5_relative_path = row['data_hdf5_path']

        possible_paths = [
            hdf5_relative_path,
            os.path.join(self.root_dir, hdf5_relative_path),
            os.path.join(self.root_dir, os.path.basename(hdf5_relative_path)),
            hdf5_relative_path.replace('\\', '/')
        ]
        
        hdf5_path = None
        for path in possible_paths:
            if os.path.exists(path):
                hdf5_path = path
                break
        
        if hdf5_path is None:
            print(f"Warning: File not found for {hdf5_relative_path}")
            empty_data = torch.zeros((1, 300))
            if self.is_training:
                return empty_data, torch.tensor(0.0), torch.tensor(0.0)
            else:
                return empty_data, row['experiment_guid']

        try:
            with h5py.File(hdf5_path, 'r') as f:
                if len(f.keys()) == 0:
                    empty_data = torch.zeros((1, 300))
                    if self.is_training:
                        return empty_data, torch.tensor(0.0), torch.tensor(0.0)
                    else:
                        return empty_data, row['experiment_guid']
                
                try:
                    sorted_keys = sorted(f.keys(), key=lambda k: int(k.split('_')[1]))
                except:
                    sorted_keys = sorted(f.keys())
                
                data_list = [f[key][:] for key in sorted_keys]

            stacked_data = torch.from_numpy(np.stack(data_list)).float()

            if self.is_training:
                param1 = float(row[self.param_keys[0]])
                param2 = float(row[self.param_keys[1]])
                param1_tensor = torch.tensor(param1, dtype=torch.float32)
                param2_tensor = torch.tensor(param2, dtype=torch.float32)
                return stacked_data, param1_tensor, param2_tensor
            else:
                return stacked_data, row['experiment_guid']
                
        except Exception as e:
            print(f"Error loading {hdf5_path}: {e}")
            empty_data = torch.zeros((1, 300))
            if self.is_training:
                return empty_data, torch.tensor(0.0), torch.tensor(0.0)
            else:
                return empty_data, row['experiment_guid']

def collate_data_train(batch):
    res1 = []
    res2 = []
    res3 = []

    f = np.vectorize(process_ppg, signature='(n)->(m)')

    for i in batch:
        ppg, dbp, sbp = i
        l = len(ppg)
        if l > 1:
            if l > 10:
                indices = np.random.choice(l, 10, replace=False)
                ppg_subset = ppg[indices]
            else:
                ppg_subset = ppg

            try:
                ppg_processed = f(ppg_subset.numpy())
                res1.append(ppg_processed)
                res2.append(dbp)
                res3.append(sbp)
            except:
                continue

    if res1:
        max_segments = max(x.shape[0] for x in res1)
        max_length = max(x.shape[1] for x in res1)

        batch_size = len(res1)
        padded_res1 = torch.zeros(batch_size, max_segments, max_length)
        
        for i, x in enumerate(res1):
            num_segments, seg_length = x.shape
            if seg_length < max_length:
                pad_right = max_length - seg_length
                x_padded = np.pad(x, ((0, 0), (0, pad_right)), mode='constant')
            else:
                x_padded = x[:, :max_length]
            
            if num_segments < max_segments:
                repeat_times = (max_segments + num_segments - 1) // num_segments
                x_repeated = np.tile(x_padded, (repeat_times, 1))
                x_final = x_repeated[:max_segments]
            else:
                x_final = x_padded[:max_segments]
            
            padded_res1[i] = torch.from_numpy(x_final)

        res2_tensor = torch.stack(res2)
        res3_tensor = torch.stack(res3)
        
        return padded_res1, res2_tensor, res3_tensor
    else:
        return torch.tensor([]), torch.tensor([]), torch.tensor([])

def collate_data_test(batch):
    res1 = []
    res2 = []

    f = np.vectorize(process_ppg, signature='(n)->(m)')

    for i in batch:
        ppg, guid = i
        if len(ppg) > 0 and len(ppg) > 1:
            try:
                ppg_processed = f(ppg.numpy())
                res1.append(ppg_processed)
                res2.append(guid)
            except:
                continue

    return res1, res2

class PPGBloodPressureModel(nn.Module):
    def __init__(self, input_length=300, num_segments=10):
        super(PPGBloodPressureModel, self).__init__()
        
        self.segment_cnn = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.MaxPool1d(2),
            
            nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.MaxPool1d(2),
            
            nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.MaxPool1d(2),
            
            nn.AdaptiveAvgPool1d(1)
        )
        
        self.lstm = nn.LSTM(128, 64, batch_first=True, bidirectional=True)
        
        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 2)
        )
        
    def forward(self, x):
        batch_size, num_segments, seq_length = x.shape
        
        segment_features = []
        for i in range(num_segments):
            segment = x[:, i, :].unsqueeze(1)
            features = self.segment_cnn(segment)
            features = features.squeeze(-1)
            segment_features.append(features)
        
        segment_features = torch.stack(segment_features, dim=1)
        
        lstm_out, (h_n, c_n) = self.lstm(segment_features)
        
        if self.lstm.bidirectional:
            h_n = torch.cat([h_n[-2], h_n[-1]], dim=1)
        else:
            h_n = h_n[-1]
        
        output = self.fc(h_n)
        
        return output[:, 0], output[:, 1]

def calculate_accuracy(y_true, y_pred, tolerance=10):
    correct = np.abs(y_true - y_pred) <= tolerance
    return np.mean(correct)

def evaluate_model(model, dataloader, device, criterion):
    model.eval()
    total_loss = 0
    sbp_preds = []
    dbp_preds = []
    sbp_true = []
    dbp_true = []
    
    with torch.no_grad():
        for batch in dataloader:
            ppg, dbp, sbp = batch
            if len(ppg) == 0:
                continue
                
            ppg = ppg.to(device)
            dbp = dbp.to(device)
            sbp = sbp.to(device)
            
            sbp_pred, dbp_pred = model(ppg)
            
            loss_sbp = criterion(sbp_pred, sbp)
            loss_dbp = criterion(dbp_pred, dbp)
            loss = (loss_sbp + loss_dbp) / 2
            
            total_loss += loss.item()
            
            sbp_preds.extend(sbp_pred.cpu().numpy())
            dbp_preds.extend(dbp_pred.cpu().numpy())
            sbp_true.extend(sbp.cpu().numpy())
            dbp_true.extend(dbp.cpu().numpy())
    
    if len(sbp_preds) == 0:
        return {
            'total_loss': 0,
            'sbp_mse': 0, 'dbp_mse': 0,
            'sbp_mae': 0, 'dbp_mae': 0,
            'sbp_accuracy': 0, 'dbp_accuracy': 0,
            'sbp_preds': [], 'dbp_preds': [],
            'sbp_true': [], 'dbp_true': []
        }
    
    sbp_preds = np.array(sbp_preds)
    dbp_preds = np.array(dbp_preds)
    sbp_true = np.array(sbp_true)
    dbp_true = np.array(dbp_true)
    
    sbp_mse = mean_squared_error(sbp_true, sbp_preds)
    dbp_mse = mean_squared_error(dbp_true, dbp_preds)
    sbp_mae = mean_absolute_error(sbp_true, sbp_preds)
    dbp_mae = mean_absolute_error(dbp_true, dbp_preds)
    
    sbp_accuracy = calculate_accuracy(sbp_true, sbp_preds)
    dbp_accuracy = calculate_accuracy(dbp_true, dbp_preds)
    
    return {
        'total_loss': total_loss / len(dataloader),
        'sbp_mse': sbp_mse,
        'dbp_mse': dbp_mse,
        'sbp_mae': sbp_mae,
        'dbp_mae': dbp_mae,
        'sbp_accuracy': sbp_accuracy,
        'dbp_accuracy': dbp_accuracy,
        'sbp_preds': sbp_preds,
        'dbp_preds': dbp_preds,
        'sbp_true': sbp_true,
        'dbp_true': dbp_true
    }

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    batch_size = 16
    learning_rate = 0.001
    num_epochs = 30
    
    train_csv_path = '/kaggle/input/metadata/train.csv'
    test_csv_path = '/kaggle/input/metadata/test.csv'
    data_root_dir = '/kaggle/input/experiment-data/experiment_data'
    
    possible_test_dirs = [
        '/kaggle/input/experiment-data-test/experiment_data_test',
        '/kaggle/input/experiment-data-test',
        '/kaggle/input/experiment_data_test',
        '/kaggle/input/experiment-data/experiment_data_test',
        data_root_dir
    ]
    
    test_data_root_dir = None
    for test_dir in possible_test_dirs:
        if os.path.exists(test_dir):
            test_data_root_dir = test_dir
            print(f"Found test data directory: {test_dir}")
            break
    
    if test_data_root_dir is None:
        print("Warning: Test data directory not found. Using train directory for test data.")
        test_data_root_dir = data_root_dir
    
    print("Checking files")
    print(f"Train data dir exists: {os.path.exists(data_root_dir)}")
    print(f"Test data dir exists: {os.path.exists(test_data_root_dir)}")
    print(f"Train CSV exists: {os.path.exists(train_csv_path)}")
    print(f"Test CSV exists: {os.path.exists(test_csv_path)}")
    
    train_dataset = HDF5FolderDataset(train_csv_path, data_root_dir, begin=0, end=0.8, is_training=True)
    val_dataset = HDF5FolderDataset(train_csv_path, data_root_dir, begin=0.8, end=1.0, is_training=True)
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Val dataset size: {len(val_dataset)}")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                            collate_fn=collate_data_train, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                          collate_fn=collate_data_train, num_workers=2)
    
    model = PPGBloodPressureModel().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    
    train_losses = []
    val_losses = []
    
    print("Starting training")
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        batch_count = 0
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for batch_idx, batch in enumerate(progress_bar):
            ppg, dbp, sbp = batch
            if len(ppg) == 0:
                continue
                
            ppg = ppg.to(device)
            dbp = dbp.to(device)
            sbp = sbp.to(device)
            
            optimizer.zero_grad()
            sbp_pred, dbp_pred = model(ppg)
            
            loss_sbp = criterion(sbp_pred, sbp)
            loss_dbp = criterion(dbp_pred, dbp)
            loss = (loss_sbp + loss_dbp) / 2
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            batch_count += 1
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        if batch_count > 0:
            avg_train_loss = epoch_loss / batch_count
        else:
            avg_train_loss = 0
            
        train_losses.append(avg_train_loss)
        
        if len(val_loader) > 0:
            val_metrics = evaluate_model(model, val_loader, device, criterion)
            if len(val_metrics['sbp_true']) > 0:
                val_losses.append(val_metrics['total_loss'])
                
                scheduler.step(val_metrics['total_loss'])
                
                print(f'Epoch {epoch+1}/{num_epochs}:')
                print(f'  Train Loss: {avg_train_loss:.4f}')
                print(f'  Val Loss: {val_metrics["total_loss"]:.4f}')
                print(f'  SBP - MSE: {val_metrics["sbp_mse"]:.2f}, MAE: {val_metrics["sbp_mae"]:.2f}, Acc: {val_metrics["sbp_accuracy"]:.3f}')
                print(f'  DBP - MSE: {val_metrics["dbp_mse"]:.2f}, MAE: {val_metrics["dbp_mae"]:.2f}, Acc: {val_metrics["dbp_accuracy"]:.3f}')
            else:
                print(f'Epoch {epoch+1}/{num_epochs}: Train Loss: {avg_train_loss:.4f} (No validation data available)')
        else:
            print(f'Epoch {epoch+1}/{num_epochs}: Train Loss: {avg_train_loss:.4f} (No validation data)')
        print()
    
    if train_losses and val_losses:
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training and Validation Loss')
        
        if len(val_loader) > 0:
            val_metrics = evaluate_model(model, val_loader, device, criterion)
            if len(val_metrics['sbp_true']) > 0:
                plt.subplot(1, 2, 2)
                plt.scatter(val_metrics['sbp_true'], val_metrics['sbp_preds'], alpha=0.6, label='SBP')
                plt.scatter(val_metrics['dbp_true'], val_metrics['dbp_preds'], alpha=0.6, label='DBP')
                plt.plot([80, 180], [80, 180], 'k--', alpha=0.5)
                plt.xlabel('True Values')
                plt.ylabel('Predicted Values')
                plt.legend()
                plt.title('True vs Predicted Blood Pressure')
        plt.tight_layout()
        plt.show()
    
    return model, test_data_root_dir

def predict_test_data(model, test_csv_path, test_data_root_dir, output_csv_path, device):
    
    test_df = pd.read_csv(test_csv_path)
    print(f"Test dataset shape: {test_df.shape}")
    print(f"Unique experiments in test: {test_df['experiment_guid'].nunique()}")
    
    test_dataset = HDF5FolderDataset(test_csv_path, test_data_root_dir, is_training=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, 
                           collate_fn=collate_data_test, num_workers=2)
    
    model.eval()
    predictions = {}
    
    print("Making predictions on test data")
    with torch.no_grad():
        for batch in tqdm(test_loader):
            ppg_list, guid_list = batch
            
            for ppg_segments, guid in zip(ppg_list, guid_list):
                if len(ppg_segments) == 0 or ppg_segments.shape[0] == 0:
                    predictions[guid] = (120.0, 80.0)
                    continue
                
                max_segments = ppg_segments.shape[0]
                max_length = ppg_segments.shape[1]
                
                ppg_tensor = torch.zeros(1, max_segments, max_length)
                ppg_tensor[0, :max_segments, :max_length] = torch.from_numpy(ppg_segments)
                
                ppg_tensor = ppg_tensor.to(device)
                
                try:
                    sbp_pred, dbp_pred = model(ppg_tensor)
                    predictions[guid] = (sbp_pred.item(), dbp_pred.item())
                except:
                    predictions[guid] = (120.0, 80.0)
    
    for guid, (sbp, dbp) in predictions.items():
        test_df.loc[test_df['experiment_guid'] == guid, 'sbp_mean_experiment'] = sbp
        test_df.loc[test_df['experiment_guid'] == guid, 'dbp_mean_experiment'] = dbp
    
    test_df.to_csv(output_csv_path, index=False)
    print(f"Predictions saved to {output_csv_path}")
    
    print(f"Predicted SBP range: {test_df['sbp_mean_experiment'].min():.1f} - {test_df['sbp_mean_experiment'].max():.1f}")
    print(f"Predicted DBP range: {test_df['dbp_mean_experiment'].min():.1f} - {test_df['dbp_mean_experiment'].max():.1f}")
    
    return test_df

if __name__ == "__main__":
    train_csv_path = '/kaggle/input/metadata/train.csv'
    test_csv_path = '/kaggle/input/metadata/test.csv'
    output_csv_path = '/kaggle/working/test_predictions.csv'
    
    trained_model, test_data_root_dir = main()
    
    test_predictions = predict_test_data(trained_model, test_csv_path, test_data_root_dir, 
                                       output_csv_path, 
                                       torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    
    print("Training and prediction completed")
    print(f"Results saved to: {output_csv_path}")
