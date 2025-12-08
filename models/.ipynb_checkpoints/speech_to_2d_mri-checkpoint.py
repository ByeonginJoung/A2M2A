import torch
import torch.nn as nn

class Speech2MRI2D(nn.Module):
    def __init__(self,
                 args,
                 n_mgc,
                 n_width,
                 n_height
    ):
        super().__init__()
        self.time_distributed_1 = nn.Linear(n_mgc, 575)
        self.time_distributed_2 = nn.Linear(575, 575)
        self.time_distributed_3 = nn.Linear(575, 575)
        self.lstm_1 = nn.LSTM(575, 575, batch_first=True)
        self.lstm_2 = nn.LSTM(575, 575, batch_first=True)
        self.dense1 = nn.Linear(575, 2048)
        self.dense2 = nn.Linear(2048, 1024)
        self.dense3 = nn.Linear(1024, 575)
        self.dense = nn.Linear(575, n_width * n_height)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.kaiming_normal_(param, nonlinearity='relu')
                    elif 'bias' in name:
                        nn.init.zeros_(param)
        
    def forward(self, x):
        # TimeDistributed layers
        x = torch.relu(self.time_distributed_1(x))
        x = torch.relu(self.time_distributed_2(x))
        x = torch.relu(self.time_distributed_3(x))
        
        # LSTM layers
        x1, _ = self.lstm_1(x)
        x2, _ = self.lstm_2(x1)
        
        # Last Dense layer
        x = torch.relu(self.dense1(x2))
        x = torch.relu(self.dense2(x))
        x = torch.relu(self.dense3(x))
        x = torch.sigmoid(self.dense(x))  # Using only the last output of the LSTM
        
        return x
