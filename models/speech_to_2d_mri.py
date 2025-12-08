import torch
import torch.nn as nn
import torch.nn.functional as F

class LinearLayer(nn.Module):
    def __init__(self, dim_in, dim_out, use_bn=True):
        super().__init__()

        self.linear = nn.Linear(dim_in, dim_out)

    def forward(self, x):
        x = torch.relu(self.linear(x))

        return x

class CNNLayer(nn.Module):
    def __init__(self, dim_in, dim_out, residual=True, use_bn=True):
        super().__init__()

        self.use_bn = use_bn
        self.residual = residual
        self.cnn = nn.Conv2d(dim_in, dim_out, 5, 3, 1)

        if use_bn:
            self.bn = nn.BatchNorm2d(dim_out)

        if self.residual:
            # only for dim_in == dim_out
            self.cnn2 = nn.Conv2d(dim_out, dim_out, 3, 1, 1)

    def forward(self, x):

        if self.use_bn:
            x = torch.relu(self.bn(self.cnn(x)))
        else:
            x = torch.relu(self.cnn(x))
        
        if self.residual:
            x = self.cnn2(x) + x
        return x

def warp_image(initial_image, deformation_field):
    # initial_image: [B, 1, H, W]
    # deformation_field: [B, 2, H, W] (displacement vectors for each pixel)
    
    # Generate a normalized grid (identity transform)
    batch_size, _, H, W = initial_image.shape
    grid_y, grid_x = torch.meshgrid(torch.arange(H), torch.arange(W))
    grid = torch.stack((grid_x, grid_y), dim=-1).float()  # Shape: [H, W, 2]
    grid = grid.unsqueeze(0).repeat(batch_size, 1, 1, 1).to(initial_image.device)
    
    # Add deformation field to the grid
    sampling_grid = grid + deformation_field.permute(0, 2, 3, 1)  # Add displacement
    sampling_grid = (2.0 * sampling_grid / torch.tensor([W - 1, H - 1]).to(initial_image.device)) - 1.0  # Normalize to [-1, 1]
    
    # Apply grid sampling
    warped_image = F.grid_sample(initial_image, sampling_grid, mode='bilinear', padding_mode='border')
    return warped_image
    
class Speech2MRI2D(nn.Module):
    def __init__(self,
                 args,
                 n_mgc=25,
                 n_width=84,
                 n_height=84,
                 dropout_rate=0.5
    ):
        super().__init__()

        self.args = args
        n_feats = self.args.model.n_feats
        self.n_width = n_width
        self.n_height = n_height

        self.time_distributed = nn.Sequential(
            LinearLayer(n_mgc, n_feats, use_bn=args.model.use_bn),
            LinearLayer(n_feats, n_feats, use_bn=args.model.use_bn),
            LinearLayer(n_feats, n_feats, use_bn=args.model.use_bn)
            )
        
        if args.model.use_deform or args.model.use_prev_frame:
            self.img_enc = nn.Sequential(
                CNNLayer(1, n_feats // 8, residual=args.model.residual, use_bn=args.model.use_bn),
                CNNLayer(n_feats // 8, n_feats // 2, residual=args.model.residual, use_bn=args.model.use_bn),
                CNNLayer(n_feats // 2, n_feats, residual=args.model.residual, use_bn=args.model.use_bn)
                )
            
            enc_input_len = int(self.args.data.lookback * self.args.data.fps_control_ratio)

            if self.args.dataset_type == 'timit':
                add_channel = 4
            elif self.args.dataset_type == '75-speaker':
                add_channel = 9
            elif self.args.dataset_type == '75-speaker-word':
                add_channel = 9
            elif self.args.dataset_type == '75-speaker-multi':
                add_channel = 9
            elif self.args.dataset_type == '75-speaker-multi-word':
                add_channel = 9
            
            self.cat_feats = nn.Sequential(
                nn.Conv1d(enc_input_len + add_channel, enc_input_len, 3, 1, 1),
                nn.BatchNorm1d(enc_input_len),
                nn.ReLU(),
                nn.Conv1d(enc_input_len, enc_input_len, 1, 1, 0)
                )
        
        if self.args.model.use_lstm:
            self.lstm_1 = nn.LSTM(n_feats, n_feats, batch_first=True, num_layers=2)
            self.lstm_2 = nn.LSTM(n_feats, n_feats, batch_first=True, num_layers=2)

        if self.args.model.use_transformer:
            self.transformer_layer = nn.TransformerEncoderLayer(d_model=n_feats, nhead=self.args.model.n_head)
            self.transformer = nn.TransformerEncoder(self.transformer_layer, num_layers=6)

        self.dense0 = nn.Linear(n_feats, n_feats)

        self.final_channel = 2 if args.model.use_deform else 1
        
        self.dense1 = nn.Linear(n_feats, n_width * n_height * self.final_channel)#LinearLayer(n_feats, n_feats, use_bn=args.model.use_bn)

        if self.args.model.use_dropout:
            self.dropout = nn.Dropout(dropout_rate)
        
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
        
    def forward(self, x, init_image=None):
        # TimeDistributed layers
        B, N, _ = x.shape
        
        x = self.time_distributed(x.view(B*N, -1))
        if self.args.model.use_dropout:
            x = self.dropout(x)
        x = x.view(B, N, -1)

        # if there is video, mix the feature which will be fed to LSTM layer
        
        if init_image is not None:
            # input: init_image.shape [H, W]
            if self.args.model.use_deform:
                y = self.img_enc(init_image.unsqueeze(0).unsqueeze(0)).view(1, x.shape[-1], -1).permute(0, 2, 1).repeat(B, 1, 1) # [1, -1, n_feats]
            elif self.args.model.use_prev_frame:
                y = self.img_enc(init_image.unsqueeze(1)).view(B, x.shape[-1], -1).permute(0,2,1)
            else:
                raise NotImplementedError
            
            y = torch.cat([x, y], dim=1)
            x = self.cat_feats(y)
            x = x.view(B, N, -1)
        
        # LSTM layers
        if self.args.model.use_lstm:
            x, _ = self.lstm_1(x)
            x, _ = self.lstm_2(x)

        # Transformer layers
        if self.args.model.use_transformer:
            x = x.permute(1, 0, 2)  # Transformer expects (sequence_length, batch_size, features)
            x = self.transformer(x)
            x = x.permute(1, 0, 2)  # Convert back to (batch_size, sequence_length, features)

        # pred deformation field here
        # use last layer
        x = self.dense0(x[:,-1:].reshape(B, -1))
        if self.args.model.use_dropout:
            x = self.dropout(x)
        x = self.dense1(x)
        x = torch.sigmoid(x)
        x = x.view(B, self.final_channel, self.n_width, self.n_height)

        if self.args.model.use_deform:
            pred_x = warp_image(init_image.unsqueeze(0).unsqueeze(0).repeat(B, 1, 1, 1), x).squeeze()
        else:
            pred_x = x.squeeze()
        
        return pred_x
