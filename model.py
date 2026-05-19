import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, thermal_output_size=6):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.thermal_output_size = thermal_output_size  # 热工特征输出维度

        # LSTM层（处理合并特征）
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

        # 输出层：BTP预测（1维）和热工特征预测（thermal_output_size维）
        self.fc_btp = nn.Linear(hidden_size, 1)
        self.fc_thermal = nn.Linear(hidden_size, thermal_output_size)

    def forward(self, x):
        # 初始化隐藏状态和细胞状态
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)

        # LSTM前向传播
        out, (_, _) = self.lstm(x, (h0, c0))  # out形状: (batch_size, seq_len, hidden_size)

        # 取最后一个时间步的输出
        last_out = out[:, -1, :]

        # 输出预测
        btp_pred = self.fc_btp(last_out)  # BTP预测
        thermal_pred = self.fc_thermal(last_out)  # 热工特征预测

        return btp_pred, thermal_pred
