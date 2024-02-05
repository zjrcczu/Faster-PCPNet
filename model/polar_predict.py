import torch
import torch.nn as nn

class predictor(nn.Module):
    def __init__(self):
        super(predictor, self).__init__()

        self.conv_past = nn.Conv1d(6, 128, 3, stride=1, padding=1)
        self.bn_past = nn.BatchNorm2d(1)

        self.encoder_past = nn.GRU(128, 128, 1, batch_first=True)
        self.decoder = nn.GRU(128, 128, 1, batch_first=False)
        self.FC_output = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.Linear(64, 6))

        self.relu = nn.ReLU()

        # weight initialization: kaiming
        # self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.conv_past.weight)
        nn.init.kaiming_normal_(self.encoder_past.weight_ih_l0)
        nn.init.kaiming_normal_(self.encoder_past.weight_hh_l0)
        nn.init.kaiming_normal_(self.decoder.weight_ih_l0)
        nn.init.kaiming_normal_(self.decoder.weight_hh_l0)
        nn.init.kaiming_normal_(self.FC_output.weight)

        nn.init.zeros_(self.conv_past.bias)
        nn.init.zeros_(self.encoder_past.bias_ih_l0)
        nn.init.zeros_(self.encoder_past.bias_hh_l0)
        nn.init.zeros_(self.decoder.bias_ih_l0)
        nn.init.zeros_(self.decoder.bias_hh_l0)
        nn.init.zeros_(self.FC_output.bias)

    def forward(self, past):

        dim_batch = past.size()[0]
        zero_padding = torch.zeros(1, dim_batch, 128).cuda()
        prediction = torch.Tensor().cuda()

        # temporal encoding for past
        past_embed = self.relu(self.conv_past(past))
        past_embed = self.bn_past(past_embed.unsqueeze(1)).squeeze(1)
        past_embed = torch.transpose(past_embed, 1, 2)


        # sequence encoding
        output_past, state_past = self.encoder_past(past_embed)
        state_fut = zero_padding.cuda()
        present = past[:, :, -1].unsqueeze(1)
        # state concatenation and decoding
        for i in range(30):
            output_decoder, state_fut = self.decoder(state_past, state_fut)
            displacement_next = self.FC_output(output_decoder)
            coords_next = present + displacement_next.squeeze(0).unsqueeze(1)
            prediction = torch.cat((prediction, coords_next), 1)
            present = coords_next
            state_past = zero_padding
        return prediction.permute(0, 2, 1)

    def mse_loss(self, prediction, future):
        return nn.MSELoss()(prediction, future)


class predictor_MLP(nn.Module):
    def __init__(self):
        super(predictor_MLP, self).__init__()
        self.MLP = nn.Sequential(
            nn.Linear(16*6, 16*24),
            nn.Linear(16*24, 16*24*2),
            nn.Linear(16*24*2, 16*24),
            nn.Linear(16*24, 30*6))


        # weight initialization: kaiming
        # self.reset_parameters()

    def forward(self, past):
        output = self.MLP(past.flatten(1)).reshape(past.shape[0], 30, 6)
        return output.permute(0, 2, 1)

    def mse_loss(self, prediction, future):
        return nn.MSELoss()(prediction, future)

