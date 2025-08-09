import torch
from torch import nn


class SlotSlotContrastiveLoss(nn.Module):
    def __init__(
        self,
        temperature: float = 0.1,
        batch_contrast: bool = True,
    ):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss()
        self.temperature = temperature
        self.batch_contrast = batch_contrast

    def forward(self, slots):
        slots = nn.functional.normalize(slots, p=2.0, dim=-1)
        if self.batch_contrast:
            slots = slots.split(1)  # [1xTxKxD]
            slots = torch.cat(slots, dim=-2)  # 1xTxK*BxD
        s1 = slots[:, :-1, :, :]
        s2 = slots[:, 1:, :, :]
        ss = torch.matmul(s1, s2.transpose(-2, -1)) / self.temperature
        B, T, S, D = ss.shape
        ss = ss.reshape(B * T, S, S)
        target = torch.eye(S).expand(B * T, S, S).to(ss.device)
        loss = self.criterion(ss, target)
        return loss
