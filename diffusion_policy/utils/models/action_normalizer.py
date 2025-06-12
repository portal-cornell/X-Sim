import torch


class ActionNormalizer:
    def __init__(self, action_min: torch.Tensor, action_max: torch.Tensor):
        # Ensure the tensors are in float32
        action_min = action_min.to(torch.float32)
        action_max = action_max.to(torch.float32)

        assert action_min.dim() == 1 and action_min.size() == action_max.size()
        assert action_min.dtype == torch.float32
        assert action_max.dtype == torch.float32

        self.action_dim = action_min.size(0)
        self.min = action_min
        self.max = action_max

        # Calculate scale and offset with torch.where
        epsilon = 1e-8
        self.scale = torch.where(
            self.max != self.min, 2 / (self.max - self.min + epsilon), self.max
        )
        self.offset = torch.where(
            self.max != self.min,
            (self.max + self.min) / (self.min - self.max + epsilon),
            torch.zeros_like(self.max),
        )

        self.scale = self.scale.unsqueeze(0)
        self.offset = self.offset.unsqueeze(0)

    def to(self, device):
        self.scale = self.scale.to(device)
        self.offset = self.offset.to(device)

    def normalize(self, value: torch.Tensor):
        shape = value.size()
        # Ensure the value is in float32
        value = value.to(torch.float32)
        value = value.reshape(-1, self.action_dim)
        normed_value = value * self.scale + self.offset
        normed_value = normed_value.reshape(shape)
        return normed_value

    def denormalize(self, normed_value: torch.Tensor):
        shape = normed_value.size()
        # Ensure the normed_value is in float32
        normed_value = normed_value.to(torch.float32)
        normed_value = normed_value.reshape(-1, self.action_dim)
        value = (normed_value - self.offset) / self.scale
        value = value.reshape(shape)
        return value
