import torch.nn as nn


class NullEmbedder(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, data):
        return data


embedder_names_to_kwargs = {
    'NullEmbedder': set(),
}