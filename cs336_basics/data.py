import numpy as np
import torch


def get_batch(dataset, batch_size, context_length, device):
    starts = torch.randint(0, len(dataset) - context_length, (batch_size,))
    x = np.stack([dataset[i : i + context_length] for i in starts])
    y = np.stack([dataset[i + 1 : i + context_length + 1] for i in starts])
    return torch.tensor(x, dtype=torch.long, device=device), torch.tensor(y, dtype=torch.long, device=device)
