"""4 -> 3 -> 2 ternary MLP with biases."""
from pathlib import Path

import torch
from safetensors.torch import save_file

OUT = Path(__file__).parent / "mlp.safetensors"

# layer 0: 4 inputs -> 3 hidden
W0 = [[1, -1,  0,  1],
      [0,  1,  1, -1],
      [1,  1, -1,  0]]
B0 = [1, -1, 0]

# layer 1: 3 hidden -> 2 outputs
W1 = [[1, -1,  1],
      [-1, 1,  1]]
B1 = [-2, 3]

save_file(
    {
        "layers.0.weight": torch.tensor(W0, dtype=torch.int8),
        "layers.0.bias":   torch.tensor(B0, dtype=torch.int32),
        "layers.1.weight": torch.tensor(W1, dtype=torch.int8),
        "layers.1.bias":   torch.tensor(B1, dtype=torch.int32),
    },
    str(OUT),
)
print(f"wrote {OUT}")
