"""int8_linear: a 3-input dot product with weights {3, -7, 12, bias=10}."""
from pathlib import Path

import torch
from safetensors.torch import save_file

from safetensors2verilog import emit_module, registry

OUT = Path(__file__).parent / "int8_demo.safetensors"
save_file(
    {
        "layers.0.weight": torch.tensor([[3, -7, 12]], dtype=torch.int8),
        "layers.0.bias":   torch.tensor([10], dtype=torch.int32),
    },
    str(OUT),
)

fe = registry.get('int8_linear')()
g = fe.parse(OUT, top='dot', activation_bits=8, weight_bits=8)

print(f'gates: {len(g.gates)} ({[(x.kind, x.name) for x in g.gates]})')
print('outputs:', [(s.name, s.width, s.signed) for s in g.outputs])

v = emit_module(g)
out_v = Path(__file__).parent / "int8_demo.v"
out_v.write_text(v, encoding="utf-8")
print()
print(v)
