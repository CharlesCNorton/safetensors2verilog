from pathlib import Path
from safetensors2verilog import emit_module, registry

fe = registry.get('bitnet_linear')()
g = fe.parse(
    Path(r'D:\safetensors2verilog\_play\mlp.safetensors'),
    top='mlp', activation_bits=4, output_clamp='-8,7', pipeline=True,
)
v_lines = emit_module(g, target='verilog').splitlines()
sv_lines = emit_module(g, target='sv').splitlines()
diffs = [(i, a, b) for i, (a, b) in enumerate(zip(v_lines, sv_lines)) if a != b]
print(f'{len(diffs)} differing lines of {len(v_lines)} total')
for i, a, b in diffs[:8]:
    print(f'L{i}:')
    print(f'  V : {a}')
    print(f'  SV: {b}')
