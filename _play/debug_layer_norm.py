import subprocess, tempfile
from pathlib import Path
from safetensors2verilog import Gate, GateGraph, Signal, emit_module
from safetensors2verilog.blocks.layer_norm import layer_norm_block

ln, rsq = layer_norm_block(K=8, gamma_int=[16384]*8, beta_int=[0]*8)
parent = GateGraph(
    inputs=[Signal("clk"), Signal("rst"), Signal("start"), Signal("x_packed", width=64, signed=False)],
    outputs=[Signal("done", width=1), Signal("y_packed", width=64, signed=False)],
    gates=[
        Gate(name="done", kind="extern_wire", output_width=1),
        Gate(name="y_packed", kind="instance", inputs=["clk", "rst", "start", "x_packed"],
             attrs={"module_name": ln.top, "instance_name": "ln",
                    "input_ports": ["clk", "rst", "start", "x_packed"],
                    "output_port": "y_packed",
                    "extra_output_ports": [("done", "done")]},
             output_width=64, output_signed=False),
    ],
    top="ln_test", submodules=[ln, rsq],
)
text = emit_module(parent)
tb = r"""`timescale 1ns/1ps
module tb;
  reg clk = 0; always #5 clk = ~clk;
  reg rst = 1, start = 0;
  reg [63:0] x_packed;
  wire done;
  wire [63:0] y_packed;
  ln_test dut(.clk(clk), .rst(rst), .start(start), .x_packed(x_packed), .done(done), .y_packed(y_packed));
  integer cycles;
  initial begin
    rst = 1; #20 rst = 0;
    @(negedge clk);
      x_packed = 64'h050a030702fefcf8;
      start <= 1;
    @(negedge clk); start <= 0;
    cycles = 0;
    while (!done) begin
      @(posedge clk);
      cycles = cycles + 1;
      if (cycles > 60) begin $display("TIMEOUT cyc=%0d", cycles); $finish; end
      $display("[cyc=%0d] state=%h sum_x=%h sum_sq=%h mean=%h rsqrt=%h y[0]=%h",
               cycles, dut.ln.state, dut.ln.sum_x, dut.ln.sum_sq,
               dut.ln.mean_latched, dut.ln.rsqrt_val, dut.ln.y_buf[0]);
    end
    $display("DONE cyc=%0d y_packed=%h", cycles, y_packed);
    $finish;
  end
endmodule
"""
with tempfile.TemporaryDirectory() as td:
    td = Path(td)
    (td / "dut.v").write_text(text)
    (td / "tb.v").write_text(tb)
    vvp = td / "out.vvp"
    p = subprocess.run(["iverilog", "-g2012", "-o", str(vvp), str(td/"dut.v"), str(td/"tb.v")],
                       capture_output=True, text=True)
    if p.returncode != 0:
        print("compile fail:", p.stderr); raise SystemExit(1)
    p = subprocess.run(["vvp", str(vvp)], capture_output=True, text=True, timeout=30)
    print(p.stdout)
