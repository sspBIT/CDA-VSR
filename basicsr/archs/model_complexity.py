import torch
from thop import profile
from cdavsr_arch import CDAVSR

if __name__ == "__main__":
    device = "cuda"
    x  = torch.randn(1, 25, 3, 180, 320)
    mv = torch.randn(1, 25, 2, 180, 320)
    res = torch.randn(1, 25, 1, 180, 320)
    model = CDAVSR()

    macs, params = profile(model, inputs=(x,mv,res,))
    flops = 2 * macs

    print("=" * 50)
    print(f"Params: {params/1e6:.2f} M")
    print(f"MACs:   {macs/1e9/25:.2f} G")
    print(f"FLOPs:  {flops/1e9/25:.2f} G  (≈ 2×MACs)")
    print("=" * 50)

