import torch

print(f"pytorch version: {torch.__version__}")

if torch.backends.mps.is_available():
    print("MPS device found.")
    mps_device = torch.device("mps")
    x = torch.ones(1, device=mps_device)
    print (x)
else:
    print ("MPS device not found.")