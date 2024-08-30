try:
    import torch
    print("PyTorch is installed. Version:", torch.__version__)
except ImportError:
    print("PyTorch is not installed.")
