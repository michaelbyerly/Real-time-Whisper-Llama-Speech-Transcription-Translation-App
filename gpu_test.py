# gpu_test.py

import torch

def check_cuda():
    print("CUDA Available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("Number of GPUs:", torch.cuda.device_count())
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("No CUDA-enabled GPUs detected.")

if __name__ == "__main__":
    check_cuda()
