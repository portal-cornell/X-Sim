import os
import subprocess
import signal
from tabulate import tabulate
import random
import numpy as np
import torch
from termcolor import cprint as _cprint


def cprint(msg, color="magenta"):
    _cprint(msg, color)


def kill_process_on_port(port):
    try:
        # Find process using the port
        result = subprocess.check_output(["lsof", "-i", f":{port}"])
        lines = result.splitlines()
        for line in lines[1:]:  # Skip the header line
            columns = line.split()
            pid = int(columns[1])
            print("[*] Found existing server process)")
            print(f"[*] Killing process with PID: {pid} on port {port}")
            os.kill(pid, signal.SIGKILL)
    except subprocess.CalledProcessError:
        print(f"[*] Starting server")
    except Exception as e:
        print(f"[*] Server error occurred: {e}")


def count_parameters(model):
    rows = []
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        rows.append([name, params])
        total_params += params

    for row in rows:
        row.append(row[-1] / total_params * 100)

    rows.append(["Total", total_params, 100])
    table = tabulate(
        rows, headers=["Module", "#Params", "%"], intfmt=",d", floatfmt=".2f", tablefmt="orgtbl"
    )
    print(table)


def get_all_files(root, file_extension, contain=None) -> list[str]:
    files = []
    for folder, _, fs in os.walk(root):
        for f in fs:
            if file_extension is not None:
                if f.endswith(file_extension):
                    if contain is None or contain in os.path.join(folder, f):
                        files.append(os.path.join(folder, f))
            else:
                if contain in f:
                    files.append(os.path.join(folder, f))
    return files


def set_all_seeds(rand_seed):
    random.seed(rand_seed)
    np.random.seed(rand_seed + 1)
    torch.manual_seed(rand_seed + 2)
    # seed_all for all gpus
    torch.cuda.manual_seed_all(rand_seed + 3)


def wrap_ruler(text: str, max_len=40):
    text_len = len(text)
    if text_len > max_len:
        return text_len

    left_len = (max_len - text_len) // 2
    right_len = max_len - text_len - left_len
    return ("=" * left_len) + text + ("=" * right_len)


class eval_mode:
    def __init__(self, *models):
        self.models = models

    def __enter__(self):
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(False)

    def __exit__(self, *args):
        for model, state in zip(self.models, self.prev_states):
            model.train(state)
        return False
