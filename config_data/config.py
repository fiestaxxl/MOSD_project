from dataclasses import dataclass
from torch import cuda, device




@dataclass
class Args:
    gpu: device
    pred_len: int
    param_len: int
    patch_size: int
    memory: int
    emb_size: int
    image_size: tuple




def load_config(path: str | None = None) -> Args:

    return Args(image_size = (60,90),
                pred_len = 12,
                param_len = 10, #TODO
                patch_size = 5,
                memory = 16,
                emb_size = 16*5**2,
                #emb_size = 5**2,
                gpu = device('cuda:0' if cuda.is_available() else 'cpu')
                )