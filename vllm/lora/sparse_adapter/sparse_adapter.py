import os
import torch
import safetensors.torch

from vllm.utils.platform_utils import is_pin_memory_available


class SparseAdapter: 

    def __init__(
        self,
        sparse_adapter_id: int,
        deltas: dict[str, torch.Tensor],
    ):
        assert sparse_adapter_id > 0

        for name, delta in deltas.items():
            if delta.ndim != 2:
                raise ValueError(
                    "vLLM only supports deltas for linear layer weights, "
                    "which must be 2D (matrix) tensors.\n"
                    f"Condition not satisfied by delta for model param: '{name}'")

        self.id = sparse_adapter_id
        self.deltas = deltas

    def clone(self, sparse_adapter_id: int) -> "SparseAdapter":
        return self.__class__(
            sparse_adapter_id,
            deltas=self.deltas.copy(), # shares underlying tensors
        )

    @classmethod
    def from_local_checkpoint(
        cls,
        sparse_adapter_dpath: str,
        sparse_adapter_id: int | None = None,
        device: str | torch.device = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        density_threshold: float = 0.1, # convert sparse delta to dense if density > threshold  
    ) -> "SparseAdapter":

        if sparse_adapter_id is None:
            sparse_adapter_id = _get_sparse_adapter_id()

        ckpt = os.path.join(sparse_adapter_dpath, 
                            "sparse_adapter.vllm.safetensors")
        
        if not os.path.isfile(ckpt):
            raise FileNotFoundError(f"checkpoint not found: {ckpt}")
        
        tensors = safetensors.torch.load_file(ckpt, device="cpu")

        pin_memory = str(device) == "cpu" and is_pin_memory_available()
        
        sparse_deltas = {}

        param_names = set('.'.join(name.split('.')[:-1]) for name in tensors)

        for param_name in param_names:

            values = tensors[f"{param_name}.values"]
            crow_indices = tensors[f"{param_name}.crow_indices"]
            col_indices = tensors[f"{param_name}.col_indices"] 
            size = tensors[f"{param_name}.size"].tolist() 

            sparse_delta = torch.sparse_csr_tensor(crow_indices, col_indices, values, size=size)

            nnz = len(sparse_delta.values())
            if nnz > density_threshold * sparse_delta.numel(): # numel as product of shape dimensions
                sparse_delta = sparse_delta.to_dense()
            
            sparse_delta = sparse_delta.to(device=device, dtype=dtype)
            if pin_memory:
                sparse_delta = sparse_delta.pin_memory()

            # sparse_deltas[param_name] = sparse_delta
            module_name = param_name.removesuffix('.weight')
            sparse_deltas[module_name] = sparse_delta

        print(f"Loaded sparse adapter from {sparse_adapter_dpath}")

        return cls(sparse_adapter_id, deltas=sparse_deltas)



# Global counter for automatic (unique) ID assignment
_GLOBAL_SPARSE_ADAPTER_ID = 0

def _get_sparse_adapter_id():
    global _GLOBAL_SPARSE_ADAPTER_ID
    _GLOBAL_SPARSE_ADAPTER_ID += 1
    return _GLOBAL_SPARSE_ADAPTER_ID

