from dataclasses import dataclass

import msgspec


@dataclass
class SparseAdapterMapping:
    token_to_adapter: tuple[int, ...]
    prompt_to_adapter: tuple[int, ...]  # needed if adapting lm_head (LogitsProcessorWithLoRA)
    requests: set["SparseAdapterRequest"]


class SparseAdapterRequest(
    msgspec.Struct,
    omit_defaults=True,
    array_like=True,
):  # type: ignore[call-arg]
    
    sparse_adapter_name: str
    sparse_adapter_id: int
    sparse_adapter_path: str = ""
    base_model_name: str | None = msgspec.field(default=None)
    tensorizer_config_dict: dict | None = None
    load_inplace: bool = False

    def __post_init__(self):
        if self.sparse_adapter_id < 1:
            raise ValueError(f"id must be > 0, got {self.sparse_adapter_id}")

        assert self.sparse_adapter_path, "sparse_adapter_path cannot be empty"

    @property
    def adapter_id(self):
        return self.sparse_adapter_id

    @property
    def name(self):
        return self.sparse_adapter_name

    @property
    def path(self):
        return self.sparse_adapter_path

    def __eq__(self, value: object) -> bool:
        return (
            isinstance(value, self.__class__)
            and self.sparse_adapter_name == value.sparse_adapter_name
        )

    def __hash__(self) -> int:
        return hash(self.sparse_adapter_name)
