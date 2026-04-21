import torch


class SparseAdapterWrapper:  # for Multi-Tenant Serving

    def __init__(self):
        self.token_to_adapter = None
        self.prompt_to_adapter = None # for lm_head adaption (if targeted)
        self.requested_adapters: set[int] = set()

    @torch.compiler.disable
    def add_deltas(
        self,
        y: torch.Tensor,
        x: torch.Tensor,
        deltas: dict[int, torch.Tensor],
    ) -> torch.Tensor | None:
        """ y += x @ S

        Args:
            y: Output tensor [num_tokens, out_dim]
            x: Input tensor [num_tokens, in_dim]
            deltas: dict of (sparse) deltas per adapter loaded in model layer (active)

        Returns:
            Updated output tensor or None if operation should be skipped
        """
        if not deltas or not self.requested_adapters:
            return None

        for adapter in self.requested_adapters:

            token_mask = self.token_to_adapter == adapter
            if not token_mask.any():
                continue
            x_subset = x[token_mask] # shape: [num_tokens_with_adapter, in_dim]

            delta = deltas.get(adapter)  # shape: [out_dim, in_dim]
            if delta is None:
                continue

            y_subset = x_subset @ delta.t().to(dtype=x.dtype) 
                    
            y[token_mask] += y_subset # shape:[num_tokens_with_adapter, out_dim]

        return y # output shape:[num_tokens, out_dim]

    @torch.compiler.disable
    def add_deltas_to_logits(
            self,
            logits: torch.Tensor,
            hidden_states: torch.Tensor,
            deltas: dict[int, torch.Tensor],
        ) -> torch.Tensor | None:
        """ logits += hidden_states @ S

        Operates on sample positions only

        Args:
            logits: [num_sampled_tokens, vocab_size]
            hidden_states: [num_sampled_tokens, hidden_state_dim]
            deltas: (sparse) deltas per adapter loaded in the lm_head layer (active)
                    {a_id: delta [vocab_size, hidden_state_dim]}     

        Returns:
            Updated logits or None if operation should be skipped
        """

        if not deltas or not self.requested_adapters:
            return None

        logits_orig = logits
        logits = logits.view(-1, logits.shape[-1])
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])

        for adapter in self.requested_adapters:

            delta = deltas.get(adapter)
            if delta is None:
                continue

            prompt_mask = self.prompt_to_adapter == adapter
            if not prompt_mask.any():
                continue
            hs = hidden_states[prompt_mask]

            logits[prompt_mask] += hs @ delta.t().to(dtype=hs.dtype)

        return logits.view_as(logits_orig)

