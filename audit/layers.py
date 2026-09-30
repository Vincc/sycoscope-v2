"""Each audited repo's layer index mapped to this repo's block index L (read from hidden_states[L + 1])."""


def from_block_output(layer: int) -> int:
    """CAA (decoder block output) and Pandey (TransformerLens blocks.L.hook_resid_post): same block index."""
    return layer


def from_hidden_states(index: int) -> int:
    """Persona Vectors indexes hidden_states[l] (0 = embeddings); block l - 1 here."""
    if index < 1:
        raise ValueError("hidden_states[0] is the embedding output, not a block")
    return index - 1


def from_vennemeyer(layer_idx: int, n_blocks: int) -> int:
    """diffmean_analysis --all_layers loops 1..n_blocks; model_utils._hidden_to_block_index returns values below
    n_blocks unchanged and maps n_blocks to the last block, so their L1..L31 are blocks 1..31 and L32 repeats block 31.
    """
    if layer_idx == 0:
        raise ValueError("Vennemeyer layer 0 (embeddings) is never computed")
    if layer_idx == n_blocks:
        return n_blocks - 1
    if 0 < layer_idx < n_blocks:
        return layer_idx
    raise ValueError(f"layer {layer_idx} outside 1..{n_blocks}")


def persona_paper_layer(layer: int) -> int:
    """Persona Vectors paper counts layers from 1 ('layer 16' is the 16th layer's output = hidden_states[16])."""
    return from_hidden_states(layer)
