"""Tests for muscriptor/modules/transformer.py — CPU only, small tensors."""

import torch

from muscriptor.modules.streaming import compact_states, increment_steps, init_states
from muscriptor.modules.transformer import (
    create_sin_embedding,
    StreamingTransformer,
)


# ---------------------------------------------------------------------------
# Sinusoidal embeddings
# ---------------------------------------------------------------------------


def test_create_sin_embedding_shape():
    positions = torch.arange(10).float().view(1, 10, 1)  # [B, T, 1]
    emb = create_sin_embedding(positions, dim=16)
    assert emb.shape == (1, 10, 16)


def test_create_sin_embedding_dim_even():
    positions = torch.arange(5).float().view(1, 5, 1)
    emb = create_sin_embedding(positions, dim=8)
    assert emb.shape == (1, 5, 8)


def test_create_sin_embedding_different_positions():
    pos1 = torch.tensor([[[0.0]]])  # [1, 1, 1]
    pos2 = torch.tensor([[[1.0]]])
    e1 = create_sin_embedding(pos1, dim=8)
    e2 = create_sin_embedding(pos2, dim=8)
    assert not torch.allclose(e1, e2)


# ---------------------------------------------------------------------------
# StreamingTransformer forward
# ---------------------------------------------------------------------------


def _make_transformer(**kwargs):
    defaults = dict(d_model=32, num_heads=2, num_layers=2, dim_feedforward=64)
    defaults.update(kwargs)
    return StreamingTransformer(**defaults)


def test_streaming_transformer_output_shape():
    model = _make_transformer()
    model.eval()
    x = torch.randn(2, 10, 32)  # [B, T, D]
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 10, 32)


def test_streaming_transformer_streaming_mode():
    """Feed tokens one at a time with explicit state; check output shape."""
    model = _make_transformer()
    model.eval()
    x = torch.randn(1, 6, 32)

    with torch.no_grad():
        full_out = model(x)
    model_state = init_states(model, batch_size=1, sequence_length=x.shape[1])
    streaming_outs = []
    with torch.no_grad():
        for t in range(x.shape[1]):
            out_t = model(x[:, t : t + 1, :], model_state=model_state)
            streaming_outs.append(out_t)
            increment_steps(model, model_state, increment=1)
    streaming_out = torch.cat(streaming_outs, dim=1)

    assert streaming_out.shape == (1, 6, 32)
    assert torch.allclose(streaming_out, full_out, atol=1e-5, rtol=1e-5)


def test_streaming_state_cursors_do_not_live_on_the_accelerator():
    """Token decoding must not synchronize a device scalar in every layer."""
    model = _make_transformer()
    state = init_states(model, batch_size=2, sequence_length=8)

    cursors = [
        module_state["offset"]
        for module_state in state.values()
        if "offset" in module_state
    ]
    assert cursors
    assert all(isinstance(cursor, int) for cursor in cursors)

    increment_steps(model, state, increment=3)
    assert all(
        module_state["offset"] == 3
        for module_state in state.values()
        if "offset" in module_state
    )


def test_streaming_transformer_fresh_state():
    model = _make_transformer()
    model.eval()
    x = torch.randn(1, 3, 32)
    state = init_states(model, batch_size=1, sequence_length=x.shape[1])
    with torch.no_grad():
        model(x, model_state=state)
    # Allocating a new state starts from scratch.
    state = init_states(model, batch_size=1, sequence_length=x.shape[1])
    with torch.no_grad():
        out = model(x, model_state=state)
    assert out.shape == (1, 3, 32)


def test_compacted_streaming_state_matches_independently_decoded_rows():
    torch.manual_seed(0)
    model = _make_transformer()
    model.eval()
    inputs = torch.randn(4, 4, 32)
    keep = torch.tensor([1, 3])
    full_state = init_states(
        model, batch_size=4, sequence_length=4, initialize_cache=False
    )
    kept_state = init_states(
        model, batch_size=2, sequence_length=4, initialize_cache=False
    )

    with torch.no_grad():
        for timestep in range(3):
            model(
                inputs[:, timestep : timestep + 1],
                model_state=full_state,
            )
            model(
                inputs[keep, timestep : timestep + 1],
                model_state=kept_state,
            )
            increment_steps(model, full_state)
            increment_steps(model, kept_state)

        compact_states(full_state, keep)
        compacted_output = model(inputs[keep, 3:4], model_state=full_state)
        independent_output = model(inputs[keep, 3:4], model_state=kept_state)

    assert torch.allclose(compacted_output, independent_output, atol=1e-5, rtol=1e-5)
