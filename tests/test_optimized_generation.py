"""Tests for the opt-in buffered and EOS-compacting decoder."""

from types import MethodType

import torch

from muscriptor.models.lm import LMModel
from muscriptor.modules.conditioners import ConditioningProvider


def test_optimized_generation_preserves_rows_and_reports_saved_work():
    model = LMModel(
        condition_provider=ConditioningProvider(
            conditioners={}, device=torch.device("cpu")
        ),
        card=16,
        dim=16,
        num_heads=2,
        hidden_scale=2,
        num_layers=1,
        max_period=10000,
        device=torch.device("cpu"),
    )
    model.eval()
    eos = 3
    call = 0

    def sample_next_token(self, sequence, *args, **kwargs):
        nonlocal call
        call += 1
        rows = {
            (1, 4): [eos, 1, 1, 1],
            (2, 4): [8, eos, 1, 1],
            (3, 2): [eos, 1],
            (4, 2): [8, eos],
        }
        return torch.tensor(rows[(call, sequence.shape[0])], device=sequence.device)

    setattr(model, "_sample_next_token", MethodType(sample_next_token, model))
    stats: dict[str, int] = {}
    steps = list(
        model.generate(
            num_samples=4,
            max_gen_len=8,
            use_sampling=False,
            early_stop_on_token=eos,
            optimized_decoding=True,
            eos_check_interval=1,
            generation_stats=stats,
        )
    )
    rows = torch.stack(steps).T.tolist()
    rows_through_eos = [row[: row.index(eos) + 1] for row in rows]

    assert rows_through_eos == [
        [eos],
        [1, eos],
        [1, 1, eos],
        [1, 1, 1, eos],
    ]
    assert stats == {
        "scheduled_row_steps": 12,
        "compactions": 1,
        "saved_row_steps": 4,
    }
