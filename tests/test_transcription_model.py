"""Tests for TranscriptionModel._generate_token_stream.

These check the streaming contract of the token stream without a real model:
a fake `generate()` yields one row (`[batch]`) per timestep and records how
many timesteps have been pulled, so we can assert that a chunk's events come
out *as soon as that chunk finishes* — before the rest of the batch is even
generated.
"""

from types import SimpleNamespace

import pytest
import torch
from muscriptor.events import ChunkBoundary, ProgressEvent
from muscriptor.transcription_model import TranscriptionModel, _split_audio_chunks
from muscriptor.utils.beats import BeatDetectionError

EOS = 99


def _run(
    batches,
    *,
    batch_size,
    seek_times,
    no_eos_is_ok=False,
    optimized_decoding=False,
    audio_end_time=None,
    chunk_boundaries=None,
):
    """Drive _generate_token_stream with a fake model.

    ``batches`` is one list of rows per expected ``generate()`` call; each row
    is the per-chunk token for one timestep. Returns ``(stream, pulled)`` where
    ``pulled`` grows by one entry every time the fake yields a timestep, so its
    length is how far generation has progressed.
    """
    pulled: list[list[int]] = []
    calls = iter(batches)

    def generate(**kwargs):
        for row in next(calls):
            pulled.append(row)
            yield torch.tensor(row)

    fake = SimpleNamespace(
        _model=SimpleNamespace(generate=generate),
        _tokenizer=SimpleNamespace(eos_id=EOS),
    )
    if chunk_boundaries is None:
        chunk_boundaries = [
            ChunkBoundary(
                seek_time,
                (
                    seek_times[index + 1]
                    if index + 1 < len(seek_times)
                    else audio_end_time
                ),
            )
            for index, seek_time in enumerate(seek_times)
        ]
    conditions = [object()] * len(chunk_boundaries)
    stream = TranscriptionModel._generate_token_stream(
        fake,
        conditions,
        chunk_boundaries,
        batch_size,
        max_gen_len=64,
        use_sampling=False,
        temperature=1.0,
        cfg_coef=2.0,
        no_eos_is_ok=no_eos_is_ok,
        # The fake tokenizer has no vocab; prelude forcing has its own tests
        # (test_prelude_forcing.py).
        prelude_forcing=False,
        optimized_decoding=optimized_decoding,
    )
    return stream, pulled


# ---------------------------------------------------------------------------
# Audio chunk construction
# ---------------------------------------------------------------------------


def test_audio_chunks_have_half_second_context_around_four_second_cores():
    sample_rate = 16_000
    wav = torch.arange(9 * sample_rate, dtype=torch.float32).unsqueeze(0)

    chunks, boundaries = _split_audio_chunks(wav)

    assert boundaries == [
        ChunkBoundary(-0.5, 4.0, 0.0),
        ChunkBoundary(3.5, 8.0, 4.0),
        ChunkBoundary(7.5, 9.0, 8.0),
    ]
    assert all(chunk.shape == (1, 5 * sample_rate) for chunk in chunks)
    assert torch.count_nonzero(chunks[0][:, : sample_rate // 2]) == 0
    assert torch.equal(
        chunks[0][:, sample_rate // 2 :], wav[:, : int(4.5 * sample_rate)]
    )
    assert torch.equal(
        chunks[1], wav[:, int(3.5 * sample_rate) : int(8.5 * sample_rate)]
    )
    assert torch.equal(
        chunks[2][:, : int(1.5 * sample_rate)], wav[:, int(7.5 * sample_rate) :]
    )
    assert torch.count_nonzero(chunks[2][:, int(1.5 * sample_rate) :]) == 0


# ---------------------------------------------------------------------------
# Emitted as soon as possible
# ---------------------------------------------------------------------------


def test_first_chunk_streams_before_the_batch_finishes():
    # batch of 2 chunks: chunk 0 ends at row 2, chunk 1 only at row 4.
    rows = [[10, 20], [11, 21], [EOS, 22], [12, 23], [13, EOS]]
    stream, pulled = _run([rows], batch_size=2, seek_times=[0.0, 5.0])
    it = iter(stream)

    assert next(it) == ChunkBoundary(0.0, 5.0)
    assert len(pulled) == 0  # the boundary is emitted before any generation
    assert next(it) == 10
    assert len(pulled) == 1  # first token after a single timestep
    assert next(it) == 11
    # Chunk 0 is fully streamed having generated only its own timesteps —
    # chunk 1 (which finishes at row 4) has not been generated to completion.
    assert len(pulled) == 2


def test_single_chunk_streams_token_by_token():
    rows = [[10], [11], [12], [EOS]]
    stream, pulled = _run([rows], batch_size=1, seek_times=[0.0])
    it = iter(stream)

    assert next(it) == ChunkBoundary(0.0, None)
    assert len(pulled) == 0
    for expected, count in [(10, 1), (11, 2), (12, 3)]:
        assert next(it) == expected
        assert len(pulled) == count


def test_final_chunk_uses_source_audio_end_as_its_exclusive_cutoff():
    rows = [[10, 20], [EOS, 21], [30, EOS]]
    stream, _ = _run(
        [rows],
        batch_size=2,
        seek_times=[0.0, 5.0],
        audio_end_time=7.35,
    )

    boundaries = [item for item in stream if isinstance(item, ChunkBoundary)]

    assert boundaries == [ChunkBoundary(0.0, 5.0), ChunkBoundary(5.0, 7.35)]


def test_optimized_decoder_buffers_generation_but_preserves_output_order():
    rows = [[10, 20], [11, 21], [EOS, 22], [12, 23], [13, EOS]]
    stream, pulled = _run(
        [rows],
        batch_size=2,
        seek_times=[0.0, 5.0],
        optimized_decoding=True,
    )
    iterator = iter(stream)

    assert next(iterator) == ChunkBoundary(0.0, 5.0)
    assert pulled == []
    assert next(iterator) == 10
    assert pulled == rows
    assert list(iterator) == [
        11,
        ChunkBoundary(5.0, None),
        20,
        21,
        22,
        23,
        ProgressEvent(completed=2, total=2),
    ]


# ---------------------------------------------------------------------------
# Ordering and buffering
# ---------------------------------------------------------------------------


def test_full_stream_order_for_a_batch():
    rows = [[10, 20], [11, 21], [EOS, 22], [12, 23], [13, EOS]]
    stream, _ = _run([rows], batch_size=2, seek_times=[0.0, 5.0])
    assert list(stream) == [
        ChunkBoundary(0.0, 5.0),
        10,
        11,
        ChunkBoundary(5.0, None),
        20,
        21,
        22,
        23,
        # End of the (only) batch: both chunks done.
        ProgressEvent(completed=2, total=2),
    ]


def test_later_chunk_finishing_first_is_buffered_until_its_turn():
    # chunk 1 hits EOS (row 1) before chunk 0 (row 3); its tokens must wait.
    rows = [[10, 20], [11, EOS], [12, 88], [EOS, 88]]
    stream, _ = _run([rows], batch_size=2, seek_times=[0.0, 5.0])
    assert list(stream) == [
        ChunkBoundary(0.0, 5.0),
        10,
        11,
        12,
        ChunkBoundary(5.0, None),
        20,
        ProgressEvent(completed=2, total=2),
    ]


def test_chunks_across_multiple_batches_stay_in_order():
    # batch_size=1 → one generate() call per chunk.
    batches = [[[10], [11], [EOS]], [[20], [EOS]]]
    stream, _ = _run(batches, batch_size=1, seek_times=[0.0, 5.0])
    assert list(stream) == [
        ChunkBoundary(0.0, 5.0),
        10,
        11,
        # batch_size=1 => a completion anchor trails each chunk.
        ProgressEvent(completed=1, total=2),
        ChunkBoundary(5.0, None),
        20,
        ProgressEvent(completed=2, total=2),
    ]


# ---------------------------------------------------------------------------
# Missing EOS
# ---------------------------------------------------------------------------


def test_missing_eos_raises_by_default():
    rows = [[10, 20], [11, 21]]  # neither chunk emits EOS
    stream, _ = _run([rows], batch_size=2, seek_times=[0.0, 5.0])
    with pytest.raises(RuntimeError, match="did not emit EOS"):
        list(stream)


def test_missing_eos_warns_and_still_emits_when_allowed():
    rows = [[10, 20], [11, 21]]
    stream, _ = _run([rows], batch_size=2, seek_times=[0.0, 5.0], no_eos_is_ok=True)
    with pytest.warns(RuntimeWarning, match="did not emit EOS"):
        events = list(stream)
    assert events == [
        ChunkBoundary(0.0, 5.0),
        10,
        11,
        ChunkBoundary(5.0, None),
        20,
        21,
        ProgressEvent(completed=2, total=2),
    ]


# ---------------------------------------------------------------------------
# Tempo detection modes
# ---------------------------------------------------------------------------


class _FakeAudio:
    """Just enough of a model for detect_beat_grid_for's mode dispatch."""

    _load_wav = staticmethod(lambda tensor, sr: tensor)
    detect_beat_grid_for = TranscriptionModel.detect_beat_grid_for


def test_detect_tempo_modes(monkeypatch):
    def boom(*args, **kwargs):
        raise BeatDetectionError("no fixed tempo")

    monkeypatch.setattr("muscriptor.transcription_model.detect_grid", boom)
    model = _FakeAudio()
    # false: never even calls the detector.
    assert model.detect_beat_grid_for((None, None), False) is None
    # best-effort: swallows the failure, no grid written.
    assert model.detect_beat_grid_for((None, None), "best-effort") is None
    # true: the caller wanted to know.
    with pytest.raises(BeatDetectionError):
        model.detect_beat_grid_for((None, None), True)
