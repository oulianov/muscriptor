"""Public streaming events and per-chunk event builder.

`TranscriptionModel.transcribe` is a generator that yields these dataclasses
one at a time. Every :class:`NoteStartEvent` is guaranteed to be followed by
exactly one matching :class:`NoteEndEvent` (same `index`) later in the stream.
"""

from collections.abc import Callable, Iterator
from dataclasses import dataclass

from muscriptor.tokenizer.notes import (
    MINIMUM_NOTE_DURATION_SEC,
    Event,
)

_DRUM_INSTRUMENT = "drums"


@dataclass
class NoteStartEvent:
    pitch: int
    start_time: float
    index: int
    instrument: str


@dataclass
class NoteEndEvent:
    end_time: float
    start_event: NoteStartEvent

    @property
    def start_event_index(self) -> int:
        return self.start_event.index


@dataclass
class ProgressEvent:
    """A coarse transcription-progress signal, woven into the event stream.

    Marks that ``completed`` of ``total`` fixed-size audio chunks have been
    transcribed (``completed == 0`` is emitted once up front so consumers learn
    ``total`` and get a timing baseline; ``completed == total`` marks the end).
    These are deliberately coarse anchors — the frontend smooths between them
    and derives an ETA, since wall-clock time per chunk is only observable
    there. Advisory only: consumers that build notes/MIDI ignore them.
    """

    completed: int
    total: int


@dataclass
class ChunkBoundary:
    """Marks the start of a new model-output chunk in the token stream.

    ``seek_time`` is the start of the audio context passed to the model.
    ``usable_start_time`` and ``next_seek_time`` delimit the non-overlapping
    core whose events belong in the reconstructed MIDI. When
    ``usable_start_time`` is omitted it defaults to ``seek_time`` for backwards
    compatibility with non-overlapping callers.
    """

    seek_time: float
    next_seek_time: float | None
    usable_start_time: float | None = None


@dataclass
class _StartNote:
    """A note opens: (program, pitch) starts sounding at `time`."""

    program: int
    pitch: int
    time: float


@dataclass
class _EndNote:
    """An open note closes: (program, pitch) stops sounding at `time`."""

    program: int
    pitch: int
    time: float


@dataclass
class _DrumHit:
    """An instantaneous drum hit at `time`; never enters the open set."""

    pitch: int
    time: float


_NoteAction = _StartNote | _EndNote | _DrumHit


class OpenNoteTracker:
    """The chunk-decoding state machine for the model's token stream.

    :meth:`feed` consumes the interleaved :class:`ChunkBoundary` markers and
    token indices and returns the note actions they imply; :meth:`finish`
    flushes the end-of-stream closes. All decode rules live here — the tie
    prologue (open notes absent from the tie set close at the boundary),
    malformed chunks (a shift before the ``tie`` token closes everything and
    drops the rest), the next_seek_time window, and retriggers.

    Two consumers share it: :func:`decode_model_tokens` turns the actions into
    indexed NoteStart/NoteEnd events, and the prelude-forcing path
    (``TranscriptionModel._generate_token_stream``) ignores the actions and
    reads :meth:`next_chunk_open_keys` before chunk boundaries — the
    ``(program, pitch)`` pairs active where the next model context begins (see
    ``MT3Tokenizer.tie_section_token_ids``). One state machine serving both
    keeps decoding and forcing consistent by construction.
    """

    def __init__(self, vocab: list[Event], frame_rate: int = 100):
        self._vocab = vocab
        self._frame_rate = frame_rate
        # (program, pitch) -> onset time. Insertion-ordered: end-of-stream
        # closes replay in onset order.
        self._open: dict[tuple[int, int], float] = {}
        # Per-chunk state, reset at every ChunkBoundary.
        self._seek_time = 0.0
        self._usable_start_time = 0.0
        self._next_seek_time: float | None = None
        self._start_tick = 0
        self._tick_state = 0
        self._program: int | None = None
        self._velocity: int | None = None
        self._in_prologue = True
        self._skip_rest = False
        self._tie_set: set[tuple[int, int]] = set()
        self._context_open: set[tuple[int, int]] = set()
        self._usable_range_started = False
        self._chunk_started = False
        self._continuation_time: float | None = None
        self._continuation_keys: list[tuple[int, int]] | None = None

    def feed(self, item: "int | ChunkBoundary") -> list[_NoteAction]:
        if isinstance(item, ChunkBoundary):
            actions: list[_NoteAction] = []
            if self._chunk_started:
                if self._in_prologue:
                    # A malformed chunk has no usable tie state.
                    self._context_open.clear()
                actions.extend(self._enter_usable_range())

            self._seek_time = item.seek_time
            self._usable_start_time = (
                item.usable_start_time
                if item.usable_start_time is not None
                else item.seek_time
            )
            self._next_seek_time = item.next_seek_time
            self._start_tick = round(item.seek_time * self._frame_rate)
            self._tick_state = self._start_tick
            self._program = None
            self._velocity = None
            self._in_prologue = True
            self._skip_rest = False
            self._tie_set = set()
            self._context_open = set()
            self._usable_range_started = False
            self._chunk_started = True
            left_context = self._usable_start_time - self._seek_time
            self._continuation_time = (
                item.next_seek_time - left_context
                if item.next_seek_time is not None
                else None
            )
            self._continuation_keys = None
            return actions

        event = self._vocab[item]
        etype = event.type

        if self._in_prologue:
            if etype == "tie":
                self._in_prologue = False
                self._velocity = None
                self._context_open = set(self._tie_set)
                if self._usable_start_time <= self._seek_time:
                    return self._enter_usable_range()
            elif etype == "shift":
                # No tie token: close the assembled state at the usable
                # boundary and ignore the malformed remainder of this chunk.
                self._in_prologue = False
                self._skip_rest = True
                self._context_open.clear()
                return self._enter_usable_range()
            elif etype == "program":
                self._program = event.value
            elif etype == "pitch" and self._program is not None:
                self._tie_set.add((self._program, event.value))
            return []

        if self._skip_rest:
            return []

        if etype == "shift":
            if event.value > 0:
                self._tick_state = self._start_tick + event.value
                time = self._tick_state / self._frame_rate
                self._capture_continuation(time)
                if time >= self._usable_start_time:
                    return self._enter_usable_range()
        elif etype == "program":
            self._program = event.value
        elif etype == "velocity":
            self._velocity = event.value
        elif etype == "drum":
            time = self._tick_state / self._frame_rate
            self._capture_continuation(time)
            if time < self._usable_start_time or (
                self._next_seek_time is not None and time >= self._next_seek_time
            ):
                return []
            return [*self._enter_usable_range(), _DrumHit(event.value, time)]
        elif etype == "pitch":
            if self._program is None or self._velocity is None:
                return []
            time = self._tick_state / self._frame_rate
            self._capture_continuation(time)
            if self._next_seek_time is not None and time >= self._next_seek_time:
                return []

            key = (self._program, event.value)
            if time < self._usable_start_time:
                if self._velocity > 0:
                    self._context_open.add(key)
                else:
                    self._context_open.discard(key)
                return []

            actions = self._enter_usable_range()
            if self._velocity > 0:
                self._context_open.add(key)
            else:
                self._context_open.discard(key)
            if key in self._open:
                del self._open[key]
                actions.append(_EndNote(*key, time))
            if self._velocity > 0:
                self._open[key] = time
                actions.append(_StartNote(*key, time))
            return actions
        return []

    def finish(self) -> list[_NoteAction]:
        """End of stream: reconcile the final core and close open notes."""
        actions: list[_NoteAction] = []
        if self._chunk_started:
            if self._in_prologue:
                self._context_open.clear()
            actions.extend(self._enter_usable_range())
        actions.extend(
            _EndNote(*key, onset + MINIMUM_NOTE_DURATION_SEC)
            for key, onset in self._open.items()
        )
        self._open.clear()
        return actions

    def _enter_usable_range(self) -> list[_NoteAction]:
        if self._usable_range_started:
            return []
        self._usable_range_started = True
        actions: list[_NoteAction] = []

        ended = [key for key in self._open if key not in self._context_open]
        for key in ended:
            del self._open[key]
            actions.append(_EndNote(*key, self._usable_start_time))

        # In overlapping mode a note can begin in the left context even when
        # the previous chunk missed it. Continue it from the usable boundary
        # without re-emitting the context attack.
        if self._usable_start_time > self._seek_time:
            for key in sorted(self._context_open):
                if key not in self._open:
                    self._open[key] = self._usable_start_time
                    actions.append(_StartNote(*key, self._usable_start_time))
        return actions

    def _capture_continuation(self, time: float) -> None:
        if (
            self._continuation_keys is None
            and self._continuation_time is not None
            and time >= self._continuation_time
        ):
            self._continuation_keys = sorted(self._context_open)

    def open_keys(self) -> list[tuple[int, int]]:
        """Sorted ``(program, pitch)`` pairs in the assembled MIDI state."""
        return sorted(self._open)

    def next_chunk_open_keys(self) -> list[tuple[int, int]]:
        """Notes active where the next model window's context begins."""
        if self._in_prologue:
            return []
        if self._continuation_keys is None:
            self._continuation_keys = sorted(self._context_open)
        return self._continuation_keys


def decode_model_tokens(
    stream: Iterator[int | ChunkBoundary | ProgressEvent],
    vocab: list[Event],
    instrument_for_program: Callable[[int], str],
    frame_rate: int = 100,
) -> Iterator[NoteStartEvent | NoteEndEvent | ProgressEvent]:
    """Stream model token indices straight into NoteStart/NoteEnd events.

    ``stream`` interleaves :class:`ChunkBoundary` markers with token indices:
    each boundary starts a new chunk, followed by that chunk's tokens (EOS and
    anything after it already stripped). Tokens are consumed strictly in
    order: no buffering, no end-of-chunk sort. Each chunk begins with a *tie
    prologue* — ``(program, pitch)`` pairs for notes sustained from the
    previous chunk, terminated by a ``tie`` token — after which any prior open
    note not in that tie set is closed at the chunk boundary. The rest of the
    chunk drives note onsets/offsets directly.

    The decode rules themselves live in :class:`OpenNoteTracker`; this
    generator only turns its actions into events — minting indices, naming
    instruments, and pairing every NoteEndEvent with its NoteStartEvent. For
    overlapping windows, left-context events refine the state before it is
    reconciled with the assembled MIDI at the usable core boundary.
    """
    tracker = OpenNoteTracker(vocab, frame_rate)
    open_notes: dict[tuple[int, int], NoteStartEvent] = {}
    next_index = 0

    def mint(pitch: int, start_time: float, instrument: str) -> NoteStartEvent:
        nonlocal next_index
        ev = NoteStartEvent(
            pitch=pitch, start_time=start_time, index=next_index, instrument=instrument
        )
        next_index += 1
        return ev

    def events_for(
        actions: list[_NoteAction],
    ) -> Iterator[NoteStartEvent | NoteEndEvent]:
        for action in actions:
            if isinstance(action, _EndNote):
                start = open_notes.pop((action.program, action.pitch))
                yield NoteEndEvent(end_time=action.time, start_event=start)
            elif isinstance(action, _StartNote):
                start = mint(
                    action.pitch, action.time, instrument_for_program(action.program)
                )
                open_notes[(action.program, action.pitch)] = start
                yield start
            else:  # _DrumHit: an instantaneous start/end pair
                start = mint(action.pitch, action.time, _DRUM_INSTRUMENT)
                yield start
                yield NoteEndEvent(
                    end_time=action.time + MINIMUM_NOTE_DURATION_SEC, start_event=start
                )

    for item in stream:
        if isinstance(item, ProgressEvent):
            # Advisory progress signal — pass straight through, untouched by
            # the decode state machine.
            yield item
            continue
        yield from events_for(tracker.feed(item))
    yield from events_for(tracker.finish())
