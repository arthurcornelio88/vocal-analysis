"""Conversões de pitch entre Hz, MIDI e notação científica."""

import numpy as np

# Referência: A4 = 440 Hz = MIDI 69
A4_HZ = 440.0
A4_MIDI = 69

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def hz_to_midi(freq: float | np.ndarray) -> float | np.ndarray:
    """Converte frequência em Hz para número MIDI.

    Args:
        freq: Frequência em Hz (escalar ou array).

    Returns:
        Número MIDI (float ou array).
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        midi = 12 * np.log2(np.asarray(freq) / A4_HZ) + A4_MIDI
    return midi


def midi_to_hz(midi: float | np.ndarray) -> float | np.ndarray:
    """Converte número MIDI para frequência em Hz.

    Args:
        midi: Número MIDI (escalar ou array).

    Returns:
        Frequência em Hz.
    """
    return A4_HZ * (2 ** ((np.asarray(midi) - A4_MIDI) / 12))


def hz_to_note(freq: float, include_cents: bool = False) -> str:
    """Converte frequência em Hz para notação científica (ex: A4, C#5).

    Args:
        freq: Frequência em Hz.
        include_cents: Se True, inclui desvio em cents (ex: A4+15).

    Returns:
        Nota em notação científica.
    """
    if freq <= 0 or np.isnan(freq):
        return "–"

    midi = hz_to_midi(freq)
    midi_rounded = int(round(midi))

    note_index = midi_rounded % 12
    octave = (midi_rounded // 12) - 1  # MIDI 60 = C4

    note_name = NOTE_NAMES[note_index]

    if include_cents:
        cents = int(round((midi - midi_rounded) * 100))
        if cents > 0:
            return f"{note_name}{octave}+{cents}"
        elif cents < 0:
            return f"{note_name}{octave}{cents}"

    return f"{note_name}{octave}"


def note_to_hz(note: str) -> float:
    """Converte notação científica para frequência em Hz.

    Args:
        note: Nota em notação científica (ex: A4, C#5, Db3).

    Returns:
        Frequência em Hz.
    """
    note = note.strip().upper()

    # Parse note name
    if len(note) >= 2 and note[1] in ("#", "B"):
        note_name = note[:2]
        octave_str = note[2:]
        # Handle flats
        if note_name[1] == "B":
            idx = NOTE_NAMES.index(note_name[0])
            note_name = NOTE_NAMES[(idx - 1) % 12]
    else:
        note_name = note[0]
        octave_str = note[1:]

    if note_name not in NOTE_NAMES:
        raise ValueError(f"Nota inválida: {note}")

    octave = int(octave_str)
    note_index = NOTE_NAMES.index(note_name)

    midi = (octave + 1) * 12 + note_index
    return midi_to_hz(midi)


def hz_range_to_notes(freq_min: float, freq_max: float) -> str:
    """Formata um range de frequências como notas.

    Args:
        freq_min: Frequência mínima em Hz.
        freq_max: Frequência máxima em Hz.

    Returns:
        String formatada (ex: "G3 – C6").
    """
    return f"{hz_to_note(freq_min)} – {hz_to_note(freq_max)}"
