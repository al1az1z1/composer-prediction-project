import numpy as np
import pandas as pd
from scipy.io.wavfile import write
import matplotlib.pyplot as plt

import mido
import pretty_midi
import librosa
import librosa.display


pd.options.mode.chained_assignment = None


def midi_notes_dataframe(path: str) -> pd.DataFrame:
    """
    Create a DataFrame of MIDI note_on and note_off messages from a MIDI file.

    Args:
        path (str): Path to the MIDI file.

    Returns:
        pd.DataFrame: DataFrame containing MIDI events with columns:
            - raw_events: Original mido message objects.
            - event_type: 'note_on' or 'note_off'.
            - channel (int): MIDI channel number.
            - note (int): MIDI note number [0, 127].
            - velocity (int): Note velocity [0, 127], note_off
            velocity set to 0).
            - time (int): Delta time in ticks since the previous event.
    """
    mid = mido.MidiFile(path)
    events = []
    for track in mid.tracks:
        for msg in track:
            if msg.type in ["note_on", "note_off"]:
                events.append(msg)

    events_df = pd.DataFrame({"raw_events": events})
    events_df["event_type"] = events_df["raw_events"].apply(lambda x: x.type)

    features = ["channel", "note", "velocity", "time"]
    for feature in features:
        events_df[feature] = events_df["raw_events"].apply(
            lambda x: getattr(x, feature))

    events_df.loc[events_df["event_type"] == "note_off", "velocity"] = 0
    return events_df


def midi_events_dataframe(path: str,
                          event_types: list | None = None) -> pd.DataFrame:
    """
    Create a DataFrame of MIDI messages from a file,
    optionally filtered by event types.

    Args:
        path (str): Path to the MIDI file.
        event_types (list | None): Optional list of
        event type strings to include.
            If None, all event types are included.

    Returns:
        pd.DataFrame: DataFrame containing 'raw_events'
        with mido message objects.
    """
    mid = mido.MidiFile(path)
    events = []
    for track in mid.tracks:
        if event_types:
            for msg in track:
                if msg.type in event_types:
                    events.append(msg)
        else:
            for msg in track:
                events.append(msg)

    return pd.DataFrame({"raw_events": events})


def midi_track_length(path: str) -> int:
    """
    Compute the total length of a MIDI file in seconds.

    Args:
        path (str): Path to the MIDI file.

    Returns:
        int: Total number of ticks from start to end of the file.
    """
    mid = mido.MidiFile(path)
    absolute_time_ticks = 0
    for msg in mid:
        absolute_time_ticks += msg.time
    return absolute_time_ticks


def hold_ticks(dataframe: pd.DataFrame, row_index: int) -> int:
    """
    Determine how long a note is held, in ticks, starting from a given row.

    Args:
        dataframe (pd.DataFrame): DataFrame containing MIDI events, including
            'note', 'velocity', and 'cum_time' columns.
        row_index (int): Row index where the note_on event occurs.

    Returns:
        int: Duration of the note hold in ticks. Returns None
        if no matching note_off found.
    """
    subset = dataframe.iloc[row_index:,]

    if subset["velocity"].values[0] == 0:
        return None
    else:
        note = subset["note"].values[0]
        start_time = subset["cum_time"].values[0]
        i_rows = subset.shape[0]

        for i in range(i_rows - 1):
            if (subset["note"].values[i+1] == note) & \
                    (subset["velocity"].values[i+1] == 0):
                end_time = subset["cum_time"].values[i+1]
                break
            elif i == i_rows - 1:
                end_time = subset["cum_time"].values[i]

        return end_time - start_time


def update_array(arr: np.ndarray,
                 note: int, start_tick: int,
                 ticks: int) -> np.ndarray:
    """
    Mark a range of ticks in an array to indicate when a note is played.

    Args:
        arr (np.ndarray): 2D array of shape (128, total_ticks),
        one row per MIDI note.
        note (int): MIDI note number [0, 127].
        start_tick (int): Start tick index.
        ticks (int): Duration in ticks to mark as active.

    Returns:
        np.ndarray: Modified array with note spans set to 1.
    """
    arr[note, start_tick:start_tick+ticks] = np.ones((1, ticks))
    return arr


def base_array(dataframe: pd.DataFrame) -> np.ndarray:
    """
    Create an empty note-tick array for a given MIDI channel's event DataFrame.

    Args:
        dataframe (pd.DataFrame): Must contain a 'cum_time' column.

    Returns:
        np.ndarray: Zero-filled array of shape (128, total_ticks).
    """
    ticks = dataframe["cum_time"].max()
    return np.zeros((128, ticks))


def interpolate_channel(dataframe: pd.DataFrame) -> np.ndarray:
    """
    Generate a note-tick matrix for a single MIDI
    channel by interpolating note holds.

    Args:
        dataframe (pd.DataFrame): DataFrame containing at least
            'note', 'velocity', and 'time' columns.

    Returns:
        np.ndarray: 2D array (128 total_ticks) marking note activity.
    """
    dataframe["cum_time"] = dataframe["time"].cumsum()
    array = base_array(dataframe)
    for i in range(dataframe.shape[0]):
        note = dataframe["note"].values[i]
        ticks = hold_ticks(dataframe, i)
        if ticks:
            start_tick = dataframe["cum_time"].values[i]
            array = update_array(array, note, start_tick, ticks)
    return array


def interpolate_multichannel(dataframe: pd.DataFrame) -> list[np.ndarray]:
    """
    Apply interpolation to all unique channels in a DataFrame.

    Args:
        dataframe (pd.DataFrame): Must contain 'channel' column.

    Returns:
        list[np.ndarray]: List of arrays, one for each channel.
    """
    channel_index = dataframe["channel"].unique()
    return [interpolate_channel(dataframe[dataframe["channel"] == i])
            for i in channel_index]


def pad_array(array: np.ndarray, max_cols: int) -> np.ndarray:
    """
    Pad a note-tick array with zeros to match a target number of columns.

    Args:
        array (np.ndarray): Array of shape (128, current_ticks).
        max_cols (int): Target number of columns.

    Returns:
        np.ndarray: Padded array with shape (128, max_cols).
    """
    width = array.shape[1]
    if width == max_cols:
        return array
    else:
        pad_array = np.zeros((127, max_cols - width))
        return np.append(array, pad_array, axis=1)


def midi_to_array(dataframe: pd.DataFrame) -> np.ndarray:
    """
    Convert multi-channel MIDI DataFrame into a normalized 3D array.

    Args:
        dataframe (pd.DataFrame): Must contain at least
        'channel', 'note', 'velocity', and 'time'.

    Returns:
        np.ndarray: Shape (num_channels, 128, max_ticks),
        padded to max channel length.
    """
    arrays = interpolate_multichannel(dataframe)
    max_ticks = np.max([arr.shape[1] for arr in arrays])
    return np.array([pad_array(arr, max_ticks) for arr in arrays])


def sample_validator(array: np.ndarray, silence_threshold: float) -> bool:
    """
    Check whether a note array's silence rate is within the allowed threshold.

    Args:
        array (np.ndarray): 2D array where nonzero columns represent sound.
        silence_threshold (float): Maximum allowable silence proportion (0–1).

    Returns:
        bool: True if silence rate <= threshold, else False.
    """
    loud_cells = (np.sum(array, axis=0) != 0).sum()
    silence_rate = 1 - (loud_cells / array.shape[1])
    return silence_rate <= silence_threshold


def reducer_sampler(array: np.ndarray,
                    duration: float,
                    sample_seconds: int,
                    silence_threshold: float,
                    attempts: int = 100) -> tuple[int, int, np.ndarray]:
    """
    Reduce a 3D channel-note-tick array to 2D and sample a time window.

    Args:
        array (np.ndarray): Shape (num_channels, 128, total_ticks).
        duration (float): Total song duration in seconds.
        sample_seconds (int): Length of sample to extract in seconds.
        silence_threshold (float): Maximum silence proportion allowed.
        attempts (int, optional): Max number of attempts to find a valid
        sample.

    Returns:
        tuple[int, int, np.ndarray]:
            - start_index (int): Starting tick index.
            - end_index (int): Ending tick index.
            - sample (np.ndarray): 2D array slice of the sample window.

    Raises:
        ValueError: If no valid sample is found after given attempts.
    """
    reduced_array = np.sum(array, axis=0)
    ticks_per_second = int(reduced_array.shape[1] / duration)
    n = ticks_per_second * sample_seconds

    passed_validation = False
    while attempts > 0:
        index = np.random.randint(0, reduced_array.shape[1] - n)
        sample = reduced_array[:, index:index + n]
        passed_validation = sample_validator(sample, silence_threshold)
        if passed_validation:
            break
        attempts -= 1

    if passed_validation:
        return index, index + n, sample
    else:
        raise ValueError("Could not pull a valid sample.")


# Functions to generate and sample MEL spectrograms.


def midi_to_wav(midi_path: str, wav_path: str, soundfont_path: str) -> None:
    """
    Converts a MIDI file to a WAV audio file using a specified SoundFont.

    Args:
        midi_path (str): Path to the input MIDI file.
        wav_path (str): Path where the output WAV file will be saved.
        soundfont_path (str): Path to the SoundFont (.sf2)
        file used for synthesis.

    Returns:
        None
    """
    fs = 44100
    midi_data = pretty_midi.PrettyMIDI(midi_path)
    audio_data = midi_data.fluidsynth(fs=fs, sf2_path=soundfont_path)

    write(wav_path, fs, (audio_data * 32767).astype("int16"))


def midi_to_soundfont_array_(midi_path: str, soundfont_path: str) -> np.array:
    """
    Converts a MIDI file to a NumPy array of audio samples using a SoundFont.

    Args:
        midi_path (str): Path to the MIDI file.
        soundfont_path (str): Path to the SoundFont (.sf2) file.

    Returns:
        np.array: Array of int16 audio samples.
    """
    fs = 44100
    midi_data = pretty_midi.PrettyMIDI(midi_path)
    audio_data = midi_data.fluidsynth(fs=fs, sf2_path=soundfont_path)

    return (audio_data * 32767).astype("int16")


def midi_to_spectrogram(midi_path: str,
                        soundfont_path: str) -> tuple[np.array, int]:
    """
    Converts a MIDI file to a mel spectrogram using a SoundFont.

    Args:
        midi_path (str): Path to the MIDI file.
        soundfont_path (str): Path to the SoundFont (.sf2) file.

    Returns:
        tuple[np.array, int]: A tuple containing the dB-scaled
                              mel spectrogram (2D array) and the sample rate.
    """
    midi_data = pretty_midi.PrettyMIDI(midi_path)
    sr = 44100
    audio_data = midi_data.fluidsynth(fs=sr, sf2_path=soundfont_path)
    S = librosa.feature.melspectrogram(y=audio_data, sr=sr, n_mels=128)
    S_db = librosa.power_to_db(S, ref=np.max)
    return S_db, sr


def audio_to_spectrogram(wav_path: str) -> tuple[np.array, int]:
    """
    Loads an audio file and converts it to a mel spectrogram.

    Args:
        wav_path (str): Path to the WAV file.

    Returns:
        tuple[np.array, int]: A tuple containing the dB-scaled mel
                              spectrogram (2D array) and the sample rate.
    """
    y, sr = librosa.load(wav_path, sr=None)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    S_db = librosa.power_to_db(S, ref=np.max)
    return S_db, sr


def spectrogram_sample(S_db: np.array,
                       sr: int,
                       sample_seconds: int,
                       hop_length=512,
                       start=None) -> tuple[int, int, np.array]:
    """
    Extracts a random fixed-duration sample from a spectrogram.

    Args:
        S_db (np.array): Input mel spectrogram (2D array).
        sr (int): Sample rate of the original audio.
        sample_seconds (int): Desired duration of the sample in seconds.
        hop_length (int): Hop length used in the spectrogram (default is 512).
        random

    Returns:
        tuple[int, int, np.array]: A tuple containing the start index,
                                   stop index, and the sliced
                                   spectrogram sample.

    Raises:
        ValueError: If sample_seconds is greater than the total track length.
    """
    ticks = S_db.shape[1]
    track_length = ticks * hop_length / sr
    if sample_seconds > track_length:
        raise ValueError(
            f"sample_seconds must be less than or equal to track length: "
            f"{track_length}"
        )
    width = int(sample_seconds * sr / hop_length)
    if start is None:
        start = np.random.randint(0, ticks - width)
    stop = start + width
    return start, stop, S_db[:, start:stop]


def plot_spectrogram(S_db: np.array, sr: int, title="Spectrogram") -> None:
    """
    Plots a mel spectrogram using librosa's visualization tools.

    Args:
        S_db (np.array): dB-scaled mel spectrogram to plot.
        sr (int): Sample rate of the audio.
        title (str): Title of the plot (default is "Spectrogram").

    Returns:
        None
    """
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.tight_layout()
    plt.show()
