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
    """create dataframe of note_on and note_off messages from midi file"""

    # basic io
    mid = mido.MidiFile(path)

    # isolate note_on events and concatenate tracks
    events = []
    for track in mid.tracks:
        for msg in track:
            if msg.type in ["note_on", "note_off"]:
                events.append(msg)

    # parsing messages into features
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
    """create dataframe of all messages from midi file"""

    # basic io
    mid = mido.MidiFile(path)

    # collect events
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


def midi_track_length(path: str):
    mid = mido.MidiFile(path)
    absolute_time_ticks = 0

    for msg in mid:
        absolute_time_ticks += msg.time

    return absolute_time_ticks


def hold_ticks(dataframe: pd.DataFrame, row_index: int) -> int:
    """determines how long a note is held in ticks"""
    subset = dataframe.iloc[row_index:,]

    if subset["velocity"].values[0] == 0:
        pass
    else:
        note = subset["note"].values[0]
        start_time = subset["cum_time"].values[0]
        i_rows = subset.shape[0]

        for i in range(i_rows-1):
            if (subset["note"].values[i+1] == note) & \
                    (subset["velocity"].values[i+1] == 0):

                end_time = subset["cum_time"].values[i+1]
                break
            elif i == i_rows-1:
                end_time = subset["cum_time"].values[i]

        return end_time - start_time


def update_array(arr: np.array,
                 note: int,
                 start_tick: int,
                 ticks: int) -> np.array:
    """update array to indicate spans of note plays"""

    arr[note, start_tick:start_tick+ticks] = np.ones((1, ticks))
    return arr


def base_array(dataframe):
    """creates an empty array with a row for each possible note
    and a column for every tick in the song"""

    ticks = dataframe["cum_time"].max()
    return np.zeros((127, ticks))


def interpolate_channel(dataframe: pd.DataFrame) -> np.array:
    """executes interpolation pipeline against a single channel"""
    # create feature with cumulative time
    dataframe["cum_time"] = dataframe["time"].cumsum()

    # create an empty array for the channel
    array = base_array(dataframe)

    # iterate through dataframe to discover note plays
    # and interpolate over hold lengths in ticks
    for i in range(dataframe.shape[0]):
        note = dataframe["note"].values[i]
        ticks = hold_ticks(dataframe, i)
        if ticks:
            start_tick = dataframe["cum_time"].values[i]
            array = update_array(array, note, start_tick, ticks)

    return array


def interpolate_multichannel(dataframe):
    """execute interploation pipeline over a batch of channels"""

    channel_index = dataframe["channel"].unique()
    return [
        interpolate_channel(dataframe[dataframe["channel"] == i])
        for i in channel_index
    ]


def pad_array(array: np.array, max_cols) -> np.array:
    """pad arrays to ensure dimensional homogeneity"""

    width = array.shape[1]
    if width == max_cols:
        return array
    else:
        pad_array = np.zeros((127, max_cols-width))
        return np.append(array, pad_array, axis=1)


def midi_to_array(dataframe) -> np.array:
    """execute full pipeline to interpolate over a batch of channels,
    normalize the respective arrays, and convert to np.array"""

    arrays = interpolate_multichannel(dataframe)
    max_ticks = np.max([arr.shape[1] for arr in arrays])
    return np.array([pad_array(arr, max_ticks) for arr in arrays])


def sample_validator(array: np.array, silence_threshold: float) -> bool:
    """Returns True if silence rate is less than
     or equal to silence_threshold"""

    loud_cells = (np.sum(array, axis=0) != 0).sum()
    silence_rate = 1-(loud_cells/array.shape[1])

    return silence_rate <= silence_threshold


def reducer_sampler(array: np.array,
                    duration: float,
                    sample_seconds: int,
                    silence_threshold: float,
                    attempts: int = 100) -> tuple[int, int, np.array]:
    """Reduces 3d tensor to 2d tensor and samples n random columns."""

    reduced_array = np.sum(array, axis=0)
    ticks_per_second = int(reduced_array.shape[1]/duration)
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
        return index, index+n, sample
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
                       hop_length=512) -> tuple[int, int, np.array]:
    """
    Extracts a random fixed-duration sample from a spectrogram.

    Args:
        S_db (np.array): Input mel spectrogram (2D array).
        sr (int): Sample rate of the original audio.
        sample_seconds (int): Desired duration of the sample in seconds.
        hop_length (int): Hop length used in the spectrogram (default is 512).

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
