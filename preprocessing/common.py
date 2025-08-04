import mido
import numpy as np
import pandas as pd

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


""" 
def midi_track_length(path: str):
    mid = mido.MidiFile(path)

    ticks_per_beat = mid.ticks_per_beat
    current_tempo = 500000  # Default tempo (120 BPM)
    total_seconds = 0

    # pattern 1 - multiple tempo changes
    for track in mid.tracks:
        for msg in track:
            if msg.type == "set_tempo":
                current_tempo = msg.tempo
                time = msg.time
                total_seconds += (current_tempo/1_000_000) * \
                    (time/ticks_per_beat)

            # pattern 2 - single tempo, end_of_track flag
            if total_seconds == 0:
                max_eot_time = 0
                for msg in track:
                    if msg.type == "set_tempo":
                        current_tempo = msg.tempo

                    if msg.type == "end_of_track":
                        if msg.time > max_eot_time:
                            max_eot_time = msg.time

                total_seconds = (current_tempo/1_000_000) * \
                    (max_eot_time/ticks_per_beat)

            # pattern 3 - single tempo, no end_of_track flag
            if total_seconds == 0:
                note_time = 0
                for msg in track:
                    if msg.type == "set_tempo":
                        current_tempo = msg.tempo

                    if msg.type in ["note_on", "note_off"]:
                        note_time += msg.time

                total_seconds = (current_tempo/1_000_000) * \
                    (note_time/ticks_per_beat)

    return total_seconds """


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
