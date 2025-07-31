import mido
import pandas as pd


def midi_dataframe(path: str) -> pd.DataFrame:
    """create dataframe of note_on messages from midi file"""

    # basic io
    mid = mido.MidiFile(path)

    # isolate note_on events and concatenate tracks
    events = []
    for track in mid.tracks:
        for msg in track:
            if msg.type == "note_on":
                events.append(msg)

    # parsing messages into features
    events_df = pd.DataFrame({"raw_events": events})

    features = ["channel", "note", "velocity", "time"]

    for feature in features:
        events_df[feature] = events_df["raw_events"].apply(
            lambda x: getattr(x, feature))

    return events_df
