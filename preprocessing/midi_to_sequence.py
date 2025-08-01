import os
from pathlib import Path

# List of target composers
target_compsers = ['Bach', 'Beethoven', 'Chopin', 'Mozart']

# Folder containing the dataset
dataset_folder = Path('../data/midi/archive/midiclassics')

# Storing paths of MIDI files for each composer
dataset = []

# Avoid dublicate filenames
seen_filenames = set()

#Looping through each target composer folder
for composer in target_compsers:
    composer_folder = dataset_folder / composer

    # Recursively find all MIDI files in all subfolders in the composer's folder
    for file_path in composer_folder.rglob('*'): # Iterate over this subtree and yield all existing files
        if file_path.suffix.lower() in ['.mid', '.midi']: # Only consider MIDI files
            
            # Check if the filename has already been seen
            if file_path.name not in seen_filenames:
                seen_filenames.add(file_path.name)
                dataset.append((str(file_path), composer)) # Add the file path to the dataset

# Check total files found 
print(f"Total MIDI files found: {len(dataset)}")

# Display the first 10 file paths
for path, composer in dataset[1500:1550]:
    print(f"Composer: {composer}, File Path: {path}")
