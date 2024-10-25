# Step 1: Install required packages
# Uncomment these lines if you haven't installed these packages yet
# !pip install pretty_midi tensorflow scikit-learn

import pretty_midi
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras import layers, models

# Step 2: Define the path to your MIDI directory
midi_dir = '/home/lilly/Desktop/miniprojectttttttt/lmd_full-20241015T171145Z-001/'

# Step 3: Function to convert MIDI file to note pitches
def midi_to_notes(midi_file):
    try:
        midi_data = pretty_midi.PrettyMIDI(midi_file)
        note_pitches = []
        for instrument in midi_data.instruments:
            if not instrument.is_drum:  # Exclude drum tracks
                for note in instrument.notes:
                    note_pitches.append(note.pitch)  # Collect note pitch
        return note_pitches
    except Exception as e:
        print(f"Error processing {midi_file}: {e}")
        return []

# Step 4: List MIDI files in the directory
midi_files = [os.path.join(root, file)
              for root, _, files in os.walk(midi_dir)
              for file in files if file.endswith('.mid')]

print(f"MIDI files found: {len(midi_files)}")  # Print the number of MIDI files found

# Step 5: Limit the number of MIDI files processed
n_files_to_process = 200  # Change this to whatever number you prefer
midi_files = midi_files[:n_files_to_process]

# Step 6: Extract notes from the MIDI files
all_notes = []
for midi_file in midi_files:  # Now processing a limited set of files
    print(f"Processing: {midi_file}")
    notes = midi_to_notes(midi_file)
    all_notes.extend(notes)

print(f"Extracted {len(all_notes)} note pitches.")  # Print the number of extracted notes

# Step 7: Encode note names as numerical labels
note_names = [pretty_midi.note_number_to_name(pitch) for pitch in all_notes]
label_encoder = LabelEncoder()
note_labels = label_encoder.fit_transform(note_names)

# Step 8: Prepare feature matrix (X) and target labels (y)
X = np.array(all_notes).reshape(-1, 1)
y = np.array(note_labels)

# Step 9: Split data into training and testing sets
if len(X) == 0 or len(y) == 0:
    print("No notes extracted, cannot proceed with train-test split.")
else:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Training samples: {X_train.shape[0]}, Testing samples: {X_test.shape[0]}")

    # Step 10: Build a simple neural network model
    model = models.Sequential([
        layers.InputLayer(input_shape=(1,)),  # Input is the note pitch
        layers.Dense(256, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(len(label_encoder.classes_), activation='softmax')  # Output: number of unique notes
    ])

    # Step 11: Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Step 12: Train the model
    history = model.fit(X_train, y_train, epochs=10, batch_size=64, validation_split=0.2)

    # Step 13: Evaluate on the test set
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f"Test accuracy: {test_acc:.4f}")

    # Step 14: Predict on new MIDI data
    def predict_notes(midi_file):
        pitches = midi_to_notes(midi_file)
        if pitches:
            pitches = np.array(pitches).reshape(-1, 1)
            predicted_labels = model.predict(pitches)
            predicted_notes = label_encoder.inverse_transform(np.argmax(predicted_labels, axis=1))
            return predicted_notes
        else:
            return []

    # Example prediction
    if len(midi_files) > 0:  # Ensure there's enough data
        new_midi_file = midi_files[0]  # Take the first MIDI file for prediction
        predicted_notes = predict_notes(new_midi_file)
        print(f"Predicted notes: {predicted_notes}")

    # Evaluate the model on the test set again if needed
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f"Test accuracy: {test_acc:.4f}")
