from flask import Flask, request, jsonify
import pretty_midi
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
import os

# Load your trained model
model = load_model('/home/lilly/Desktop/miniprojectttttttt/use_model.h5')  # Adjust the path
label_encoder = LabelEncoder()

app = Flask(__name__)

def midi_to_notes(midi_file_path):
    midi_data = pretty_midi.PrettyMIDI(midi_file_path)
    note_pitches = []
    for instrument in midi_data.instruments:
        if not instrument.is_drum:
            for note in instrument.notes:
                note_pitches.append(note.pitch)
    return note_pitches

def predict_notes(midi_file_path):
    pitches = midi_to_notes(midi_file_path)
    if pitches:
        pitches = np.array(pitches).reshape(-1, 1)
        predicted_labels = model.predict(pitches)
        predicted_notes = label_encoder.inverse_transform(np.argmax(predicted_labels, axis=1))
        return predicted_notes.tolist()
    return []

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        file_path = os.path.join('uploads', file.filename)
        file.save(file_path)
        notes = predict_notes(file_path)
        os.remove(file_path)
        return jsonify({'notes': notes})

if __name__ == '__main__':
    app.run(debug=True)
    


