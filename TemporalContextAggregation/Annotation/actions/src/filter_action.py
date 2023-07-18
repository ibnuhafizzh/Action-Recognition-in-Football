import json

# Langkah 1: Baca file JSON yang ada
with open('/Users/ibnuhafizh/Documents/ITB/TA/Action-Recognition-in-Football/TemporalContextAggregation/Annotation/actions/src/results_spotting.json', 'r') as file:
    data = json.load(file)

# Langkah 2 & 3: Iterasi dan filter aksi dengan 'confidence' >= 0.5
filtered_actions = []
url = data.get('UrlLocal')
for action in data.get('predictions', []):
    confidence = action.get('confidence')
    if confidence is not None:
        # Konversi nilai 'confidence' ke float (jika dalam bentuk string)
        try:
            confidence_value = float(confidence)
        except ValueError:
            continue

        if confidence_value >= 0.001:
            filtered_actions.append(action)

# Langkah 4: Simpan data yang telah difilter ke dalam file JSON baru
with open('filtered_output2.json', 'w') as output_file:
    json.dump({'UrlLocal':url,'predictions': filtered_actions}, output_file, indent=4)

print("Aksi dengan 'confidence' >= 0.01 telah disimpan dalam 'filtered_output.json'")
