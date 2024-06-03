from flask import Flask, request, jsonify
from joblib import load

app = Flask(__name__)

# Charger le modèle
model = load('heartAttackrisque.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    # Effectuer la prédiction
    prediction = model.predict([data])[0]  # Suppose que la prédiction est une liste de résultats (à adapter selon votre modèle)
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True, port=53169)
