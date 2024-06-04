import os
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
    port = int(os.environ.get('PORT', 53169))  # Utilisez le port défini par la variable d'environnement PORT ou 53169 par défaut
    app.run(debug=True, host='0.0.0.0', port=port)
