import os
import pickle
import numpy as np
from flask import Flask, request, jsonify, render_template_string

app = Flask(__name__)

# 1. Load the Naive Bayes Model
# Make sure Naive_Bayes_Model.pkl is in the same directory as app.py
MODEL_PATH = "Naive_Bayes_Model.pkl"

try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
except Exception as e:
    model = None
    print(f"Error loading model: {e}")

# 2. HTML Template with Tailwind CSS for a beautiful, modern UI
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Customer Prediction Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/@tailwindcss/browser@4"></script>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        body { font-family: 'Inter', sans-serif; }
    </style>
</head>
<body class="bg-slate-50 min-h-screen flex items-center justify-center p-4 md:p-8">

    <div class="bg-white shadow-xl rounded-2xl max-w-4xl w-full overflow-hidden border border-slate-100 flex flex-col md:flex-row">
        
        <div class="p-6 md:p-10 md:w-3/5">
            <div class="mb-6">
                <h1 class="text-2xl font-bold text-slate-800">Customer Analytics</h1>
                <p class="text-sm text-slate-500 mt-1">Fill in the metrics below to predict customer behavior.</p>
            </div>

            <form id="predictionForm" class="space-y-4">
                <div class="grid grid-cols-1 sm:grid-cols-2 gap-4">
                    <div>
                        <label class="block text-xs font-semibold text-slate-600 uppercase tracking-wider mb-1">Age</label>
                        <input type="number" name="age" required min="18" max="100" value="30" class="w-full px-3 py-2 border border-slate-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-indigo-500 text-sm">
                    </div>
                    
                    <div>
                        <label class="block text-xs font-semibold text-slate-600 uppercase tracking-wider mb-1">Gender</label>
                        <select name="gender" class="w-full px-3 py-2 border border-slate-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-indigo-500 text-sm bg-white">
                            <option value="0">Female / Other</option>
                            <option value="1">Male</option>
                        </select>
                    </div>

                    <div>
                        <label class="block text-xs font-semibold text-slate-600 uppercase tracking-wider mb-1">City Code (Numeric)</label>
                        <input type="number" name="city" required value="1" class="w-full px-3 py-2 border border-slate-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-indigo-500 text-sm">
                    </div>

                    <div>
                        <label class="block text-xs font-semibold text-slate-600 uppercase tracking-wider mb-1">Tenure (Months)</label>
                        <input type="number" name="tenure_months" required min="0" value="12" class="w-full px-3 py-2 border border-slate-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-indigo-500 text-sm">
                    </div>

                    <div>
                        <label class="block text-xs font-semibold text-slate-600 uppercase tracking-wider mb-1">Avg Order Value ($)</label>
                        <input type="number" step="0.01" name="avg_order_value" required min="0" value="75.50" class="w-full px-3 py-2 border border-slate-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-indigo-500 text-sm">
                    </div>

                    <div>
                        <label class="block text-xs font-semibold text-slate-600 uppercase tracking-wider mb-1">Total Orders</label>
                        <input type="number" name="total_orders" required min="0" value="5" class="w-full px-3 py-2 border border-slate-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-indigo-500 text-sm">
                    </div>

                    <div>
                        <label class="block text-xs font-semibold text-slate-600 uppercase tracking-wider mb-1">Last Purchase (Days Ago)</label>
                        <input type="number" name="last_purchase_days_ago" required min="0" value="14" class="w-full px-3 py-2 border border-slate-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-indigo-500 text-sm">
                    </div>

                    <div>
                        <label class="block text-xs font-semibold text-slate-600 uppercase tracking-wider mb-1">Support Tickets</label>
                        <input type="number" name="support_tickets" required min="0" value="0" class="w-full px-3 py-2 border border-slate-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-indigo-500 text-sm">
                    </div>
                </div>

                <div>
                    <label class="block text-xs font-semibold text-slate-600 uppercase tracking-wider mb-1">Subscription Type</label>
                    <select name="subscription_type" class="w-full px-3 py-2 border border-slate-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-indigo-500 text-sm bg-white">
                        <option value="0">Basic</option>
                        <option value="1">Standard</option>
                        <option value="2">Premium</option>
                    </select>
                </div>

                <button type="submit" class="w-full bg-indigo-600 hover:bg-indigo-700 text-white font-medium py-2.5 px-4 rounded-lg transition duration-200 shadow-md shadow-indigo-100 mt-2 text-sm cursor-pointer">
                    Run Analysis
                </button>
            </form>
        </div>

        <div class="bg-slate-900 p-6 md:p-10 md:w-2/5 flex flex-col justify-center items-center text-center text-white border-t md:border-t-0 md:border-l border-slate-800">
            <div id="resultPlaceholder">
                <div class="w-16 h-16 bg-slate-800 rounded-full flex items-center justify-center mb-4 mx-auto text-indigo-400">
                    📊
                </div>
                <h2 class="text-xl font-semibold mb-2">Awaiting Input</h2>
                <p class="text-slate-400 text-sm max-w-xs">Fill out the customer parameters and click run to view the generated model insights.</p>
            </div>

            <div id="resultDisplay" class="hidden w-full animate-fade-in">
                <span class="text-xs font-bold uppercase tracking-widest text-indigo-400 bg-indigo-950/50 px-3 py-1 rounded-full border border-indigo-900/50">
                    Prediction Result
                </span>
                <div id="predictionValue" class="text-5xl font-extrabold my-6 text-white">
                    --
                </div>
                <div class="bg-slate-800/50 rounded-xl p-4 border border-slate-700/30">
                    <p class="text-xs text-slate-400 mb-1">Confidence Metrics</p>
                    <div id="probabilityValue" class="text-lg font-medium text-emerald-400">
                        --
                    </div>
                </div>
            </div>
        </div>

    </div>

    <script>
        document.getElementById('predictionForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            
            const formData = new FormData(e.target);
            const data = {};
            formData.forEach((value, key) => {
                data[key] = parseFloat(value);
            });

            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(data)
                });

                const result = await response.json();

                if (result.success) {
                    document.getElementById('resultPlaceholder').classList.add('hidden');
                    document.getElementById('resultDisplay').classList.remove('hidden');
                    
                    // Display prediction class
                    document.getElementById('predictionValue').innerText = `Class ${result.prediction}`;
                    
                    // Display probability formatting
                    const probText = result.probabilities.map((p, idx) => `Class ${idx}: ${(p * 100).toFixed(1)}%`).join(' | ');
                    document.getElementById('probabilityValue').innerText = probText;
                } else {
                    alert('Error making prediction: ' + result.error);
                }
            } catch (error) {
                alert('Server communication error.');
                console.error(error);
            }
        });
    </script>
</body>
</html>
"""

@app.route("/")
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"success": False, "error": "Model pickle file is missing or unreadable on the server."}), 500
    
    try:
        data = request.json
        
        # Extract features exactly matching the model's feature_names_in_ array order
        features = [
            float(data.get("age", 0)),
            float(data.get("gender", 0)),
            float(data.get("city", 0)),
            float(data.get("tenure_months", 0)),
            float(data.get("avg_order_value", 0)),
            float(data.get("total_orders", 0)),
            float(data.get("last_purchase_days_ago", 0)),
            float(data.get("support_tickets", 0)),
            float(data.get("subscription_type", 0))
        ]
        
        # Reshape for individual sample inference
        input_data = np.array([features])
        
        # Generate prediction and probabilities
        prediction = int(model.predict(input_data)[0])
        probabilities = model.predict_proba(input_data)[0].tolist()
        
        return jsonify({
            "success": True,
            "prediction": prediction,
            "probabilities": probabilities
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400

if __name__ == "__main__":
    # Local fallback port
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
