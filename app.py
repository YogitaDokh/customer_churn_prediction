import os
import pickle
import numpy as np
from flask import Flask, request, jsonify, render_template_string

app = Flask(__name__)

# Load the Naive Bayes Model securely
MODEL_PATH = "Naive_Bayes_Model.pkl"

try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
except Exception as e:
    model = None
    print(f"Error loading model: {e}")

# Static Categorical Mapping Encoders
GENDER_MAP = {"Female": 0, "Male": 1, "Other": 2}
SUBSCRIPTION_MAP = {"Basic": 0, "Standard": 1, "Premium": 2}

# Re-engineered HTML Template with clean string concatenation to avoid escaping bugs
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Predictive Customer Analytics</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        body { font-family: 'Plus Jakarta Sans', sans-serif; }
    </style>
</head>
<body class="bg-gradient-to-br from-slate-50 to-indigo-50/50 min-h-screen flex items-center justify-center p-4 md:p-8">

    <div class="bg-white shadow-2xl rounded-3xl max-w-5xl w-full overflow-hidden border border-slate-100 flex flex-col lg:flex-row min-h-[640px]">
        
        <div class="p-8 lg:p-12 lg:w-3/5 flex flex-col justify-between">
            <div>
                <div class="flex items-center gap-3 mb-3">
                    <span class="p-2 bg-indigo-50 text-indigo-600 rounded-xl font-bold text-xl">🧠</span>
                    <h1 class="text-2xl font-bold text-slate-800 tracking-tight">Customer Intelligence</h1>
                </div>
                <p class="text-sm text-slate-500 mb-8">Input real-time customer data to predict behavioral classifications instantly.</p>

                <form id="predictionForm" class="space-y-5">
                    <div class="grid grid-cols-1 sm:grid-cols-2 gap-5">
                        <div class="flex flex-col gap-1.5">
                            <label class="text-xs font-semibold text-slate-600 uppercase tracking-wider">Age (Years)</label>
                            <input type="number" name="age" required min="18" max="100" value="34" class="w-full px-3.5 py-2.5 border border-slate-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-indigo-500/20 focus:border-indigo-500 text-sm transition-all text-slate-700 font-medium">
                        </div>
                        
                        <div class="flex flex-col gap-1.5">
                            <label class="text-xs font-semibold text-slate-600 uppercase tracking-wider">Gender</label>
                            <select name="gender" class="w-full px-3.5 py-2.5 border border-slate-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-indigo-500/20 focus:border-indigo-500 text-sm transition-all text-slate-700 font-medium bg-white">
                                <option value="Female">Female</option>
                                <option value="Male" selected>Male</option>
                                <option value="Other">Other</option>
                            </select>
                        </div>

                        <div class="flex flex-col gap-1.5">
                            <label class="text-xs font-semibold text-slate-600 uppercase tracking-wider">City Location</label>
                            <input type="text" name="city" required placeholder="e.g. San Francisco, Paris, Tokyo" value="London" class="w-full px-3.5 py-2.5 border border-slate-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-indigo-500/20 focus:border-indigo-500 text-sm transition-all text-slate-700 font-medium">
                        </div>

                        <div class="flex flex-col gap-1.5">
                            <label class="text-xs font-semibold text-slate-600 uppercase tracking-wider">Account Tenure (Months)</label>
                            <input type="number" name="tenure_months" required min="0" value="18" class="w-full px-3.5 py-2.5 border border-slate-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-indigo-500/20 focus:border-indigo-500 text-sm transition-all text-slate-700 font-medium">
                        </div>

                        <div class="flex flex-col gap-1.5">
                            <label class="text-xs font-semibold text-slate-600 uppercase tracking-wider">Avg. Order Value ($)</label>
                            <input type="number" step="0.01" name="avg_order_value" required min="0" value="112.40" class="w-full px-3.5 py-2.5 border border-slate-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-indigo-500/20 focus:border-indigo-500 text-sm transition-all text-slate-700 font-medium">
                        </div>

                        <div class="flex flex-col gap-1.5">
                            <label class="text-xs font-semibold text-slate-600 uppercase tracking-wider">Total Orders Placed</label>
                            <input type="number" name="total_orders" required min="0" value="14" class="w-full px-3.5 py-2.5 border border-slate-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-indigo-500/20 focus:border-indigo-500 text-sm transition-all text-slate-700 font-medium">
                        </div>

                        <div class="flex flex-col gap-1.5">
                            <label class="text-xs font-semibold text-slate-600 uppercase tracking-wider">Days Since Last Purchase</label>
                            <input type="number" name="last_purchase_days_ago" required min="0" value="6" class="w-full px-3.5 py-2.5 border border-slate-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-indigo-500/20 focus:border-indigo-500 text-sm transition-all text-slate-700 font-medium">
                        </div>

                        <div class="flex flex-col gap-1.5">
                            <label class="text-xs font-semibold text-slate-600 uppercase tracking-wider">Open Support Tickets</label>
                            <input type="number" name="support_tickets" required min="0" value="0" class="w-full px-3.5 py-2.5 border border-slate-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-indigo-500/20 focus:border-indigo-500 text-sm transition-all text-slate-700 font-medium">
                        </div>
                    </div>

                    <div class="flex flex-col gap-1.5 pt-1">
                        <label class="text-xs font-semibold text-slate-600 uppercase tracking-wider">Subscription Tiers</label>
                        <select name="subscription_type" class="w-full px-3.5 py-2.5 border border-slate-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-indigo-500/20 focus:border-indigo-500 text-sm transition-all text-slate-700 font-medium bg-white">
                            <option value="Basic">Basic Plan</option>
                            <option value="Standard">Standard Tier</option>
                            <option value="Premium" selected>Premium Experience</option>
                        </select>
                    </div>
                </form>
            </div>

            <button type="submit" form="predictionForm" class="w-full bg-indigo-600 hover:bg-indigo-700 active:bg-indigo-800 text-white font-semibold py-3 px-4 rounded-xl transition duration-200 shadow-lg shadow-indigo-100 mt-8 text-sm cursor-pointer flex items-center justify-center gap-2">
                Evaluate Analytics Profile
            </button>
        </div>

        <div class="bg-slate-900 p-8 lg:p-12 lg:w-2/5 flex flex-col justify-center items-center text-center text-white relative overflow-hidden">
            <div class="absolute -top-12 -right-12 w-40 h-40 bg-indigo-500/10 rounded-full blur-2xl"></div>
            <div class="absolute -bottom-12 -left-12 w-40 h-40 bg-emerald-500/10 rounded-full blur-2xl"></div>

            <div id="resultPlaceholder" class="relative z-10">
                <div class="w-20 h-20 bg-slate-800/80 backdrop-blur-md rounded-2xl flex items-center justify-center mb-6 mx-auto border border-slate-700/50 shadow-inner text-2xl">
                    ⚡
                </div>
                <h2 class="text-xl font-bold tracking-tight mb-2">Ready for Execution</h2>
                <p class="text-slate-400 text-sm max-w-xs mx-auto leading-relaxed">Provide feature sets inside the console to generate real-time predictive indices.</p>
            </div>

            <div id="resultLoading" class="hidden relative z-10 flex flex-col items-center">
                <div class="w-12 h-12 border-4 border-indigo-500 border-t-transparent rounded-full animate-spin mb-4"></div>
                <p class="text-sm text-slate-300">Processing metrics through model layers...</p>
            </div>

            <div id="resultDisplay" class="hidden w-full relative z-10 space-y-6">
                <div>
                    <span class="text-[10px] font-bold uppercase tracking-widest text-indigo-400 bg-indigo-950/60 px-3.5 py-1.5 rounded-full border border-indigo-500/30">
                        Target Prediction Class
                    </span>
                </div>
                
                <div id="predictionValue" class="text-6xl font-extrabold tracking-tighter text-transparent bg-clip-text bg-gradient-to-r from-white to-slate-300 py-2">
                    --
                </div>
                
                <div class="bg-slate-800/40 backdrop-blur-md rounded-2xl p-5 border border-slate-700/40 max-w-xs mx-auto">
                    <p class="text-[11px] uppercase tracking-wider font-semibold text-slate-400 mb-3">Class Probabilities</p>
                    <div id="probabilityContainer" class="space-y-2 text-sm font-medium text-left">
                        </div>
                </div>
            </div>
        </div>

    </div>

    <script>
        document.getElementById('predictionForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            
            const placeholder = document.getElementById('resultPlaceholder');
            const loader = document.getElementById('resultLoading');
            const display = document.getElementById('resultDisplay');

            placeholder.classList.add('hidden');
            display.classList.add('hidden');
            loader.classList.remove('hidden');
            
            const formData = new FormData(e.target);
            const payload = {};
            formData.forEach((value, key) => {
                payload[key] = isNaN(value) || value.trim() === '' ? value : parseFloat(value);
            });

            try {
                await new Promise(resolve => setTimeout(resolve, 400));

                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });

                const result = await response.json();
                loader.classList.add('hidden');

                if (result.success) {
                    display.classList.remove('hidden');
                    document.getElementById('predictionValue').innerText = result.prediction;
                    
                    const probContainer = document.getElementById('probabilityContainer');
                    probContainer.innerHTML = '';
                    
                    result.probabilities.forEach((prob, idx) => {
                        const percentage = (prob * 100).toFixed(1);
                        
                        // Using classic concatenation to completely avoid template-literal character clashing with python
                        let htmlRow = '<div class="space-y-1">';
                        htmlRow += '  <div class="flex justify-between text-xs text-slate-300 font-normal">';
                        htmlRow += '    <span>Class ' + idx + '</span>';
                        htmlRow += '    <span class="font-bold text-indigo-400">' + percentage + '%</span>';
                        htmlRow += '  </div>';
                        htmlRow += '  <div class="w-full bg-slate-700 rounded-full h-1.5 overflow-hidden">';
                        htmlRow += '    <div class="bg-gradient-to-r from-indigo-500 to-cyan-400 h-1.5 rounded-full" style="width: ' + percentage + '%"></div>';
                        htmlRow += '  </div>';
                        htmlRow += '</div>';
                        
                        probContainer.innerHTML += htmlRow;
                    });
                } else {
                    placeholder.classList.remove('hidden');
                    alert('Prediction Refused: ' + result.error);
                }
            } catch (error) {
                loader.classList.add('hidden');
                placeholder.classList.remove('hidden');
                alert('Connection to server failed.');
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
        return jsonify({"success": False, "error": "Model object profile not loaded on server side."}), 500
    
    try:
        data = request.json
        
        # Read city text and map to numerical value dynamically
        raw_city = str(data.get("city", "Other")).strip().lower()
        encoded_city = abs(hash(raw_city)) % 5
        
        encoded_gender = GENDER_MAP.get(data.get("gender"), 0)
        encoded_subscription = SUBSCRIPTION_MAP.get(data.get("subscription_type"), 0)
        
        # Feature array matching model layout precisely
        features = [
            float(data.get("age", 0)),
            float(encoded_gender),
            float(encoded_city),
            float(data.get("tenure_months", 0)),
            float(data.get("avg_order_value", 0)),
            float(data.get("total_orders", 0)),
            float(data.get("last_purchase_days_ago", 0)),
            float(data.get("support_tickets", 0)),
            float(encoded_subscription)
        ]
        
        input_matrix = np.array([features])
        
        prediction = int(model.predict(input_matrix)[0])
        probabilities = model.predict_proba(input_matrix)[0].tolist()
        
        return jsonify({
            "success": True,
            "prediction": prediction,
            "probabilities": probabilities
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400

if __name__ == "__main__":
    # Ensure standard binding to environment variable port for Render
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
