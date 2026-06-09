# 🧠 Predictive Customer Analytics Dashboard

An elegant, production-ready machine learning web application that exposes a trained **Naive Bayes (`GaussianNB`)** model via a responsive Flask API. Built with a modern glassmorphic UI using Tailwind CSS, this dashboard allows real-time customer behavioral classification and probability tracking.

## 🔗 Live Demo

🚀 **Experience the app live here:** [Customer churn prediction](https://customer-churn-prediction-xpj1.onrender.com)

---

## ✨ Features

- **🎯 Live Model Predictions:** Instantly computes behavioral classifications directly from your serialized Naive Bayes pipeline.
- **📊 Real-time Confidence Tracking:** Dynamically outputs percentage-based probability metrics with animated visual bars for all possible target classes.
- **🎨 Modern Glassmorphic UI:** A beautiful dashboard utilizing Tailwind CSS, interactive states, embedded loading animations, and split-pane layout geometry.
- **⚡ Dynamic Text Feature Hashing:** Features like "City" allow open text input from users, which is dynamically converted into stable numerical inputs matching the model’s expected matrix configuration.
- **🚀 Cloud Native Deployment:** Configured out of the box to deploy seamlessly to web services like **Render**.

---

## 🛠️ Architecture & Tech Stack

- **Backend Logic:** Python 3, Flask
- **Machine Learning Core:** Scikit-Learn (`sklearn`), NumPy
- **Production Web Server:** Gunicorn
- **Frontend Dashboard:** HTML5, JavaScript (Fetch API), Tailwind CSS, Plus Jakarta Sans Typography

---

## 📂 Project Structure

To deploy seamlessly, ensure your repository keeps this exact structure:

```text
├── app.py                   # Main Flask application with embedded UI templates
├── requirements.txt         # Production library dependencies
└── Naive_Bayes_Model.pkl    # Your serialized Scikit-Learn model binary
```

---

## 🧬 Feature Matrix Map

The application automatically reads inputs and maps them precisely to match the underlying 9-dimensional array topology your model requires:

| # | Input Field | Data Type | Backend Process |
|---|-------------|-----------|-----------------|
| 1 | age | Numeric | Passed as Float |
| 2 | gender | Categorical Dropdown | Decoded mapped integer (0, 1, 2) |
| 3 | city | Freeform Text Input | Dynamically hashed to integer index [0-4] |
| 4 | tenure_months | Numeric | Passed as Float |
| 5 | avg_order_value | Decimal Currency | Passed as Float |
| 6 | total_orders | Numeric | Passed as Float |
| 7 | last_purchase_days_ago | Numeric | Passed as Float |
| 8 | support_tickets | Numeric | Passed as Float |
| 9 | subscription_type | Categorical Dropdown | Decoded mapped integer (0, 1, 2) |

---

## 🚀 Local Installation & Setup

### Clone the Repository

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

### Establish Environment & Dependencies

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
pip install -r requirements.txt
```

### Verify Model Placement

Ensure your generated `Naive_Bayes_Model.pkl` binary is sitting in the root project folder.

### Boot Up the Local Server

```bash
python app.py
```

Open your browser and navigate to `http://localhost:5000` to interact with your dashboard.

---

## 🌐 Deploying to Render (Step-by-Step)

This application is fully optimized for a single-click deployment pipeline on Render:

1. Push your code repository (`app.py`, `requirements.txt`, and your model file) onto GitHub.
2. Log into the Render Dashboard.
3. Click **New +** and choose **Web Service**.
4. Link your connected GitHub repository.
5. In the configuration settings page, fill out these fields:

| Setting | Value |
|----------|--------|
| Runtime | Python 3 |
| Build Command | `pip install -r requirements.txt` |
| Start Command | `gunicorn app:app` |

6. Click **Deploy Web Service** at the bottom of the page.

Render will provision your virtual machine, install your Scikit-Learn dependencies, attach to the dynamic environment port, and output a live public URL for your dashboard!

---

## 📈 Prediction Workflow

1. User enters customer profile information into the dashboard form.
2. Frontend validates and submits the request using the JavaScript Fetch API.
3. Flask receives the request and transforms all incoming values into the expected feature matrix format.
4. Text-based fields such as **City** are dynamically hashed into numerical representations.
5. The serialized **GaussianNB** model is loaded and used for inference.
6. Prediction probabilities are calculated using the model's `predict_proba()` method.
7. Results are returned as JSON to the frontend.
8. The dashboard renders:
   - Predicted customer class
   - Confidence percentage
   - Animated probability bars
   - Real-time classification insights

---

## 🔒 Model Serialization

The machine learning model is stored as:

```text
Naive_Bayes_Model.pkl
```

The application loads the model during startup and uses it for low-latency real-time predictions.

Example loading logic:

```python
import pickle

with open("Naive_Bayes_Model.pkl", "rb") as file:
    model = pickle.load(file)
```

---

## ⚙️ API Endpoint

### Predict Customer Behavior

**Endpoint**

```http
POST /predict
```

**Request Body Example**

```json
{
  "age": 32,
  "gender": 1,
  "city": "Pune",
  "tenure_months": 18,
  "avg_order_value": 2500,
  "total_orders": 40,
  "last_purchase_days_ago": 12,
  "support_tickets": 2,
  "subscription_type": 1
}
```

**Response Example**

```json
{
  "prediction": "Loyal Customer",
  "confidence": 92.47,
  "probabilities": {
    "Loyal Customer": 92.47,
    "At Risk": 5.12,
    "Churn Likely": 2.41
  }
}
```

---

## 🎨 User Interface Highlights

- Glassmorphism design language
- Responsive mobile-first layout
- Split-pane dashboard architecture
- Animated loading states
- Interactive prediction feedback
- Dynamic confidence visualization
- Tailwind CSS utility-based styling
- Modern typography using Plus Jakarta Sans

---

## 📦 Production Dependencies

Example `requirements.txt`:

```text
Flask
gunicorn
numpy
scikit-learn
joblib
```

---

## 🚀 Performance Characteristics

- Fast inference using Gaussian Naive Bayes
- Lightweight deployment footprint
- Low memory consumption
- Cloud-ready architecture
- Real-time prediction latency
- Easy horizontal scalability

---

## 📝 Future Enhancements

- Authentication & user accounts
- Prediction history tracking
- Database integration
- Advanced customer segmentation
- Model monitoring dashboard
- Automated retraining pipelines
- Explainable AI (SHAP/LIME)
- Batch prediction uploads
- Power BI integration
- Real-time streaming analytics

---

## 📄 License

This project is intended for educational, demonstration, and portfolio purposes. Feel free to modify and extend the application for your own machine learning deployment workflows.

---

## 👨‍💻 Author
- **Yogita Dokh**
- **dokhyogita20@gmail.com**

---

⭐ If you found this project useful, consider giving it a star on GitHub and sharing it with others interested in machine learning deployment and customer analytics.
