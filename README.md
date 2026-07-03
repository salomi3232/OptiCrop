# 🌾 OptiCrop – Smart Agricultural Production Optimization Engine

An AI-powered web application that recommends the best crops based on soil and environmental parameters using Machine Learning.

---

## 📁 Project Structure

```
OptiCrop/
├── app.py                  # Flask web application
├── train_model.py          # ML model training + EDA plots
├── generate_dataset.py     # Dataset generation script
├── requirements.txt        # Python dependencies
├── run.bat                 # One-click setup & run (Windows)
├── Crop_recommendation.csv # Generated dataset (after running generate_dataset.py)
├── model/
│   ├── crop_model.pkl      # Trained Random Forest model
│   ├── scaler.pkl          # StandardScaler
│   ├── label_encoder.pkl   # LabelEncoder
│   └── crop_stats.csv      # Crop statistics for suitability check
├── templates/
│   ├── base.html           # Base layout
│   ├── index.html          # Home page
│   ├── recommend.html      # Scenario 1 – Crop Recommendation
│   ├── suitability.html    # Scenario 2 – Suitability Assessment
│   └── research.html       # Scenario 3 – Research & Analytics
├── static/
│   ├── css/style.css       # Stylesheet
│   ├── js/main.js          # JavaScript
│   └── plots/              # EDA plots (generated after training)
└── notebooks/
    └── OptiCrop_EDA.ipynb  # Jupyter Notebook for EDA
```

---

## 🚀 Quick Start

### Option 1 – One Click (Windows)
```
Double-click run.bat
```

### Option 2 – Manual Steps
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate dataset
python generate_dataset.py

# 3. Train the model (generates plots too)
python train_model.py

# 4. Run the Flask app
python app.py
```

Open **http://127.0.0.1:5000** in your browser.

---

## 🌐 Application Scenarios

| Scenario | URL | Description |
|----------|-----|-------------|
| Crop Recommendation | `/recommend` | Enter N, P, K, temp, humidity, pH, rainfall → get best crop |
| Suitability Check   | `/suitability` | Check if conditions suit a specific crop with a score |
| Research Analytics  | `/research` | View EDA plots and crop statistics table |
| REST API            | `POST /api/predict` | JSON API for programmatic access |

---

## 🔌 REST API

```bash
POST /api/predict
Content-Type: application/json

{
  "N": 90, "P": 42, "K": 43,
  "temperature": 25.5, "humidity": 80.5,
  "ph": 6.5, "rainfall": 200.9
}
```

Response:
```json
{
  "recommended_crop": "rice",
  "top5": [["rice", 92.0], ["jute", 4.0], ...]
}
```

---

## 🛠️ Technologies Used

| Category | Technology |
|----------|-----------|
| ML Model | Scikit-learn (Random Forest) |
| Data Processing | NumPy, Pandas |
| Visualization | Matplotlib, Seaborn |
| Statistical Analysis | SciPy |
| Web Framework | Flask |
| Frontend | HTML5, CSS3, JavaScript |

---

## 👥 Team

| Name | Role |
|------|------|
| Ganjamula Salomi | Team Lead |
| Gayathri Chenchalamoni | Member |
| Pushpalatha Chakali | Member |
| Chukka Rekha | Member |
| Vadla Deepika | Member |
