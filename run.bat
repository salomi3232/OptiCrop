@echo off
echo ============================================
echo   OptiCrop - Setup and Run
echo ============================================

echo [1/4] Installing dependencies...
pip install -r requirements.txt

echo [2/4] Generating dataset...
python generate_dataset.py

echo [3/4] Training ML model...
python train_model.py

echo [4/4] Starting Flask app...
echo Open http://127.0.0.1:5000 in your browser
python app.py
