numpy
pandas
scikit-learn
matplotlib
rasterio
joblib

Step 1: Open Anaconda Prompt
Step 2: Update conda
	conda update -n base -c defaults conda 
	(Type y and press Enter)
Step 3: Create a fresh environment (VERY IMPORTANT)
	conda create -n geo_env python=3.9
	Activate it:conda activate geo_env
Step 4: Install rasterio ONLY
	conda install -c conda-forge rasterio
	pip install numpy pandas scikit-learn matplotlib joblib

Train first → Predict later
	python train_model.py
	python predict.py
	python app.py