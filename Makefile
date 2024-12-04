# # Define paths to checkpoints and outputs from params.py
# PREPROCESSED_DATA = data/processed/X_train.npy data/processed/y_train.npy
# MODEL_CHECKPOINT = models/trained_model.keras
# VISUALIZATION_DIR = logs/visualizations

# # Default target
# .PHONY: all
# all: preprocess train visualize

# # Preprocess step
# .PHONY: preprocess
# preprocess:
# 	@echo "Preprocessing data..."
# 	python scripts/run_preprocess.py
# 	@echo "Data preprocessing completed."

# # Training step
# .PHONY: train
# train: $(MODEL_CHECKPOINT)

# $(MODEL_CHECKPOINT): $(PREPROCESSED_DATA)
# 	@echo "Training the model..."
# 	python -c "from models.unet import build_unet; \
# 	from scripts.train import train_model; \
# 	from utils.params import MODEL_DIR, PROCESSED_DIR; \
# 	import numpy as np, os; \
# 	print('Loading preprocessed data...'); \
# 	X_train = np.load(os.path.join(PROCESSED_DIR, 'X_train.npy')); \
# 	y_train = np.load(os.path.join(PROCESSED_DIR, 'y_train.npy')); \
# 	print('Building and compiling the model...'); \
# 	model = build_unet(); \
# 	model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']); \
# 	print('Starting training...'); \
# 	history, X_val, y_val = train_model(model, X_train, y_train); \
# 	print('Saving model and validation data...'); \
# 	os.makedirs(MODEL_DIR, exist_ok=True); \
# 	model.save(os.path.join(MODEL_DIR, 'trained_model.keras')); \
# 	np.save(os.path.join(PROCESSED_DIR, 'X_val.npy'), X_val); \
# 	np.save(os.path.join(PROCESSED_DIR, 'y_val.npy'), y_val)"
# 	@echo "Model training completed."

# # Visualization step
# .PHONY: visualize
# visualize: $(VISUALIZATION_DIR)

# $(VISUALIZATION_DIR): $(MODEL_CHECKPOINT)
# 	@echo "Generating visualizations..."
# 	python -c "import tensorflow as tf, os; \
# 	from scripts.visualize import visualize_predictions; \
# 	from utils.params import VISUALIZATION_DIR, PROCESSED_DIR, MODEL_DIR; \
# 	import numpy as np; \
# 	model = tf.keras.models.load_model(os.path.join(MODEL_DIR, 'trained_model.keras')); \
# 	X_val = np.load(os.path.join(PROCESSED_DIR, 'X_val.npy')); \
# 	y_val = np.load(os.path.join(PROCESSED_DIR, 'y_val.npy')); \
# 	val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(8); \
# 	os.makedirs(VISUALIZATION_DIR, exist_ok=True); \
# 	visualize_predictions(model, val_dataset, save_dir=VISUALIZATION_DIR)"
# 	@echo "Visualizations saved to $(VISUALIZATION_DIR)."

# # Clean up generated files
# .PHONY: clean
# clean:
# 	rm -rf data/processed/*.npy models/*.keras logs/visualizations/*
# 	@echo "Cleaned up all generated files."

train:
	python models/build_unet.py
