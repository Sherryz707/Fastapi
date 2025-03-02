import pathlib
import torch
from fastai.learner import load_learner

# Force pathlib to use Windows paths temporarily
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

# Load the model
learn = load_learner("model.pkl")

# Restore pathlib to its original state (IMPORTANT)
pathlib.PosixPath = temp

# Re-save the model to fix pathing issues
learn.export("model_fixed.pkl")

print("Model saved as model_fixed.pkl")
