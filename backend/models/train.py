from vae_copy import PatternModel   # or from vae import PatternModel if that's where it lives
from trainingdata import patterns
import numpy as np


def main():
    # patterns comes from trainingdata.py
    # shape should be (N, 8, 8), values 0/1
    print("Loaded patterns from trainingdata.py with shape:", patterns.shape)

    # 1) Save patterns as a clean .npy file (optional but nice to have)
    np.save("trainingdata.npy", patterns.astype(np.int8))
    print("Saved trainingdata.npy in the current folder.")

    # 2) Train the PatternModel on these patterns
    model = PatternModel(latent_dim=8)
    model.fit(patterns.astype(float))
    print("Finished training PatternModel.")

    # 3) Save trained model weights to pattern_model.npz
    model.save("pattern_model.npz")
    print("Saved trained model to pattern_model.npz")


if __name__ == "__main__":
    main()