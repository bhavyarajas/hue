import numpy as np


class PatternModel:

    def __init__(self, latent_dim=8, random_seed=None):
        self.latent_dim = latent_dim
        self.rng = np.random.default_rng(random_seed)

        # Learned parameters (populated after fit() or load())
        self.mean_ = None           # shape: (D,)
        self.components_ = None     # shape: (D, K)
        self.latent_samples_ = None # shape: (N, K)
        self.canonical_shape = None # (H, W)
        self.is_trained = False

    # ---------- training ----------

    def fit(self, patterns: np.ndarray):
        if patterns.ndim != 3:
            raise ValueError("patterns must have shape (N, H, W)")

        N, H, W = patterns.shape
        self.canonical_shape = (H, W)

        # Flatten to (N, D)
        X = patterns.reshape(N, -1).astype(float)
        D = X.shape[1]

        # Compute mean and center data
        mean = X.mean(axis=0)
        X_centered = X - mean

        # SVD for PCA
        # X_centered = U S V^T  => rows of V^T are principal directions
        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)

        K = min(self.latent_dim, Vt.shape[0])
        components = Vt[:K].T  # (D, K)

        # Encode training data: Z = X_centered * components
        Z = X_centered @ components  # (N, K)

        self.mean_ = mean
        self.components_ = components
        self.latent_samples_ = Z
        self.is_trained = True

    # ---------- saving / loading ----------

    def save(self, path: str):
        """
        Save trained parameters to a .npz file.
        """
        if not self.is_trained:
            raise RuntimeError("Cannot save: model has not been trained (fit) yet.")

        H, W = self.canonical_shape
        np.savez(
            path,
            mean=self.mean_,
            components=self.components_,
            latent_samples=self.latent_samples_,
            canonical_height=np.array([H]),
            canonical_width=np.array([W]),
        )

    def load(self, path: str):
        """
        Load trained parameters from a .npz file created by save().
        """
        data = np.load(path)
        self.mean_ = data["mean"]
        self.components_ = data["components"]
        self.latent_samples_ = data["latent_samples"]
        H = int(data["canonical_height"][0])
        W = int(data["canonical_width"][0])
        self.canonical_shape = (H, W)
        self.is_trained = True

    # ---------- generation ----------

    def _decode_latent(self, Z: np.ndarray) -> np.ndarray:
        """
        Decode latent vectors Z back into pattern space.

        Z : (n_samples, K)
        returns X_recon : (n_samples, D)
        """
        X_centered = Z @ self.components_.T  # (n, D)
        X_recon = X_centered + self.mean_
        return X_recon

    def _sample_latents(self, n: int, noise_scale: float) -> np.ndarray:
        """
        Sample n latent vectors near the training latents.

        - Randomly pick training latents.
        - Add Gaussian noise with std = noise_scale.
        """
        if self.latent_samples_ is None:
            raise RuntimeError("Model has no latent_samples_; did you fit or load it?")

        N_train, K = self.latent_samples_.shape
        indices = self.rng.integers(0, N_train, size=n)
        base = self.latent_samples_[indices]
        noise = self.rng.normal(scale=noise_scale, size=base.shape)
        return base + noise

    def _resize_binary_pattern(self, pattern: np.ndarray, out_shape):
        """
        Resize a 2D binary pattern from canonical_shape to out_shape using
        nearest-neighbor sampling. This avoids any hand-coded motif logic;
        it just resamples the learned pattern.

        pattern : (H, W)
        out_shape : (rows, cols)
        """
        H, W = pattern.shape
        rows, cols = out_shape

        if (H, W) == (rows, cols):
            return pattern

        # Nearest-neighbor sampling
        ys = np.linspace(0, H - 1, rows)
        xs = np.linspace(0, W - 1, cols)
        ys_idx = np.round(ys).astype(int)
        xs_idx = np.round(xs).astype(int)

        ys_idx = np.clip(ys_idx, 0, H - 1)
        xs_idx = np.clip(xs_idx, 0, W - 1)

        resized = pattern[np.ix_(ys_idx, xs_idx)]
        return resized

    def generate(self, n: int = 1, out_shape=None, noise_scale: float = 0.15) -> np.ndarray:
        """
        Generate n new binary patterns.

        Parameters
        ----------
        n : int
            Number of patterns to generate.
        out_shape : tuple or None
            If None, return patterns in the canonical training shape (H, W).
            If (rows, cols), resize each generated pattern to this shape.
        noise_scale : float
            Amount of noise to add in latent space. Larger => more variation,
            possibly weirder patterns.

        Returns
        -------
        patterns : np.ndarray
            Binary array of shape (n, rows, cols) with values {0, 1}.
        """
        if not self.is_trained:
            raise RuntimeError("Model has not been trained (fit) or loaded.")

        H, W = self.canonical_shape

        # 1) Sample latents near training patterns
        Z = self._sample_latents(n, noise_scale=noise_scale)  # (n, K)

        # 2) Decode to pattern space and reshape
        X_recon = self._decode_latent(Z)  # (n, D)
        X_recon = X_recon.reshape(n, H, W)

        # 3) Clip to [0,1] and interpret as probabilities, then sample Bernoulli
        X_prob = np.clip(X_recon, 0.0, 1.0)
        rand = self.rng.random(X_prob.shape)
        binary = (rand < X_prob).astype(int)  # (n, H, W)

        # 4) Optional resize
        if out_shape is None or out_shape == self.canonical_shape:
            return binary

        resized = np.zeros((n, out_shape[0], out_shape[1]), dtype=int)
        for i in range(n):
            resized[i] = self._resize_binary_pattern(binary[i], out_shape)
        return resized


class VAEModel:
    """
    Wrapper that preserves your old interface:

        vae = VAEModel(weights_path="pattern_model.npz")
        grid = vae.generate_grid(level="easy", shape=(3, 5))

    - Internally uses PatternModel (PCA-based generative model).
    - Returns a 2D grid of ints in [0, 255]:
        0   -> dot tile (fixed / black circle)
        255 -> normal tile
    """

    def __init__(self, input_shape=(8, 8, 1), latent_dim=8, weights_path=None, random_seed=None):
        H, W, _ = input_shape
        self.input_shape = input_shape
        self.model = PatternModel(latent_dim=latent_dim, random_seed=random_seed)
        self.rng = np.random.default_rng(random_seed)

        if weights_path is not None:
            self.model.load(weights_path)

    def generate_grid(self, level="easy", shape=(3, 5)):
        noise_map = {
            "easy": 0.10,
            "medium": 0.18,
            "hard": 0.25,
        }
        noise_scale = noise_map.get(level, 0.15)

        rows, cols = shape
        max_tries = 30  # plenty of tries, model is fast

        for _ in range(max_tries):
            # --- 1) generate pattern ---
            pattern = self.model.generate(
                n=1,
                out_shape=shape,
                noise_scale=noise_scale,
            )[0]  # (rows, cols), values in {0,1}

            # --- 2) convert to game grid ---
            # pattern == 1 -> dot -> 0
            # pattern == 0 -> normal -> 255
            grid = (1 - pattern) * 255

            # --- 3) check for at least one dot ---
            if (grid == 0).any():
                return grid.astype(int)

        # If we somehow failed 30 times, RESTART generation loop from scratch
        # (not fallback, still retry logic)
        return self.generate_grid(level=level, shape=shape)