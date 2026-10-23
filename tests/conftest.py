"""Test-suite configuration that keeps plotting headless without environment setup."""

import matplotlib

matplotlib.use("Agg", force=True)
