"""Thread P: pixel world models on a rented GPU. See docs/GPU_RUN.md and docs/HYPOTHESES.md.

This file imports nothing, so the scheduler and config stay importable before torch is installed.

Where each thing runs, and why:
    GPU   rendering, simulation during data collection and closed loop, JEPA encoder training, encoding, rollouts, CEM.
    CPU   the dynamics core (intr.train), adaptation (intr.adapt, intr.ebayes), the error-fed core. These are the existing,
          tested code paths, unchanged, on small MLPs with small batches, where a GPU step is no faster than a CPU one.
"""
