"""Training-time visual augmentation aligned to the LIBERO-Plus eval families.

Train-time augmentation and eval-time perturbation share a *family* but live on *separate
splits* (honesty guard: in-dist recovery is never presented as generalization).
"""
