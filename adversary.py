# adversary.py
import numpy as np
import torch
from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import ProjectedGradientDescent

def build_art_classifier(model, loss_fn, input_shape=(3,112,112), nb_classes=100):
    model.eval()
    # If model returns embeddings rather than logits, ART still expects a classifier interface.
    # We keep optimizer=None and let ART use loss / model passed.
    classifier = PyTorchClassifier(
        model=model,
        loss=loss_fn,
        input_shape=input_shape,
        nb_classes=nb_classes,
        clip_values=(0.0, 1.0),
        optimizer=None
    )
    return classifier

def pgd_attack(classifier, x, eps=0.03, eps_step=0.007, max_iter=10):
    # x is a torch tensor with shape (B, C, H, W) and values in [0,1]
    x_np = x.cpu().numpy()
    attack = ProjectedGradientDescent(
        estimator=classifier,
        norm=np.inf,
        eps=eps,
        eps_step=eps_step,
        max_iter=max_iter
    )
    adv = attack.generate(x=x_np)
    return torch.from_numpy(adv).type_as(x)
