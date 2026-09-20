"""Tests for the building classifier."""

import torch

import pytest


def test_classifier_forward():
    pytest.importorskip("torchvision")
    from perception.classifier import BuildingClassifier

    model = BuildingClassifier(num_types=5, max_level=10)
    x = torch.randn(2, 3, 224, 224)
    type_logits, level = model(x)
    assert type_logits.shape == (2, 5)
    assert level.shape == (2,)


def test_classifier_save_and_load(tmp_path):
    pytest.importorskip("torchvision")
    from perception.classifier import BuildingClassifier

    model = BuildingClassifier(num_types=3, max_level=5)
    path = tmp_path / "best.pt"
    model.save(path, metadata={"foo": "bar"})
    loaded = BuildingClassifier.load(path)
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        out1 = model(x)
        out2 = loaded(x)
    assert torch.allclose(out1[0], out2[0], atol=1e-6)
    assert torch.allclose(out1[1], out2[1], atol=1e-6)
