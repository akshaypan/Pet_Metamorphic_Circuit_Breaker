"""
scripts/verify_resnet.py — Full verification of ResNet-18 Oxford-IIIT Pet Adapter.

Validates:
1. Checkpoint loading and class alignment (37 Oxford-IIIT Pet breeds).
2. Forward inference (predict) with probability distributions.
3. Grad-CAM attention heatmap generation and retain_grad safety under no_grad().
4. End-to-end Metamorphic Circuit Breaker hierarchical evaluation (22 tests, Level 1 & 2 normalization).
5. evaluate_with_gradcam and multi-panel comparison figure synthesis.

Run:
    python -m scripts.verify_resnet
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from PIL import Image
import numpy as np

# Ensure project root is in path
sys.path.insert(0, ".")

from src.utils import OXFORD_PET_CLASSES, OXFORD_PET_NUM_CLASSES, RESNET18_MODEL_PATH
from src.models.resnet_adapter import ResNet18OxfordPetAdapter
from src.circuit_breaker import MetamorphicCircuitBreaker, CircuitBreakerConfig
from src.gradcam import make_comparison_figure


def main():
    print("=" * 65)
    print("STEP 1: ResNet-18 Checkpoint Loading & Architecture Check")
    print("=" * 65)
    print(f"  Checkpoint path: {RESNET18_MODEL_PATH}")
    assert RESNET18_MODEL_PATH.exists(), f"Checkpoint not found at {RESNET18_MODEL_PATH}"
    print(f"  Total Oxford-IIIT Pet classes: {OXFORD_PET_NUM_CLASSES}")
    print(f"  First 5 classes: {OXFORD_PET_CLASSES[:5]}")
    print(f"  Last 5 classes:  {OXFORD_PET_CLASSES[-5:]}")

    adapter = ResNet18OxfordPetAdapter()
    print("  [OK] Model successfully loaded and set to eval mode")

    print("\n" + "=" * 65)
    print("STEP 2: Forward Inference (predict) Verification")
    print("=" * 65)
    sample_dog_path = Path("data/sample_dog.jpg")
    if sample_dog_path.exists():
        test_img = Image.open(sample_dog_path).convert("RGB")
        print(f"  Testing on real image: {sample_dog_path}")
    else:
        test_img = Image.new("RGB", (224, 224), color=(200, 180, 150))
        print("  Testing on synthetic image")

    pred_idx, label, conf, probs = adapter.predict(test_img)
    print(f"  Predicted Index : {pred_idx}")
    print(f"  Predicted Breed : {label}")
    print(f"  Confidence      : {conf:.2%}")
    print(f"  Probabilities   : len={len(probs)}, sum={sum(probs):.4f}")

    assert 0 <= pred_idx < OXFORD_PET_NUM_CLASSES, "pred_idx out of range"
    assert label == OXFORD_PET_CLASSES[pred_idx], "Label mismatch"
    assert 0.0 <= conf <= 1.0, "Confidence out of range"
    assert len(probs) == OXFORD_PET_NUM_CLASSES, "Probability length mismatch"
    assert abs(sum(probs) - 1.0) < 1e-4, "Probabilities do not sum to 1"
    print("  [OK] Forward prediction verified")

    print("\n" + "=" * 65)
    print("STEP 3: ResNet Grad-CAM & retain_grad Safety Verification")
    print("=" * 65)
    # Generate Grad-CAM
    cam = adapter.explain(test_img, target_class=pred_idx)
    assert cam is not None, "Grad-CAM heatmap is None"
    assert cam.ndim == 2, f"Grad-CAM shape is {cam.shape}, expected 2D"
    print(f"  Grad-CAM heatmap shape : {cam.shape}")
    print(f"  Heatmap range          : [{cam.min():.4f}, {cam.max():.4f}]")
    assert 0.0 <= cam.min() <= cam.max() <= 1.0001, "Grad-CAM not normalized to [0, 1]"

    # Test that calling predict under no_grad after explain does NOT crash
    try:
        idx2, lbl2, conf2, _ = adapter.predict(test_img)
        print(f"  predict() after explain(): OK ({lbl2} {conf2:.2%})")
    except RuntimeError as e:
        print(f"  FAIL: retain_grad error: {e}")
        sys.exit(1)
    print("  [OK] Grad-CAM and retain_grad safety verified")

    print("\n" + "=" * 65)
    print("STEP 4: Metamorphic Circuit Breaker Full Pipeline")
    print("=" * 65)
    cb = MetamorphicCircuitBreaker(adapter)
    t0 = time.time()
    report = cb.evaluate(test_img)
    elapsed = time.time() - t0

    flips = sum(1 for t in report.test_results if t.flipped)
    print(f"  Baseline Prediction  : {report.baseline_label} ({report.baseline_confidence:.2%})")
    print(f"  Metamorphic Flips    : {flips} / {len(report.test_results)}")
    print(f"  Type Scores (L1)     : {report.type_scores}")
    print(f"  Family IR_k (L2)     : {report.family_instability}")
    print(f"  Peak Family          : {report.peak_family} ({report.peak_family_score:.4f})")
    print(f"  Cross-Family Spread  : {report.cross_family_spread:.4f}")
    print(f"  CBI Risk Score       : {report.cbi:.4f}")
    print(f"  Operational State    : {report.state.value}")
    print(f"  Action               : {report.action}")
    print(f"  Inference Time (23x) : {elapsed:.2f}s")

    # Mathematical consistency checks
    a = report.details["config"]["alpha"]
    b = report.details["config"]["beta"]
    g = report.details["config"]["gamma"]
    expected_cbi = a * report.peak_family_score + b * report.cross_family_spread + g * (1.0 - report.baseline_confidence)
    expected_cbi = max(0.0, min(1.0, round(expected_cbi, 4)))
    assert abs(report.cbi - expected_cbi) < 1e-4, f"CBI math mismatch: {report.cbi} vs {expected_cbi}"
    print("  [OK] Hierarchical mathematical attribution verified")

    print("\n" + "=" * 65)
    print("STEP 5: evaluate_with_gradcam & Comparison Figure")
    print("=" * 65)
    t0 = time.time()
    rep_gcam, gcam_data = cb.evaluate_with_gradcam(test_img, max_gradcam_flips=2)
    elapsed_gcam = time.time() - t0
    print(f"  Baseline CAM present : {gcam_data['baseline_cam'] is not None}")
    print(f"  Flip CAMs generated  : {len(gcam_data['flip_cams'])}")
    print(f"  Elapsed              : {elapsed_gcam:.2f}s")

    # Generate test comparison figure
    fake_transformed = test_img.rotate(15)
    fig_img = make_comparison_figure(
        original_image=test_img,
        baseline_cam=cam,
        transformed_image=fake_transformed,
        flip_cam=cam,
        baseline_label=report.baseline_label,
        baseline_conf=report.baseline_confidence,
        flipped_label=report.baseline_label,
        flipped_conf=report.baseline_confidence,
        test_name="Rotation (15 deg)",
        out_size=224,
    )
    assert fig_img.size[0] > 300, "Figure size too small"
    print(f"  Synthesized Figure   : {fig_img.size}")
    print("  [OK] evaluate_with_gradcam and figure synthesis verified")

    print("\n" + "=" * 65)
    print("ALL RESNET-18 OXFORD PET VERIFICATION CHECKS PASSED!")
    print("=" * 65)


if __name__ == "__main__":
    main()
