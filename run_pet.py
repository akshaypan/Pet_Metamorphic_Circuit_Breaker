"""
run_pet.py — CLI Runner to evaluate any pet image with ResNet-18 and Metamorphic Circuit Breaker.

Usage:
    python run_pet.py
    python run_pet.py data/sample_dog.jpg
    python run_pet.py path/to/any_image.jpg --save_gradcam
"""

import argparse
import sys
from pathlib import Path
from PIL import Image

sys.path.insert(0, ".")

from src.models.resnet_adapter import ResNet18OxfordPetAdapter
from src.circuit_breaker import MetamorphicCircuitBreaker
from src.gradcam import make_comparison_figure


def main():
    parser = argparse.ArgumentParser(description="Run Metamorphic Circuit Breaker on a Pet image.")
    parser.add_argument(
        "image",
        nargs="?",
        default="data/sample_dog.jpg",
        help="Path to the image to evaluate (default: data/sample_dog.jpg)",
    )
    parser.add_argument(
        "--save_gradcam",
        action="store_true",
        default=True,
        help="Save Grad-CAM attention comparison figures if flips are detected or for baseline.",
    )
    args = parser.parse_args()

    img_path = Path(args.image)
    if not img_path.exists():
        print(f"[ERROR] Image not found: {img_path}")
        sys.exit(1)

    print(f"\nEvaluating: {img_path}")
    image = Image.open(img_path).convert("RGB")

    print("Loading ResNet-18 Oxford Pet Model...")
    adapter = ResNet18OxfordPetAdapter()
    cb = MetamorphicCircuitBreaker(adapter)

    print("Running 22 Metamorphic Tests across Geometric, Photometric, and Sensor/Noise families...")
    report, gcam_data = cb.evaluate_with_gradcam(image, max_gradcam_flips=3)

    flips = sum(1 for t in report.test_results if t.flipped)

    print("\n" + "=" * 60)
    print(f"DIAGNOSTIC REPORT")
    print("=" * 60)
    print(f"Predicted Breed      : {report.baseline_label}")
    print(f"Confidence (c0)      : {report.baseline_confidence:.2%}")
    print(f"Metamorphic Flips    : {flips} / {len(report.test_results)}")
    print(f"Peak Family (max IR) : {report.peak_family} ({report.peak_family_score:.4f})")
    print(f"Cross-Family Spread  : {report.cross_family_spread:.4f}")
    print(f"Composite Risk (CBI) : {report.cbi:.4f}")
    print("-" * 60)
    print(f"BREAKER STATE        : [{report.state.value}]")
    print(f"ACTION               : {report.action}")
    print("=" * 60)

    # ── Fallback Mechanism: Route to Preferred Zoologist if Unreliable ─────
    from src.zoologist_fallback import ZoologistFallbackService, DEFAULT_ZOOLOGISTS

    fallback_service = ZoologistFallbackService()
    draft_ticket = fallback_service.prepare_referral(
        report=report,
        query_image=image,
        preferred_zoologist=DEFAULT_ZOOLOGISTS[0],
        gradcam_data=gcam_data,
    )

    if draft_ticket:
        z = draft_ticket.assigned_zoologist
        print("\n" + "!" * 60)
        print("🚨 FALLBACK PROTOCOL TRIGGERED: UNRELIABLE PREDICTION DETECTED")
        print("!" * 60)
        print(f"Urgency Status       : {draft_ticket.urgency_level}")
        print(f"Recommended Action   : {draft_ticket.recommended_action}")
        print(f"Preferred Zoologist  : {z['name']} ({z['title']})")
        print(f"Institution          : {z['institution']}")
        print(f"Direct Contact       : {z['email']} | {z['phone']}")
        print(f"Specialization       : {z['specialization']}")
        print(f"Staged Ticket ID     : {draft_ticket.ticket_id}")
        print("!" * 60)

        # Ask the user explicitly for confirmation
        user_choice = input(f"\n[?] Would you like to dispatch this diagnostic dossier and specimen data to {z['name']}? (y/N): ").strip().lower()
        if user_choice in ("y", "yes"):
            confirmed_ticket = fallback_service.confirm_and_dispatch(draft_ticket)
            print(f"\n✔ [CONFIRMED & SENT] {confirmed_ticket.dispatch_status}")
            print(f"✔ Referral Dossier saved: {confirmed_ticket.dossier_filepath}")
        else:
            print(f"\n✖ [CANCELLED BY USER] Data was NOT sent to the zoologist.")
            print("  Automated prediction remains unapproved; diagnostic details kept local.")
    else:
        print("\n[Fallback Status] Prediction is verified reliable (CLOSED). Zoologist escalation not required.")

    # Save Grad-CAM comparison figure if requested
    out_dir = Path("outputs") / "pet_evaluations"
    out_dir.mkdir(parents=True, exist_ok=True)

    if gcam_data["flip_cams"]:
        print(f"\n[Grad-CAM] Saving {len(gcam_data['flip_cams'])} flip comparison figure(s)...")
        for i, fc in enumerate(gcam_data["flip_cams"]):
            fig = make_comparison_figure(
                original_image=image,
                baseline_cam=gcam_data["baseline_cam"],
                transformed_image=fc["transformed_image"],
                flip_cam=fc["cam"],
                baseline_label=report.baseline_label,
                baseline_conf=report.baseline_confidence,
                flipped_label=fc["flipped_label"],
                flipped_conf=fc["flipped_confidence"],
                test_name=f"{fc['test_name']} ({fc['severity']})",
            )
            out_file = out_dir / f"{img_path.stem}_flip_{i+1}_{fc['test_id']}.png"
            fig.save(out_file)
            print(f"  -> Saved: {out_file}")
    else:
        print("\n[Grad-CAM] Zero prediction flips. Prediction is invariant across metamorphic tests.")


if __name__ == "__main__":
    main()

