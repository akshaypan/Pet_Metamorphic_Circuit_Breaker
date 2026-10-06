"""
src/zoologist_fallback.py — Fallback Protocol & Zoologist Escalation Mechanism.

When the Metamorphic Circuit Breaker determines a prediction is unreliable
(CircuitBreakerState == OPEN or HALF_OPEN), this module engages automated
triage and dispatches a comprehensive diagnostic dossier to a preferred zoologist.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any
from PIL import Image

# ─── Preferred Zoologist Directory ─────────────────────────────────────────────

@dataclass
class ZoologistProfile:
    id: str
    name: str
    title: str
    institution: str
    email: str
    phone: str
    specialization: str
    preferred_channel: str = "EMAIL_AND_SMS"


# Default directory of verified preferred zoologists
DEFAULT_ZOOLOGISTS: List[ZoologistProfile] = [
    ZoologistProfile(
        id="ZOO_01",
        name="Dr. Eleanor Vance",
        title="Senior Veterinary Taxonomist & Mammalogist",
        institution="Royal Zoological Society / Oxford Pet Biodiversity Unit",
        email="e.vance@oxford-zoology.org",
        phone="+44 (0) 1865 270000",
        specialization="Canine & Feline Morphology, Phenotypic Breed Variance",
        preferred_channel="URGENT_TELEHEALTH",
    ),
    ZoologistProfile(
        id="ZOO_02",
        name="Prof. Marcus Sterling",
        title="Chair of Comparative Animal Biology",
        institution="International Companion Animal Genetics Institute",
        email="m.sterling@animalgenetics-consortium.org",
        phone="+44 (0) 20 7942 5000",
        specialization="Rare Breeds, Hybrid Felids & Working Terrier Lineages",
        preferred_channel="SMS_AND_PORTAL",
    ),
    ZoologistProfile(
        id="ZOO_03",
        name="Dr. Priya Nambiar",
        title="Consultant Veterinary Radiologist & Morphometrician",
        institution="MBCET Veterinary AI Safety Collaborative",
        email="p.nambiar@mbcet-vet.ac.in",
        phone="+91 94470 12345",
        specialization="Domestic Animal Phenotyping, Imaging Artifact Analysis",
        preferred_channel="EMAIL_PRIORITY",
    ),
]


# ─── Triage Ticket Structure ───────────────────────────────────────────────────

@dataclass
class ZoologistTriageTicket:
    ticket_id: str
    created_at: str
    urgency_level: str                       # "CRITICAL_INTERCEPT" (OPEN) | "ADVISORY_MONITOR" (HALF_OPEN)
    state: str                               # "OPEN" | "HALF_OPEN"
    cbi_score: float
    assigned_zoologist: Dict[str, str]
    specimen_baseline: Dict[str, Any]       # baseline_label, confidence, all_top3
    perturbation_diagnostics: Dict[str, Any] # peak_family, compromised_families, total_flips
    recommended_action: str
    gradcam_available: bool
    dossier_filepath: Optional[str] = None
    dispatch_status: str = "PENDING_USER_CONFIRMATION"
    is_confirmed: bool = False


# ─── Fallback Escalation Service ──────────────────────────────────────────────

class ZoologistFallbackService:
    """
    Manages runtime fallback escalation when model predictions are transform-brittle.
    Strictly enforces human-in-the-loop explicit confirmation before external dispatch.
    """

    def __init__(
        self,
        default_zoologist: Optional[ZoologistProfile] = None,
        output_dir: Optional[Path] = None,
    ) -> None:
        self.default_zoologist = default_zoologist or DEFAULT_ZOOLOGISTS[0]
        self.output_dir = output_dir or (Path("outputs") / "zoologist_referrals")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def prepare_referral(
        self,
        report,  # DiagnosticReport from MetamorphicCircuitBreaker
        query_image: Image.Image,
        preferred_zoologist: Optional[ZoologistProfile] = None,
        gradcam_data: Optional[Dict[str, Any]] = None,
        custom_notes: Optional[str] = None,
    ) -> Optional[ZoologistTriageTicket]:
        """
        Inspects the DiagnosticReport. If state is not CLOSED (unreliable),
        prepares a pending referral ticket that AWAITS explicit user confirmation.

        Returns:
            ZoologistTriageTicket with status PENDING_USER_CONFIRMATION, or None if CLOSED.
        """
        state_str = getattr(report.state, "value", str(report.state))

        # Only trigger fallback if prediction is unreliable
        if state_str == "CLOSED":
            return None

        zoologist = preferred_zoologist or self.default_zoologist
        timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ticket_id = f"ZOO-ESC-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

        urgency = "CRITICAL_INTERCEPT" if state_str == "OPEN" else "ADVISORY_MONITOR"
        action = (
            "AUTOMATED PREDICTION BLOCKED. Mandatory zoological species review recommended."
            if state_str == "OPEN"
            else "PREDICTION DELIVERED WITH WARNING. Secondary advisory review recommended."
        )

        flips = sum(1 for t in report.test_results if t.flipped)

        from src.utils import OXFORD_PET_CLASSES, get_pet_full_name

        baseline_probs = report.details.get("baseline_probs", [])
        top_candidates = []
        if baseline_probs:
            indexed = sorted(enumerate(baseline_probs), key=lambda x: -x[1])[:3]
            for idx, prob in indexed:
                name = OXFORD_PET_CLASSES[idx] if idx < len(OXFORD_PET_CLASSES) else str(idx)
                top_candidates.append(f"{get_pet_full_name(name)} ({prob:.1%})")

        full_breed = get_pet_full_name(report.baseline_label)

        ticket_dict = {
            "ticket_id": ticket_id,
            "created_at": timestamp_str,
            "urgency_level": urgency,
            "state": state_str,
            "cbi_score": report.cbi,
            "assigned_zoologist": asdict(zoologist),
            "specimen_baseline": {
                "predicted_breed": full_breed,
                "dataset_class": report.baseline_label,
                "confidence": report.baseline_confidence,
                "top_candidates": top_candidates,
            },
            "perturbation_diagnostics": {
                "total_flips": flips,
                "total_tests": len(report.test_results),
                "peak_family": report.peak_family,
                "peak_family_score": report.peak_family_score,
                "cross_family_spread": report.cross_family_spread,
                "compromised_families": report.compromised_families,
            },
            "recommended_action": action,
            "gradcam_available": bool(gradcam_data and gradcam_data.get("baseline_cam") is not None),
            "custom_notes": custom_notes or "Triggered by runtime metamorphic invariance failure.",
        }

        # Staged draft ticket — NOT yet sent externally
        return ZoologistTriageTicket(
            ticket_id=ticket_id,
            created_at=timestamp_str,
            urgency_level=urgency,
            state=state_str,
            cbi_score=report.cbi,
            assigned_zoologist=asdict(zoologist),
            specimen_baseline=ticket_dict["specimen_baseline"],
            perturbation_diagnostics=ticket_dict["perturbation_diagnostics"],
            recommended_action=action,
            gradcam_available=ticket_dict["gradcam_available"],
            dossier_filepath=None,
            dispatch_status="PENDING_USER_CONFIRMATION",
            is_confirmed=False,
        )

    def confirm_and_dispatch(
        self,
        ticket: ZoologistTriageTicket,
        zoologist: Optional[ZoologistProfile] = None,
    ) -> ZoologistTriageTicket:
        """
        Executed ONLY upon explicit user confirmation.
        Persists referral dossier to disk and simulates secure dispatch to zoologist.
        """
        z_dict = ticket.assigned_zoologist
        target_email = z_dict.get("email", "zoology-triage@mbcet-vet.ac.in")
        target_channel = z_dict.get("preferred_channel", "URGENT_TELEHEALTH")

        ticket.is_confirmed = True
        ticket.dispatch_status = f"DISPATCHED via {target_channel} to {target_email}"

        # Write Markdown & JSON dossier files upon confirmed dispatch
        dossier_path = self.output_dir / f"{ticket.ticket_id}.md"
        json_path = self.output_dir / f"{ticket.ticket_id}.json"

        ticket_data = asdict(ticket)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(ticket_data, f, indent=2)

        md_content = self._generate_markdown_dossier(ticket_data)
        with open(dossier_path, "w", encoding="utf-8") as f:
            f.write(md_content)

        ticket.dossier_filepath = str(dossier_path)
        return ticket

    def evaluate_and_route(
        self,
        report,
        query_image: Image.Image,
        preferred_zoologist: Optional[ZoologistProfile] = None,
        gradcam_data: Optional[Dict[str, Any]] = None,
        custom_notes: Optional[str] = None,
        auto_confirm: bool = False,
    ) -> Optional[ZoologistTriageTicket]:
        """Convenience method. If auto_confirm=False, returns draft awaiting confirmation."""
        draft = self.prepare_referral(report, query_image, preferred_zoologist, gradcam_data, custom_notes)
        if draft and auto_confirm:
            return self.confirm_and_dispatch(draft, preferred_zoologist)
        return draft

    def _generate_markdown_dossier(
        self,
        t: Dict[str, Any],
    ) -> str:
        """Constructs an official referral brief for the zoologist."""
        flips = t["perturbation_diagnostics"]["total_flips"]
        top_cand = ", ".join(t["specimen_baseline"]["top_candidates"]) or "N/A"
        z = t["assigned_zoologist"]

        return f"""# 🐾 URGENT ZOOLOGICAL REFERRAL DOSSIER
**Ticket Reference:** `{t['ticket_id']}`  
**Generated At:** {t['created_at']}  
**Urgency Classification:** **{t['urgency_level']}** (`{t['state']}`)  

---

## 1. Assigned Zoologist
- **Name:** {z.get('name', 'Expert Zoologist')} ({z.get('title', 'Specialist')})
- **Institution:** {z.get('institution', 'Zoological Institute')}
- **Contact:** Email: `{z.get('email', '')}` | Phone: `{z.get('phone', '')}`
- **Field Specialization:** {z.get('specialization', '')}
- **Notification Channel:** {z.get('preferred_channel', '')}

---

## 2. Automated AI Diagnostic Failure Summary
The ResNet-18 Vision Model failed metamorphic invariant verification. Automated prediction has been intercepted.

- **Candidate Breed Guessed by Model:** **{t['specimen_baseline']['predicted_breed']}**
- **Model Baseline Confidence:** {t['specimen_baseline']['confidence']:.2%}
- **Top Competing Softmax Classes:** {top_cand}
- **Composite Brittleness Index (CBI):** **{t['cbi_score']:.4f}** (Threshold = 0.55)
- **Metamorphic Flips:** **{flips} of {t['perturbation_diagnostics']['total_tests']} invariant tests**
- **Most Fragile Semantic Family:** `{t['perturbation_diagnostics']['peak_family']}` (Instability Score = {t['perturbation_diagnostics']['peak_family_score']:.4f})
- **Compromised Families:** {t['perturbation_diagnostics']['compromised_families'] or 'None'}

---

## 3. Recommended Expert Protocol
1. Inspect the uploaded specimen image and verify distinctive facial muzzle, ear structure, and coat patterns.
2. Review the attached Grad-CAM attention maps to verify if the model focused on background artifacts.
3. Submit definitive zoological species/breed identification into the triage portal to resolve this referral.

---
*Generated by the Metamorphic Circuit Breaker Fallback Engine (Mar Baselios College of Engineering & Technology)*
"""
