"""
streamlit_pet_app.py — Interactive Live Dashboard for ResNet-18 Oxford-IIIT Pet Model.

Features:
- Live real-time PyTorch inference with ResNet-18 (37 pet breeds).
- Full 22 hierarchical metamorphic tests across Geometric, Photometric, and Sensor/Noise families.
- Grad-CAM visual attention mapping and 4-panel flip comparison figures.
- Live parameter calibration (alpha, beta, gamma, theta_warn, theta_trip, tau_fam).
- Exploded mathematical attribution breakdown and CBI risk gauge.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st

# Setup paths
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.utils import OXFORD_PET_CLASSES, RESNET18_MODEL_PATH, get_pet_full_name, OXFORD_PET_FULL_NAMES
from src.models.resnet_adapter import ResNet18OxfordPetAdapter
from src.circuit_breaker import MetamorphicCircuitBreaker, CircuitBreakerConfig
from src.gradcam import make_comparison_figure
from src.metamorphic_families import HIERARCHICAL_TEST_MATRIX

# Page config
st.set_page_config(
    page_title="Pet Circuit Breaker Studio",
    page_icon="🐾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;900&display=swap');
.stApp { background: #0d0c1e; color: #e2e2f0; font-family: 'Inter', sans-serif; }
[data-testid="stSidebar"] { background: #13122a; border-right: 1px solid #2a2a4a; }
[data-testid="stSidebar"] * { color: #c8c8e8 !important; }
[data-testid="stMetric"] {
    background: #181730; border: 1px solid #2a2a50;
    border-radius: 12px; padding: 14px 18px;
    min-height: 110px;
}
[data-testid="stMetricLabel"] { color: #8080aa !important; font-size: 0.78rem; text-transform: uppercase; }
[data-testid="stMetricValue"] {
    color: #7fffcc !important;
    font-size: 1.28rem !important;
    font-weight: 800 !important;
    white-space: normal !important;
    word-break: normal !important;
    overflow-wrap: break-word !important;
    line-height: 1.25 !important;
}
[data-testid="stMetricValue"] > div {
    white-space: normal !important;
    overflow: visible !important;
    text-overflow: clip !important;
    word-break: normal !important;
    overflow-wrap: break-word !important;
}
.banner { border-radius: 14px; padding: 20px 28px; text-align: center;
    font-size: 1.5rem; font-weight: 900; margin: 14px 0 16px; line-height: 1.4; }
.banner-closed   { background: linear-gradient(135deg, #00c896, #06d6a0); color: #041a10; }
.banner-halfopen { background: linear-gradient(135deg, #f0a500, #ffd166); color: #1a1000; }
.banner-open     { background: linear-gradient(135deg, #c0002a, #e94560); color: #fff; }
.sec { font-size: 1.05rem; font-weight: 700; color: #b27cf4;
    border-left: 4px solid #b27cf4; padding-left: 10px; margin: 20px 0 10px; }
.card { background: #181730; border: 1px solid #2a2a50; border-radius: 10px;
    padding: 14px 18px; margin: 8px 0; font-size: .88rem; line-height: 1.6; }
.fbox { background: #0e0d22; border: 1px solid #4a3d88; border-radius: 10px;
    padding: 16px 20px; margin: 10px 0; font-family: 'Courier New', monospace;
    font-size: .86rem; line-height: 1.8; color: #d0d0ff; }
.live-badge { display: inline-block; background: #2b1d52; color: #c499f3;
    padding: 4px 14px; border-radius: 20px; font-size: 0.78rem; font-weight: 700;
    border: 1px solid #6b3fb8; margin-bottom: 8px; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading ResNet-18 Oxford Pet Model (~1.5s)...")
def get_adapter():
    return ResNet18OxfordPetAdapter()


def render_gauge(cbi: float, tw: float, tt: float):
    fig, ax = plt.subplots(figsize=(7, 0.9))
    fig.patch.set_facecolor("#0d0c1e")
    ax.set_facecolor("#181730")
    ax.barh(0.5, tw, height=0.55, left=0, color="#06d6a0", alpha=0.25)
    ax.barh(0.5, tt - tw, height=0.55, left=tw, color="#ffd166", alpha=0.25)
    ax.barh(0.5, 1.0 - tt, height=0.55, left=tt, color="#e94560", alpha=0.25)
    col = "#06d6a0" if cbi < tw else ("#ffd166" if cbi < tt else "#e94560")
    ax.axvline(cbi, color=col, lw=3.5)
    ax.text(cbi, 0.9, f" {cbi:.4f}", ha="left" if cbi < 0.85 else "right",
            va="top", color=col, fontsize=9, fontweight="bold")
    for val, lbl in [(tw, f"θ_warn={tw:.2f}"), (tt, f"θ_trip={tt:.2f}")]:
        ax.axvline(val, color="#666", lw=1, ls="--")
        ax.text(val, 0.08, lbl, ha="center", va="bottom", color="#888", fontsize=7)
    ax.set_xlim(0, 1)
    ax.set_yticks([])
    ax.set_xlabel("Composite Brittleness Index (CBI)", color="#888", fontsize=8)
    ax.tick_params(axis="x", colors="#888", labelsize=7)
    for sp in ax.spines.values():
        sp.set_edgecolor("#2a2a50")
    plt.tight_layout(pad=0.2)
    return fig


def render_family_bars(fi: dict, tau: float):
    fig, ax = plt.subplots(figsize=(4.5, 1.5))
    fig.patch.set_facecolor("#0d0c1e")
    ax.set_facecolor("#181730")
    labels = ["Geometric", "Photometric", "Sensor/Noise"]
    keys = ["geometric", "photometric", "sensor_noise"]
    vals = [fi.get(k, 0.0) for k in keys]
    colors = ["#e94560" if v >= tau else "#06d6a0" for v in vals]
    bars = ax.barh(labels, vals, color=colors, height=0.5)
    ax.axvline(tau, color="#ffd166", lw=1.5, ls="--", label=f"τ_fam={tau:.2f}")
    ax.set_xlim(0, max(1.0, max(vals) + 0.1))
    ax.tick_params(colors="#888", labelsize=7.5)
    for sp in ax.spines.values():
        sp.set_edgecolor("#2a2a50")
    ax.legend(facecolor="#181730", edgecolor="#2a2a50", labelcolor="#ccc", fontsize=7, loc="lower right")
    plt.tight_layout(pad=0.2)
    return fig


# ─── Sidebar Calibration ──────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🐾 Model Information")
    st.markdown("""
    <div class='card'>
      <b>Architecture:</b> ResNet-18<br>
      <b>Task:</b> 37 Oxford-IIIT Pet Breeds<br>
      <b>Accuracy:</b> 88.55% (Epoch 5)<br>
      <b>Weight:</b> resnet18_oxford_pet.pth<br>
      <b>Interface:</b> BaseVisionModel
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### ⚙️ Calibration Controls")
    alpha = st.slider("α (Peak Family Instability)", 0.0, 1.0, 0.50, 0.05)
    beta = st.slider("β (Cross-Family Spread)", 0.0, 1.0, 0.35, 0.05)
    gamma = round(max(0.0, 1.0 - alpha - beta), 2)
    st.caption(f"γ (Model Uncertainty 1-c0) auto-constrained: **{gamma:.2f}**")

    st.markdown("---")
    theta_warn = st.slider("θ_warn (Warning Threshold)", 0.05, 0.50, 0.25, 0.05)
    theta_trip = st.slider("θ_trip (Trip Threshold)", 0.30, 0.90, 0.55, 0.05)
    tau_fam = st.slider("τ_fam (Family Activation)", 0.10, 0.60, 0.35, 0.05)

    st.markdown("---")
    st.markdown("### 👨‍⚕️ Preferred Zoologist Contact")
    from src.zoologist_fallback import DEFAULT_ZOOLOGISTS
    zoo_names = [f"{z.name} ({z.title.split()[0]}.)" for z in DEFAULT_ZOOLOGISTS]
    selected_zoo_idx = st.selectbox(
        "Escalation Recipient:",
        range(len(DEFAULT_ZOOLOGISTS)),
        format_func=lambda i: zoo_names[i],
    )
    selected_zoologist = DEFAULT_ZOOLOGISTS[selected_zoo_idx]

    st.markdown(f"""
    <div class='card' style='font-size:0.82rem; border-color:#8040b0;'>
      <b>{selected_zoologist.name}</b><br>
      <span style='color:#a080d0;'>{selected_zoologist.institution}</span><br>
      📧 <code>{selected_zoologist.email}</code><br>
      📞 <code>{selected_zoologist.phone}</code><br>
      🔬 <i>{selected_zoologist.specialization}</i><br>
      ⚡ Preferred: <b>{selected_zoologist.preferred_channel}</b>
    </div>
    """, unsafe_allow_html=True)

# ─── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div>
  <span class='live-badge'>REAL-TIME PYTORCH INFERENCE</span><br>
  <span style='font-size:2.2rem;font-weight:900;'>Metamorphic Circuit Breaker Studio</span><br>
  <span style='color:#9999bb;font-size:.95rem'>
    Runtime Safety & Transform-Brittleness Detection for <b>ResNet-18 Oxford-IIIT Pet Classification</b>
  </span>
</div>
""", unsafe_allow_html=True)
st.markdown("---")

# ─── Main Input Columns ───────────────────────────────────────────────────────
col_input, col_info = st.columns([3, 2])

with col_input:
    mode = st.radio("Choose Input Image:", ["Sample Dog (Samoyed)", "Upload your own Pet Image"], horizontal=True)
    target_img = None

    if mode == "Sample Dog (Samoyed)":
        sample_path = ROOT / "data" / "sample_dog.jpg"
        if sample_path.exists():
            target_img = Image.open(sample_path).convert("RGB")
            st.image(target_img, caption="data/sample_dog.jpg (Samoyed)", width=260)
        else:
            st.warning("Sample dog image not found.")
    else:
        uf = st.file_uploader("Upload a dog or cat image (JPG / PNG):", type=["jpg", "jpeg", "png"])
        if uf:
            target_img = Image.open(uf).convert("RGB")
            st.image(target_img, caption=f"Uploaded: {uf.name}", width=260)

with col_info:
    st.markdown("""
    <div class='card'>
      <b>How the Metamorphic Circuit Breaker Evaluates Your Pet:</b><br>
      1. <b>Baseline Inference:</b> ResNet-18 predicts breed and confidence.<br>
      2. <b>22 Metamorphic Tests:</b> Generates 22 invariant versions across 3 families (Geometric, Photometric, Sensor/Noise).<br>
      3. <b>Level 1 & 2 Normalization:</b> Calculates weighted flip rates and family stability.<br>
      4. <b>CBI Decision:</b> Computes the Composite Brittleness Index and assigns <code>CLOSED</code>, <code>HALF-OPEN</code>, or <code>OPEN</code>.
    </div>
    """, unsafe_allow_html=True)

st.markdown("")
run_btn = st.button("🚀 Run Live Circuit Breaker Analysis", type="primary", disabled=(target_img is None))

# Initialize evaluation session state
if "evaluation_data" not in st.session_state:
    st.session_state["evaluation_data"] = None

if run_btn and target_img is not None:
    cfg = CircuitBreakerConfig(
        alpha=alpha, beta=beta, gamma=gamma,
        tau_fam=tau_fam, theta_warn=theta_warn, theta_trip=theta_trip,
        open_action="ESCALATE_TO_ZOOLOGIST",
    )

    with st.status("Executing hierarchical metamorphic runtime pipeline...", expanded=True) as status:
        st.write("Initializing ResNet18OxfordPetAdapter...")
        adapter = get_adapter()
        cb = MetamorphicCircuitBreaker(adapter, config=cfg)

        st.write(f"Executing 23 PyTorch forward passes across 22 metamorphic tests...")
        t0 = time.time()
        report, gcam_data = cb.evaluate_with_gradcam(target_img, max_gradcam_flips=3)
        elapsed = time.time() - t0

        status.update(label=f"Analysis complete in {elapsed:.2f} seconds!", state="complete")

    # Persist in session state
    st.session_state["evaluation_data"] = {
        "report": report,
        "gcam_data": gcam_data,
        "target_img": target_img,
        "elapsed": elapsed,
        "cfg": cfg,
    }
    st.session_state.pop("confirmed_ticket", None)
    st.session_state.pop("dispatch_cancelled", None)

# ── Render Evaluation Results (Persisted Across Reruns) ───────────────────────
if st.session_state.get("evaluation_data") is not None:
    eval_data = st.session_state["evaluation_data"]
    report = eval_data["report"]
    gcam_data = eval_data["gcam_data"]
    active_target_img = eval_data["target_img"]
    elapsed = eval_data["elapsed"]

    # ── Operational Banner ────────────────────────────────────────────────────
    flips = sum(1 for t in report.test_results if t.flipped)
    if report.state.value == "CLOSED":
        cls = "banner-closed"
        status_txt = "CLOSED  |  AUTO-APPROVED"
    elif report.state.value == "HALF_OPEN":
        cls = "banner-halfopen"
        status_txt = "HALF-OPEN  |  MONITOR / WARNING"
    else:
        cls = "banner-open"
        status_txt = "OPEN  |  SAFETY INTERCEPT  |  PREDICTION BLOCKED"

    # Clean action string
    display_action = "ESCALATE TO ZOOLOGIST" if "DOCTOR" in report.action else report.action.replace("_", " ")

    st.markdown(
        f"<div class='banner {cls}'>{status_txt}<br>"
        f"<span style='font-size:1rem;font-weight:600;opacity:0.9'>"
        f"Composite Brittleness Index (CBI) = {report.cbi:.4f}  |  Action: {display_action}</span></div>",
        unsafe_allow_html=True,
    )

    # ── Metric Row ────────────────────────────────────────────────────────────
    full_pred_label = get_pet_full_name(report.baseline_label)
    m1, m2, m3, m4 = st.columns([1.3, 0.9, 0.9, 0.9])
    m1.metric("Predicted Breed", full_pred_label, help=f"Dataset class: {report.baseline_label} (Oxford-IIIT Pet)")
    m2.metric("Confidence (c0)", f"{report.baseline_confidence:.1%}")
    m3.metric("Metamorphic Flips", f"{flips} / {len(report.test_results)}")
    m4.metric("CBI Risk Score", f"{report.cbi:.4f}")

    # ── Fallback Mechanism: User-Confirmed Route to Preferred Zoologist ─────
    from src.zoologist_fallback import ZoologistFallbackService
    fallback_service = ZoologistFallbackService()
    draft_ticket = fallback_service.prepare_referral(
        report=report,
        query_image=active_target_img,
        preferred_zoologist=selected_zoologist,
        gradcam_data=gcam_data,
    )

    if draft_ticket:
        is_open = draft_ticket.state == "OPEN"
        card_bg = "#2e0f17" if is_open else "#2a1e0a"
        border_col = "#e94560" if is_open else "#ffd166"
        badge_txt = "CRITICAL SAFETY INTERCEPT" if is_open else "ADVISORY WARNING"
        badge_bg = "#e94560" if is_open else "#d49b10"

        st.markdown(f"""
        <div style='background:{card_bg}; border:2px solid {border_col}; border-radius:12px; padding:18px 22px; margin:16px 0 16px;'>
          <span style='background:{badge_bg}; color:#fff; padding:3px 10px; border-radius:14px; font-size:0.75rem; font-weight:800; letter-spacing:0.5px;'>
            🚨 FALLBACK PROTOCOL TRIGGERED: {badge_txt}
          </span>
          <div style='font-size:1.15rem; font-weight:800; margin:8px 0 4px; color:#ffffff;'>
            Automated Prediction Unreliable — Expert Zoologist Consultation Available
          </div>
          <div style='color:#e0d0d8; font-size:0.88rem; line-height:1.6;'>
            The input exhibited acute transform-brittleness (CBI = <b>{report.cbi:.4f}</b>). Automated classification is <b>blocked</b>.
            The system can package your specimen image, 22 metamorphic test results, and Grad-CAM attention drift heatmaps for <b>{selected_zoologist.name}</b>.
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Explicit Human-in-the-Loop Confirmation Form
        st.markdown("#### 🔒 Zoologist Referral Authorization")
        st.markdown(
            f"**Recipient:** `{selected_zoologist.name}` ({selected_zoologist.institution})  \n"
            f"**Notification Channel:** `{selected_zoologist.preferred_channel}` to `{selected_zoologist.email}`"
        )

        confirm_dispatch = st.checkbox(
            f"I authorize transmitting the specimen image, Grad-CAM attention heatmaps, and perturbation log to {selected_zoologist.name}",
            value=False,
            key="authorize_zoo_dispatch",
        )

        btn_c1, btn_c2 = st.columns([2, 3])
        with btn_c1:
            send_btn = st.button("📤 Confirm & Dispatch to Zoologist", type="primary", disabled=not confirm_dispatch)
        with btn_c2:
            cancel_btn = st.button("✖ Keep Data Local (Do Not Dispatch)")

        if send_btn:
            confirmed_ticket = fallback_service.confirm_and_dispatch(draft_ticket)
            st.session_state["confirmed_ticket"] = confirmed_ticket
            st.success(f"✔ **Referral Confirmed & Sent!** Dispatched via {selected_zoologist.preferred_channel} to {selected_zoologist.email}")

        if cancel_btn:
            st.session_state.pop("confirmed_ticket", None)
            st.info("ℹ️ **Dispatch Cancelled:** No data was transmitted externally. Diagnostic details and heatmaps are retained strictly on this local machine.")

        # If confirmed, show ticket details and download button
        active_ticket = st.session_state.get("confirmed_ticket", None)
        if active_ticket:
            fb_col1, fb_col2 = st.columns([3, 2])
            with fb_col1:
                st.markdown(f"""
                <div class='card' style='font-size:0.86rem;'>
                  <b>Assigned Zoologist:</b> {selected_zoologist.name} ({selected_zoologist.title})<br>
                  <b>Institution:</b> {selected_zoologist.institution}<br>
                  <b>Specialization:</b> {selected_zoologist.specialization}<br>
                  <b>Direct Dispatch:</b> <code>{selected_zoologist.email}</code> | <code>{selected_zoologist.phone}</code><br>
                  <b>Triage Ticket ID:</b> <code>{active_ticket.ticket_id}</code>
                </div>
                """, unsafe_allow_html=True)
            with fb_col2:
                st.markdown(f"""
                <div class='card' style='font-size:0.86rem;'>
                  <b>Dispatch Status:</b><br>
                  <span style='color:#06d6a0;'>✔ {active_ticket.dispatch_status}</span><br>
                  <b>Grad-CAM Attached:</b> {'Yes (Attention drift included)' if active_ticket.gradcam_available else 'Baseline only'}<br>
                  <b>Local Dossier:</b> <code>{active_ticket.dossier_filepath}</code>
                </div>
                """, unsafe_allow_html=True)

            with st.expander("📄 View Official Zoologist Referral Dossier (.md)", expanded=False):
                if active_ticket.dossier_filepath and Path(active_ticket.dossier_filepath).exists():
                    dossier_text = Path(active_ticket.dossier_filepath).read_text(encoding="utf-8")
                    st.markdown(dossier_text)
                    st.download_button(
                        label="⬇️ Download Referral Dossier (.md)",
                        data=dossier_text,
                        file_name=f"{active_ticket.ticket_id}.md",
                        mime="text/markdown",
                    )
        elif not cancel_btn and not send_btn:
            st.caption("⏸️ **Status:** Awaiting explicit authorization. No patient/specimen data has been transmitted externally.")
    else:
        st.markdown("""
        <div style='background:#0f2b20; border:1px solid #06d6a0; border-radius:10px; padding:10px 16px; margin:14px 0;'>
          <span style='color:#06d6a0; font-weight:700; font-size:0.9rem;'>
            ✔ Verification Confirmed: Prediction is invariant across all 22 metamorphic tests. Zoologist fallback not required.
          </span>
        </div>
        """, unsafe_allow_html=True)

    # ── Mathematical Attribution Breakdown ────────────────────────────────────
    st.markdown("<div class='sec'>Mathematical Attribution Breakdown</div>", unsafe_allow_html=True)
    unc = 1.0 - report.baseline_confidence
    st.markdown(
        f"<div class='fbox'>"
        f"<b>CBI(x)</b> = α · max_k IR_k(x)  +  β · CFS(x)  +  γ · (1 - c0)<br>"
        f"&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;= {alpha:.2f} × {report.peak_family_score:.4f}  +  "
        f"{beta:.2f} × {report.cross_family_spread:.4f}  +  {gamma:.2f} × {unc:.4f}<br>"
        f"&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;= {alpha * report.peak_family_score:.4f}  +  "
        f"{beta * report.cross_family_spread:.4f}  +  {gamma * unc:.4f}<br>"
        f"&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;= <b>{report.cbi:.4f}</b>"
        f"</div>",
        unsafe_allow_html=True,
    )

    g1, g2 = st.columns([3, 2])
    with g1:
        st.markdown("**CBI Risk Gauge**")
        st.pyplot(render_gauge(report.cbi, theta_warn, theta_trip), use_container_width=True)
    with g2:
        st.markdown("**Family Instability Ratio (IR_k)**")
        st.pyplot(render_family_bars(report.family_instability, tau_fam), use_container_width=True)

    # ── Grad-CAM Attention Heatmaps ───────────────────────────────────────────
    st.markdown("<div class='sec'>Grad-CAM Attention & Flip Explanations</div>", unsafe_allow_html=True)
    if flips == 0:
        st.success("✅ **Zero prediction flips detected!** The model's prediction is completely invariant across all 22 transformations.")
        gc1, gc2 = st.columns(2)
        with gc1:
            st.image(active_target_img.resize((240, 240)), caption=f"Original: {full_pred_label}", width=240)
        with gc2:
            if gcam_data["baseline_cam"] is not None:
                from src.gradcam import overlay_heatmap_on_image
                ov = overlay_heatmap_on_image(active_target_img, gcam_data["baseline_cam"], out_size=240)
                st.image(ov, caption=f"Grad-CAM Attention Map ({full_pred_label})", width=240)
    else:
        st.warning(f"⚠️ **{flips} prediction flip(s) detected!** Generating visual attention shift comparisons...")
        for i, fc in enumerate(gcam_data["flip_cams"]):
            fig = make_comparison_figure(
                original_image=active_target_img,
                baseline_cam=gcam_data["baseline_cam"],
                transformed_image=fc["transformed_image"],
                flip_cam=fc["cam"],
                baseline_label=full_pred_label,
                baseline_conf=report.baseline_confidence,
                flipped_label=get_pet_full_name(fc["flipped_label"]),
                flipped_conf=fc["flipped_confidence"],
                test_name=f"{fc['test_name']} ({fc['severity']})",
                out_size=220,
            )
            st.image(fig, caption=f"Flip #{i+1}: {fc['test_name']}", use_container_width=True)

    # ── Full Test Matrix Table ────────────────────────────────────────────────
    with st.expander("📊 View Complete 22-Test Metamorphic Execution Matrix", expanded=False):
        matrix_rows = []
        for t in report.test_results:
            matrix_rows.append({
                "Test ID": t.test_id,
                "Family": t.family.capitalize(),
                "Transformation": t.test_name,
                "Severity": t.severity.capitalize(),
                "Weight w(s)": t.weight,
                "Predicted Breed": get_pet_full_name(t.pred_label),
                "Confidence": f"{t.confidence:.2%}",
                "Flipped?": "🚨 FLIP" if t.flipped else "✅ Stable",
            })
        st.dataframe(pd.DataFrame(matrix_rows), use_container_width=True, hide_index=True)
