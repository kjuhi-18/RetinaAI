import os
import json
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import cv2

import torch
import torchvision.models as models
import torchvision.transforms as transforms
import streamlit as st

# -----------------------------------------------------------------------------
# Configuration & Constants
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="RetinaAI - Diabetic Retinopathy Diagnostic System",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "train_results_v2"
FINAL_PKG_DIR = RESULTS_DIR / "final_model_package_outputs"
EDA_DIR = BASE_DIR / "eda_outputs"
SAMPLE_DIR = BASE_DIR / "sample_images"

CLASS_NAMES = [
    "No DR",
    "Mild",
    "Moderate",
    "Severe",
    "Proliferative DR"
]

CLASS_COLORS = {
    0: "#10b981",  # Emerald Green
    1: "#eab308",  # Amber Yellow
    2: "#f97316",  # Bright Orange
    3: "#ef4444",  # Crimson Red
    4: "#a855f7"   # Purple Magenta
}

CLASS_DESCRIPTIONS = {
    0: "No retinal abnormalities or diabetic microvascular lesions observed.",
    1: "Microaneurysms only. Early retinal microvascular compromise.",
    2: "More than microaneurysms but less than severe DR (microaneurysms, hard exudates, retinal hemorrhages).",
    3: "Severe intraretinal hemorrhages (≥20 in 4 quadrants), definite venous beading (2+ quadrants), or IRMA (1+ quadrant).",
    4: "Neovascularization, vitreous/preretinal hemorrhage, or fibrous tissue proliferation."
}

CLINICAL_RECOMMENDATIONS = {
    0: "Standard annual diabetic eye screening. Maintain optimal glycemic and blood pressure control.",
    1: "Repeat comprehensive dilated retinal exam within 9–12 months. Review systemic diabetes management.",
    2: "Refer to an ophthalmologist for detailed evaluation within 3–6 months. Optical coherence tomography (OCT) recommended.",
    3: "Urgent referral to retina specialist within 2–4 weeks. High risk of rapid progression to proliferative retinopathy.",
    4: "Immediate referral to retina specialist (within 24–48 hours) for panretinal photocoagulation (PRP) or anti-VEGF therapy."
}

IMAGE_SIZE = 512
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------------------------------------------------------
# Custom CSS Loader
# -----------------------------------------------------------------------------
def load_css():
    css_path = BASE_DIR / "style.css"
    if css_path.exists():
        with open(css_path, "r", encoding="utf-8") as f:
            css_content = f.read()
        st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)

load_css()

# -----------------------------------------------------------------------------
# PyTorch Model & Grad-CAM++ Cache
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_densenet_model(checkpoint_path: str):
    net = models.densenet121(weights=None, num_classes=5)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        ckpt = ckpt["model_state_dict"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]
    clean_dict = {k.replace("module.", ""): v for k, v in ckpt.items()}
    net.load_state_dict(clean_dict, strict=False)
    net.to(DEVICE)
    net.eval()
    return net

def get_transform():
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

def apply_clahe(img_pil: Image.Image, clip_limit=2.0, tile_grid_size=(8, 8)):
    img_np = np.array(img_pil.convert("RGB"))
    lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    return Image.fromarray(cv2.cvtColor(limg, cv2.COLOR_LAB2RGB))

def compute_gradcam_pp(model, input_tensor, target_class):
    target_layer = model.features.denseblock4
    cam_store = {}

    def forward_hook(module, inp, output):
        cam_store["activations"] = output
        output.register_hook(lambda grad: cam_store.__setitem__("gradients", grad))

    hook = target_layer.register_forward_hook(forward_hook)
    model.zero_grad(set_to_none=True)

    with torch.enable_grad():
        x = input_tensor.to(DEVICE).requires_grad_(True)
        logits = model(x)
        score = logits[0, int(target_class)]
        score.backward()

    hook.remove()

    A = cam_store["activations"][0]  # [channels, h, w]
    G = cam_store["gradients"][0]
    G2 = G.pow(2)
    G3 = G.pow(3)
    denom = 2.0 * G2 + (A * G3).sum(dim=(1, 2), keepdim=True)
    alpha = G2 / (denom + 1e-8)
    weights = (alpha * torch.relu(G)).sum(dim=(1, 2), keepdim=True)
    cam = torch.relu((weights * A).sum(dim=0))
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)
    cam = torch.nn.functional.interpolate(
        cam[None, None], size=(IMAGE_SIZE, IMAGE_SIZE),
        mode="bilinear", align_corners=False
    )[0, 0]
    return cam.detach().cpu().numpy(), logits.detach()

def neuro_symbolic_audit(probs, min_conf=0.55, min_adj=0.70, max_ent=1.25):
    p = np.asarray(probs, dtype=np.float64)
    p = p / (p.sum() + 1e-12)
    pred = int(np.argmax(p))
    confidence = float(p[pred])
    entropy = float(-(p * np.log(p + 1e-12)).sum())
    lo, hi = max(0, pred - 1), min(len(CLASS_NAMES), pred + 2)
    adjacent_mass = float(p[lo:hi].sum())
    
    reasons = []
    if confidence < min_conf:
        reasons.append(f"Low Model Confidence ({confidence:.1%} < {min_conf:.0%})")
    if adjacent_mass < min_adj:
        reasons.append(f"Grade Dispersion: Probability mass spread across distant grades ({adjacent_mass:.1%} < {min_adj:.0%})")
    if entropy > max_ent:
        reasons.append(f"High Uncertainty: Prediction entropy is elevated ({entropy:.2f} > {max_ent:.2f})")
    
    manual_review = len(reasons) > 0
    return {
        "pred": pred,
        "confidence": confidence,
        "entropy": entropy,
        "adjacent_mass": adjacent_mass,
        "manual_review": manual_review,
        "reasons": reasons
    }

# -----------------------------------------------------------------------------
# Sidebar Navigation & Settings
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ System Configuration")
    
    # Model Selection
    available_ckpts = {
        "DenseNet-121 C (Fine-Tuned - Primary)": RESULTS_DIR / "dn121_512_c_finetuned_best.pt",
        "DenseNet-121 C (Selected Base)": RESULTS_DIR / "dn121_512_c_best.pt",
        "DenseNet-121 B": RESULTS_DIR / "dn121_512_b_best.pt"
    }
    
    # Filter only available files
    valid_ckpts = {k: v for k, v in available_ckpts.items() if v.exists()}
    if not valid_ckpts:
        st.error("No model checkpoints found in train_results_v2.")
        selected_model_name = list(available_ckpts.keys())[0]
        selected_ckpt_path = available_ckpts[selected_model_name]
    else:
        selected_model_name = st.selectbox(
            "Select Inference Model",
            list(valid_ckpts.keys()),
            index=0
        )
        selected_ckpt_path = valid_ckpts[selected_model_name]

    st.markdown("---")
    st.markdown("### 🎛️ Neuro-Symbolic Rules")
    min_confidence = st.slider("Min Confidence Threshold", 0.40, 0.90, 0.55, 0.05)
    min_adjacent_mass = st.slider("Min Adjacent Grade Mass", 0.50, 0.95, 0.70, 0.05)
    max_entropy = st.slider("Max Uncertainty Entropy", 0.80, 2.00, 1.25, 0.05)

    st.markdown("---")
    st.markdown("### 💡 CLAHE Preprocessing")
    enable_clahe = st.checkbox("Enhance Retina with CLAHE", value=True)
    clahe_clip = st.slider("CLAHE Clip Limit", 1.0, 5.0, 2.0, 0.5)

    st.markdown("---")
    st.markdown(
        f"""
        <div style="font-size: 0.8rem; color: #94a3b8;">
            <b>Compute Device:</b> <code style="color: #38bdf8;">{DEVICE.type.upper()}</code><br>
            <b>Model:</b> DenseNet-121 (512x512)<br>
            <b>Dataset:</b> OIA-DDR Benchmark<br>
            <b>Status:</b> Ready
        </div>
        """,
        unsafe_allow_html=True
    )

# -----------------------------------------------------------------------------
# Main Header Banner
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="header-container">
        <h1 class="header-title">
            <span>👁️ RetinaAI</span>
            <span style="font-size: 1.1rem; font-weight: 500; color: #38bdf8; background: rgba(56, 189, 248, 0.15); border: 1px solid rgba(56, 189, 248, 0.3); border-radius: 9999px; padding: 0.2rem 0.8rem;">Clinical Edition</span>
        </h1>
        <div class="header-subtitle">
            Diabetic Retinopathy Automated Grading with DenseNet-121 Deep Learning, Grad-CAM++ Explainability, and Neuro-Symbolic Safety Verification.
        </div>
        <div class="pill-group">
            <span class="pill-badge cyan">⚡ DenseNet-121 Backbone</span>
            <span class="pill-badge green">🛡️ Neuro-Symbolic Audit</span>
            <span class="pill-badge purple">🔬 Grad-CAM++ Visual Heatmap</span>
            <span class="pill-badge">🏆 QWK 0.865 Benchmark Score</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# -----------------------------------------------------------------------------
# Top Navigation Tabs
# -----------------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "🔬 Retinal Fundus Analysis",
    "🧠 Neuro-Symbolic & Quantum Layer",
    "📊 Benchmark Analytics",
    "👁️ Grad-CAM++ Gallery"
])

# =============================================================================
# TAB 1: Retinal Fundus Analysis & Inference
# =============================================================================
with tab1:
    col_input, col_results = st.columns([1, 1.3], gap="large")

    with col_input:
        st.markdown(
            """
            <div class="glass-card">
                <div class="card-title">📥 Upload Retinal Fundus Image</div>
                <div style="color: #94a3b8; font-size: 0.88rem; margin-bottom: 1rem;">
                    Upload a high-resolution fundus photograph (JPEG or PNG) or choose a verified sample from the OIA-DDR test benchmark.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        sample_options = {
            "Select a preloaded sample...": None,
            "Sample 1: Grade 0 - No DR (Healthy)": "sample_grade_0_no_dr.jpg",
            "Sample 2: Grade 1 - Mild Non-Proliferative DR": "sample_grade_1_mild.jpg",
            "Sample 3: Grade 2 - Moderate DR": "sample_grade_2_moderate.jpg",
            "Sample 4: Grade 3 - Severe DR": "sample_grade_3_severe.jpg",
            "Sample 5: Grade 4 - Proliferative DR": "sample_grade_4_proliferative.jpg"
        }

        selected_sample_key = st.selectbox(
            "Or choose a verified sample retinal case:",
            list(sample_options.keys())
        )

        uploaded_file = st.file_uploader(
            "Upload custom fundus image",
            type=["jpg", "jpeg", "png"]
        )

        # Determine image source
        pil_image = None
        source_label = "Custom Upload"

        if uploaded_file is not None:
            pil_image = Image.open(uploaded_file).convert("RGB")
            source_label = uploaded_file.name
        elif sample_options[selected_sample_key] is not None:
            sample_path = SAMPLE_DIR / sample_options[selected_sample_key]
            if sample_path.exists():
                pil_image = Image.open(sample_path).convert("RGB")
                source_label = selected_sample_key
            else:
                st.warning(f"Sample file {sample_path.name} not found.")

        if pil_image is not None:
            processed_image = apply_clahe(pil_image, clip_limit=clahe_clip) if enable_clahe else pil_image

            st.markdown(
                f"""
                <div class="glass-card">
                    <div class="card-title">🖼️ Retinal Fundus Preview</div>
                    <div style="font-size: 0.85rem; color: #94a3b8; margin-bottom: 0.8rem;">
                        <b>Source:</b> {source_label} | <b>Resolution:</b> {pil_image.width} × {pil_image.height} px
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            img_col1, img_col2 = st.columns(2)
            with img_col1:
                st.image(pil_image, caption="Original Fundus", use_container_width=True)
            with img_col2:
                st.image(processed_image, caption="CLAHE Enhanced", use_container_width=True)

            analyze_button = st.button("🚀 Run Comprehensive Diagnostic Analysis", use_container_width=True)

    with col_results:
        if pil_image is None:
            st.markdown(
                """
                <div class="glass-card" style="text-align: center; padding: 4rem 2rem;">
                    <div style="font-size: 3rem; margin-bottom: 1rem;">🔬</div>
                    <div style="font-size: 1.3rem; font-weight: 700; color: #f1f5f9; margin-bottom: 0.5rem;">
                        Awaiting Fundus Input
                    </div>
                    <div style="color: #94a3b8; max-width: 400px; margin: 0 auto; font-size: 0.95rem; line-height: 1.6;">
                        Please upload a retinal fundus photograph or pick one of the preloaded clinical cases on the left to trigger DenseNet-121 inference, Grad-CAM++ explainability, and neuro-symbolic review.
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            # Run inference
            with st.spinner("Analyzing retinal microvasculature with DenseNet-121..."):
                model = load_densenet_model(str(selected_ckpt_path))
                transform = get_transform()
                input_tensor = transform(processed_image).unsqueeze(0)

                with torch.no_grad():
                    logits = model(input_tensor.to(DEVICE))
                    probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

                pred_grade = int(np.argmax(probs))
                pred_conf = float(probs[pred_grade])

                # Neuro-symbolic evaluation
                ns_result = neuro_symbolic_audit(
                    probs,
                    min_conf=min_confidence,
                    min_adj=min_adjacent_mass,
                    max_ent=max_entropy
                )

                # Compute Grad-CAM++
                cam_heatmap, _ = compute_gradcam_pp(model, input_tensor, pred_grade)

            # Diagnosis Banner
            color = CLASS_COLORS[pred_grade]
            badge_class = f"badge-dr-{pred_grade}"
            diagnosis_class = f"diagnosis-grade-{pred_grade}"

            st.markdown(
                f"""
                <div class="diagnosis-banner {diagnosis_class}">
                    <div class="diagnosis-title-row">
                        <div>
                            <div style="font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.08em; color: rgba(255,255,255,0.7); font-weight: 700;">
                                Predicted Diagnostic Grade
                            </div>
                            <h2 class="diagnosis-name">{CLASS_NAMES[pred_grade]}</h2>
                        </div>
                        <div class="diagnosis-badge {badge_class}">
                            Grade {pred_grade} • {pred_conf:.1%} Confidence
                        </div>
                    </div>
                    <div style="font-size: 0.92rem; color: #e2e8f0; margin-top: 0.4rem; line-height: 1.5;">
                        <b>Clinical Pathophysiology:</b> {CLASS_DESCRIPTIONS[pred_grade]}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            # KPI Metrics Row
            st.markdown(
                f"""
                <div class="kpi-container">
                    <div class="kpi-box">
                        <div class="kpi-value" style="color: {color};">{pred_conf:.1%}</div>
                        <div class="kpi-label">Confidence</div>
                    </div>
                    <div class="kpi-box">
                        <div class="kpi-value">{ns_result['entropy']:.2f}</div>
                        <div class="kpi-label">Shannon Entropy</div>
                    </div>
                    <div class="kpi-box">
                        <div class="kpi-value">{ns_result['adjacent_mass']:.1%}</div>
                        <div class="kpi-label">Adjacent Mass</div>
                    </div>
                    <div class="kpi-box">
                        <div class="kpi-value" style="color: {'#ef4444' if ns_result['manual_review'] else '#10b981'};">
                            {'FLAGGED' if ns_result['manual_review'] else 'PASSED'}
                        </div>
                        <div class="kpi-label">Safety Audit</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            # Neuro-Symbolic Audit Box
            if ns_result['manual_review']:
                reasons_html = "".join([f"<li>{r}</li>" for r in ns_result['reasons']])
                st.markdown(
                    f"""
                    <div class="ns-box ns-flag">
                        <div style="font-size: 1.6rem; line-height: 1;">⚠️</div>
                        <div>
                            <b style="font-size: 1rem;">Human Ophthalmologist Secondary Review Recommended</b><br>
                            The Neuro-Symbolic rule verification layer detected borderline diagnostic indicators:
                            <ul style="margin: 0.4rem 0 0.2rem 1.2rem; padding: 0;">
                                {reasons_html}
                            </ul>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    """
                    <div class="ns-box ns-pass">
                        <div style="font-size: 1.6rem; line-height: 1;">✅</div>
                        <div>
                            <b style="font-size: 1rem;">Neuro-Symbolic Safety Verification Cleared</b><br>
                            High model confidence, low Shannon entropy, and consistent probability mass concentration around the diagnosed grade.
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            # Explainability Heatmap Overlay
            st.markdown(
                """
                <div class="glass-card" style="margin-top: 1.5rem;">
                    <div class="card-title">🔍 Grad-CAM++ Attention Localization</div>
                    <div style="font-size: 0.88rem; color: #94a3b8; margin-bottom: 0.8rem;">
                        DenseNet-121 gradient-weighted class activation mapping (denseblock4) highlighting microaneurysms, hemorrhages, and vascular tortuosity driving the classification.
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            # Render Matplotlib Grad-CAM overlay
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.imshow(processed_image)
            ax.imshow(cam_heatmap, cmap="jet", alpha=0.45, vmin=0, vmax=1)
            ax.axis("off")
            fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

            # Probability Breakdown
            st.markdown(
                """
                <div class="glass-card">
                    <div class="card-title">📊 Class Probability Distribution</div>
                """,
                unsafe_allow_html=True
            )

            for idx, cname in enumerate(CLASS_NAMES):
                prob_val = float(probs[idx])
                bar_color = CLASS_COLORS[idx]
                st.markdown(
                    f"""
                    <div class="prob-item">
                        <div class="prob-header">
                            <span><b>Grade {idx}:</b> {cname}</span>
                            <span style="font-family: monospace; color: {bar_color};">{prob_val:.2%}</span>
                        </div>
                        <div class="prob-bar-track">
                            <div class="prob-bar-fill" style="width: {prob_val * 100}%; background: {bar_color};"></div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            st.markdown("</div>", unsafe_allow_html=True)

            # Clinical Action Guidelines
            st.markdown(
                f"""
                <div class="glass-card">
                    <div class="card-title">📋 Recommended Clinical Referral Pathway</div>
                    <div style="font-size: 0.95rem; line-height: 1.6; color: #e2e8f0;">
                        <b>Protocol:</b> {CLINICAL_RECOMMENDATIONS[pred_grade]}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            # JSON Report Export
            report_data = {
                "source": source_label,
                "model": selected_model_name,
                "predicted_grade_id": pred_grade,
                "predicted_grade": CLASS_NAMES[pred_grade],
                "confidence": round(pred_conf, 4),
                "entropy": round(ns_result['entropy'], 4),
                "adjacent_grade_mass": round(ns_result['adjacent_mass'], 4),
                "manual_review_recommended": ns_result['manual_review'],
                "triggered_reasons": ns_result['reasons'],
                "probabilities": {CLASS_NAMES[i]: round(float(probs[i]), 4) for i in range(5)},
                "clinical_recommendation": CLINICAL_RECOMMENDATIONS[pred_grade]
            }

            st.download_button(
                label="📥 Export Clinical Report (JSON)",
                data=json.dumps(report_data, indent=2),
                file_name=f"DR_Report_{source_label.replace(' ', '_')}.json",
                mime="application/json",
                use_container_width=True
            )

# =============================================================================
# TAB 2: Neuro-Symbolic & Quantum Layer
# =============================================================================
with tab2:
    st.markdown(
        """
        <div class="glass-card">
            <h2 style="font-size: 1.5rem; color: #f8fafc; margin-bottom: 0.5rem;">
                🧠 Neuro-Symbolic Reasoning & Quantum Hybrid Pipeline
            </h2>
            <div style="color: #94a3b8; font-size: 0.95rem; line-height: 1.6;">
                Conventional deep neural networks act as black-box probabilistic estimators. In medical diagnostics, pure argmax prediction can miss critical edge cases. This system integrates an explicit Neuro-Symbolic logic layer and a Variational Quantum Circuit (VQC) feature classification stream.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    col_ns1, col_ns2 = st.columns(2, gap="medium")

    with col_ns1:
        st.markdown(
            r"""
            <div class="glass-card">
                <div class="card-title">📐 Neuro-Symbolic Rule Formulations</div>
                <div style="font-size: 0.9rem; line-height: 1.6; color: #cbd5e1;">
                    <p><b>1. Minimum Confidence Barrier:</b><br>
                    Requires \( P(\hat{y}) \ge \tau_{\text{conf}} \) (default: 0.55). Predictions with low peak confidence are intercepted for human ophthalmologist verification.</p>
                    
                    <p><b>2. Grade Consistency / Adjacent Probability Mass:</b><br>
                    Since Diabetic Retinopathy severity is ordinal (\(0 \to 1 \to 2 \to 3 \to 4\)), plausible neural uncertainty should be concentrated in adjacent grades:<br>
                    \( M_{\text{adj}} = \sum_{k = \max(0, \hat{y}-1)}^{\min(4, \hat{y}+1)} P(y = k) \ge \tau_{\text{adj}} \) (default: 0.70).<br>
                    Dispersed probabilities indicate high-risk conflicting visual features.</p>
                    
                    <p><b>3. Shannon Uncertainty Entropy:</b><br>
                    \( H(P) = -\sum_{k=0}^{4} P_k \log(P_k + \epsilon) \le \tau_{\text{entropy}} \) (default: 1.25).<br>
                    Bounding entropy prevents chaotic distributions from reaching automatic sign-off.</p>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col_ns2:
        st.markdown(
            r"""
            <div class="glass-card">
                <div class="card-title">⚛️ Variational Quantum Circuit (VQC)</div>
                <div style="font-size: 0.9rem; line-height: 1.6; color: #cbd5e1;">
                    <p><b>PennyLane Quantum Node:</b><br>
                    The framework includes a 4-qubit parameterized quantum circuit evaluated via <code>PennyLane</code> and PyTorch backpropagation interface.</p>
                    
                    <p><b>Quantum State Encoding:</b><br>
                    Dimensionality-reduced latent retinal embeddings from DenseNet are encoded into quantum qubit rotations (\(R_x, R_y, R_z\)) followed by strongly entangling CNOT layers.</p>
                    
                    <p><b>Optimal Expected Cutoff Ensemble:</b><br>
                    Combining the DenseNet ensemble with threshold optimization yields an exceptional <b>QWK of 0.865</b> on the held-out test split:</p>
                    <div style="background: rgba(15, 23, 42, 0.7); border: 1px solid rgba(255, 255, 255, 0.1); padding: 0.8rem; border-radius: 10px; font-family: monospace; font-size: 0.85rem; color: #38bdf8;">
                        Rule: expected_cutoffs<br>
                        Cutoffs: [0.20, 0.85, 2.90, 3.025]<br>
                        Ensemble: DenseNet-121 A + DenseNet-121 B
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # Rule JSON and Summary Display
    rule_path = RESULTS_DIR / "final_rule.json"
    if rule_path.exists():
        with open(rule_path, "r", encoding="utf-8") as f:
            rule_data = json.load(f)
        st.markdown(
            f"""
            <div class="glass-card">
                <div class="card-title">⚙️ Deployed Decision Rule JSON (train_results_v2/final_rule.json)</div>
                <pre style="background: rgba(15, 23, 42, 0.8); border: 1px solid rgba(255,255,255,0.08); padding: 1rem; border-radius: 12px; color: #38bdf8;">{json.dumps(rule_data, indent=2)}</pre>
            </div>
            """,
            unsafe_allow_html=True
        )

# =============================================================================
# TAB 3: Benchmark Evaluation & Performance Analytics
# =============================================================================
with tab3:
    st.markdown(
        """
        <div class="glass-card">
            <h2 style="font-size: 1.5rem; color: #f8fafc; margin-bottom: 0.5rem;">
                📊 Model Benchmark & Test Set Performance
            </h2>
            <div style="color: #94a3b8; font-size: 0.95rem; line-height: 1.6;">
                Rigorous empirical evaluation on the OIA-DDR held-out test set (3,759 retinal images) across individual models, fine-tuned checkpoints, and neuro-symbolic threshold ensembles.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Final Results Table
    results_csv = RESULTS_DIR / "final_results.csv"
    if results_csv.exists():
        df_results = pd.read_csv(results_csv)
        if "Unnamed: 0" in df_results.columns:
            df_results.rename(columns={"Unnamed: 0": "Configuration"}, inplace=True)
        st.markdown("#### 🏆 Overall Test Set Performance Comparison")
        st.dataframe(df_results, use_container_width=True)

    # Class-wise Metrics Table
    classwise_csv = FINAL_PKG_DIR / "final_test_classwise_comparison.csv"
    if classwise_csv.exists():
        st.markdown("#### 📈 Class-wise Precision, Recall & F1-score")
        df_classwise = pd.read_csv(classwise_csv)
        st.dataframe(df_classwise, use_container_width=True)

    # Confusion Matrix Visuals
    st.markdown("#### 🔲 Confusion Matrix Visualizations")
    cm_col1, cm_col2 = st.columns(2)
    
    cm_c_path = FINAL_PKG_DIR / "densenet_c_confusion_matrix.png"
    cm_vqc_path = FINAL_PKG_DIR / "vqc_confusion_matrix.png"
    cm_final_path = RESULTS_DIR / "final_confusion_matrix.png"

    with cm_col1:
        if cm_c_path.exists():
            st.image(str(cm_c_path), caption="DenseNet-121 C Confusion Matrix", use_container_width=True)
        elif cm_final_path.exists():
            st.image(str(cm_final_path), caption="Final Confusion Matrix", use_container_width=True)
            
    with cm_col2:
        if cm_vqc_path.exists():
            st.image(str(cm_vqc_path), caption="VQC Quantum Feature Confusion Matrix", use_container_width=True)

    # Dataset Distribution Visual
    class_dist_path = EDA_DIR / "class_distribution.png"
    if class_dist_path.exists():
        st.markdown("#### 📉 OIA-DDR Dataset Class Imbalance Distribution")
        st.image(str(class_dist_path), caption="OIA-DDR Grade Distribution (Imbalance Ratio: 26.55x)", use_container_width=True)

# =============================================================================
# TAB 4: Grad-CAM++ Explainability Gallery
# =============================================================================
with tab4:
    st.markdown(
        """
        <div class="glass-card">
            <h2 style="font-size: 1.5rem; color: #f8fafc; margin-bottom: 0.5rem;">
                👁️ Grad-CAM++ Multi-Stage Lesion Gallery
            </h2>
            <div style="color: #94a3b8; font-size: 0.95rem; line-height: 1.6;">
                Visual verification across each Diabetic Retinopathy severity grade. Grad-CAM++ pinpoints whether DenseNet-121 attends to clinically valid biomarkers (e.g., microaneurysms, hard exudates, cotton wool spots, and retinal neovascularization).
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    gallery_items = [
        ("Grade 0: No DR", "gradcampp_true_no_dr_idx_0.png", "Clear macula and vascular arcades with zero pathological attention flags."),
        ("Grade 1: Mild DR", "gradcampp_true_mild_idx_6.png", "Focal attention concentrated on isolated microaneurysms near posterior pole."),
        ("Grade 2: Moderate DR", "gradcampp_true_moderate_idx_3.png", "Clustered activations corresponding to multiple hemorrhages and hard exudates."),
        ("Grade 3: Severe DR", "gradcampp_true_severe_idx_21.png", "Widespread quadrant attention across venous beading and extensive flame hemorrhages."),
        ("Grade 4: Proliferative DR", "gradcampp_true_proliferative_idx_2.png", "Intense localization at disc neovascularization and vitreoretinal traction points.")
    ]

    for title, fname, desc in gallery_items:
        fpath = FINAL_PKG_DIR / fname
        if fpath.exists():
            st.markdown(
                f"""
                <div class="glass-card">
                    <h3 style="font-size: 1.15rem; color: #38bdf8; margin: 0 0 0.4rem 0;">{title}</h3>
                    <div style="font-size: 0.9rem; color: #cbd5e1; margin-bottom: 0.8rem;">{desc}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
            st.image(str(fpath), use_container_width=True)
            st.markdown("<br>", unsafe_allow_html=True)
