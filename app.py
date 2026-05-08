import streamlit as st
import numpy as np
import pickle

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Crop Recommendation System",
    page_icon="🌾",
    layout="centered"
)

# ── Load model ───────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open('crop_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('label_encoder.pkl', 'rb') as f:
        le = pickle.load(f)
    return model, le

model, le = load_model()

# ── Crop info dict ────────────────────────────────────────────
crop_info = {
    'rice':        {'emoji': '🌾', 'season': 'Kharif', 'tip': 'Needs flooded fields and high humidity.'},
    'maize':       {'emoji': '🌽', 'season': 'Kharif/Rabi', 'tip': 'Grows well in well-drained loamy soil.'},
    'chickpea':    {'emoji': '🫘', 'season': 'Rabi', 'tip': 'Drought resistant, ideal for dry regions.'},
    'kidneybeans': {'emoji': '🫘', 'season': 'Kharif', 'tip': 'Needs moderate rainfall and cool climate.'},
    'pigeonpeas':  {'emoji': '🌿', 'season': 'Kharif', 'tip': 'Very drought tolerant legume crop.'},
    'mothbeans':   {'emoji': '🌱', 'season': 'Kharif', 'tip': 'Thrives in arid and semi-arid zones.'},
    'mungbean':    {'emoji': '🌱', 'season': 'Kharif', 'tip': 'Short duration crop, good for rotation.'},
    'blackgram':   {'emoji': '🫘', 'season': 'Kharif', 'tip': 'Ideal after rice/wheat in rotation.'},
    'lentil':      {'emoji': '🫘', 'season': 'Rabi', 'tip': 'Needs cool weather and less water.'},
    'pomegranate': {'emoji': '🍎', 'season': 'Annual', 'tip': 'Drought tolerant, good for dry zones.'},
    'banana':      {'emoji': '🍌', 'season': 'Annual', 'tip': 'Needs high moisture and rich soil.'},
    'mango':       {'emoji': '🥭', 'season': 'Summer', 'tip': 'Deep roots, needs dry flowering season.'},
    'grapes':      {'emoji': '🍇', 'season': 'Annual', 'tip': 'Best in Mediterranean-type climate.'},
    'watermelon':  {'emoji': '🍉', 'season': 'Summer', 'tip': 'Loves hot weather and sandy soil.'},
    'muskmelon':   {'emoji': '🍈', 'season': 'Summer', 'tip': 'Needs warm temps and dry air.'},
    'apple':       {'emoji': '🍏', 'season': 'Winter', 'tip': 'Needs cold winters for good yield.'},
    'orange':      {'emoji': '🍊', 'season': 'Winter', 'tip': 'Subtropical climate is ideal.'},
    'papaya':      {'emoji': '🍑', 'season': 'Annual', 'tip': 'Fast growing, needs frost-free areas.'},
    'coconut':     {'emoji': '🥥', 'season': 'Annual', 'tip': 'Coastal areas with high humidity.'},
    'cotton':      {'emoji': '🌸', 'season': 'Kharif', 'tip': 'Needs long frost-free growing season.'},
    'jute':        {'emoji': '🌿', 'season': 'Kharif', 'tip': 'Grows best in high humidity & rainfall.'},
    'coffee':      {'emoji': '☕', 'season': 'Annual', 'tip': 'Shade grown, needs hilly terrain.'},
}

# ── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #f5f7f2; }
    .title-box {
        background: linear-gradient(135deg, #2d6a4f, #52b788);
        padding: 2rem;
        border-radius: 16px;
        text-align: center;
        margin-bottom: 1.5rem;
        color: white;
    }
    .title-box h1 { font-size: 2.2rem; margin: 0; }
    .title-box p  { font-size: 1rem; margin: 0.4rem 0 0; opacity: 0.9; }
    .result-box {
        background: linear-gradient(135deg, #1b4332, #2d6a4f);
        padding: 2rem;
        border-radius: 16px;
        text-align: center;
        color: white;
        margin-top: 1.5rem;
    }
    .result-box h2 { font-size: 1.4rem; margin: 0 0 0.5rem; opacity: 0.85; }
    .result-box h1 { font-size: 2.8rem; margin: 0; }
    .info-card {
        background: white;
        padding: 1rem 1.5rem;
        border-radius: 12px;
        border-left: 4px solid #52b788;
        margin-top: 1rem;
    }
    .stSlider > div { padding-top: 0.3rem; }
    .section-label {
        font-weight: 700;
        font-size: 1.05rem;
        color: #2d6a4f;
        margin: 1.2rem 0 0.3rem;
    }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────
st.markdown("""
<div class="title-box">
    <h1>🌾 Crop Recommendation System</h1>
    <p>Enter your soil and climate conditions to get the best crop suggestion</p>
</div>
""", unsafe_allow_html=True)

st.markdown("### 📋 Enter Field Conditions")

# ── Input Form ────────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.markdown('<p class="section-label">🧪 Soil Nutrients</p>', unsafe_allow_html=True)
    N = st.slider("Nitrogen (N) — kg/ha", 0, 140, 60,
                  help="Amount of Nitrogen in soil")
    P = st.slider("Phosphorus (P) — kg/ha", 5, 145, 50,
                  help="Amount of Phosphorus in soil")
    K = st.slider("Potassium (K) — kg/ha", 5, 210, 45,
                  help="Amount of Potassium in soil")
    ph = st.slider("Soil pH", 3.5, 9.5, 6.5, step=0.1,
                   help="pH of soil (7 = neutral)")

with col2:
    st.markdown('<p class="section-label">🌤️ Climate Conditions</p>', unsafe_allow_html=True)
    temperature = st.slider("Temperature (°C)", 5.0, 45.0, 25.0, step=0.5)
    humidity    = st.slider("Humidity (%)", 10.0, 100.0, 70.0, step=0.5)
    rainfall    = st.slider("Rainfall (mm)", 10.0, 300.0, 120.0, step=1.0)

# ── Predict Button ────────────────────────────────────────────
st.markdown("---")
predict_btn = st.button("🌱 Recommend Crop", use_container_width=True, type="primary")

if predict_btn:
    input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    pred_encoded = model.predict(input_data)
    probabilities = model.predict_proba(input_data)[0]
    confidence = round(max(probabilities) * 100, 1)
    crop = le.inverse_transform(pred_encoded)[0]
    info = crop_info.get(crop, {'emoji': '🌿', 'season': 'N/A', 'tip': ''})

    st.markdown(f"""
    <div class="result-box">
        <h2>✅ Recommended Crop</h2>
        <h1>{info['emoji']} {crop.upper()}</h1>
        <p style="margin-top:0.5rem;font-size:1rem;opacity:0.85;">Model Confidence: <b>{confidence}%</b></p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="info-card">
        <b>🗓️ Season:</b> {info['season']}<br>
        <b>💡 Tip:</b> {info['tip']}
    </div>
    """, unsafe_allow_html=True)

    # Top 3 crops
    st.markdown("#### 🏆 Top 3 Crop Suggestions")
    top3_idx = np.argsort(probabilities)[::-1][:3]
    top3_crops = le.inverse_transform(top3_idx)
    top3_probs = probabilities[top3_idx]

    for i, (c, p_val) in enumerate(zip(top3_crops, top3_probs)):
        emoji = crop_info.get(c, {}).get('emoji', '🌿')
        medal = ['🥇', '🥈', '🥉'][i]
        st.progress(float(p_val), text=f"{medal} {emoji} {c.capitalize()} — {round(p_val*100,1)}%")

# ── Footer ────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center;color:gray;font-size:0.85rem;'>"
    "Built with ❤️ by <b>Ekamveer Singh</b> | LPU | Random Forest Classifier | Accuracy: 99.77%"
    "</p>",
    unsafe_allow_html=True
)
