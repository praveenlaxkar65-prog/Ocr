import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
import time
import traceback

from engines import ENGINES

st.set_page_config(page_title="Heavy OCR Tester", page_icon="🔥", layout="wide")

st.title("🔥 Heavy Multi-Engine OCR Tester")
st.caption("Har heavy OCR engine ek hi image pe test karo — speed + accuracy compare.")

# ---------- Sidebar ----------
with st.sidebar:
    st.header("⚙️ Config")

    engine_names = list(ENGINES.keys())
    selected = st.multiselect(
        "Engines",
        engine_names,
        default=["Tesseract", "EasyOCR", "PaddleOCR", "RapidOCR", "SuryaOCR"],
    )

    lang = st.selectbox("Language", ["en", "hi", "en+hi"], index=0)

    gpu = st.checkbox("Use GPU (agar available ho)", value=False)

    st.divider()
    st.caption("Heavy engines pehli baar 30-90s le sakte hain (model download).")

# ---------- Upload ----------
uploaded = st.file_uploader(
    "Image upload karo",
    type=["png", "jpg", "jpeg", "bmp", "tiff", "webp"],
)

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    img_np = np.array(image)

    c1, c2 = st.columns([1, 2])
    with c1:
        st.image(image, caption="Input", use_container_width=True)

    with c2:
        st.markdown(f"**Size:** {image.size}")
        st.markdown(f"**Mode:** {image.mode}")

    if st.button("🚀 Run ALL selected", type="primary", use_container_width=True):
        results, timings, errors = {}, {}, {}

        progress = st.progress(0, text="Starting...")
        for i, engine in enumerate(selected):
            progress.progress((i) / len(selected), text=f"Running {engine}...")
            info = ENGINES[engine]
            fn = info["fn"]
            needs = info["needs"]

            t0 = time.time()
            try:
                if engine in ("EasyOCR", "PaddleOCR"):
                    text = fn(img_np, lang=lang, gpu=gpu)
                elif engine in ("Tesseract", "SuryaOCR"):
                    text = fn(image, lang=lang)
                elif engine == "TrOCR":
                    text = fn(image)
                else:
                    text = fn(img_np)
            except Exception as e:
                text = ""
                errors[engine] = f"{type(e).__name__}: {e}\n\n{traceback.format_exc()}"

            timings[engine] = round(time.time() - t0, 2)
            results[engine] = text

        progress.progress(1.0, text="Done ✅")

        # ---------- Results ----------
        st.subheader("📄 Extracted Text")

        for engine in selected:
            if engine in errors:
                with st.expander(f"❌ **{engine}** — FAILED ({timings[engine]}s)"):
                    st.error(errors[engine])
                continue

            with st.expander(
                f"✅ **{engine}** — ⏱ {timings[engine]}s — {len(results[engine])} chars",
                expanded=True,
            ):
                st.text_area(
                    "out",
                    results[engine],
                    height=180,
                    key=f"o_{engine}",
                    label_visibility="collapsed",
                )
                st.download_button(
                    "⬇ Download",
                    results[engine],
                    file_name=f"{engine}.txt",
                    key=f"d_{engine}",
                )

        # ---------- Comparison ----------
        st.subheader("📊 Timing Comparison")
        df = pd.DataFrame(
            {
                "Engine": list(timings.keys()),
                "Time (s)": list(timings.values()),
                "Chars": [len(results[e]) for e in timings],
                "Status": ["❌" if e in errors else "✅" for e in timings],
            }
        ).sort_values("Time (s)")

        st.dataframe(df, use_container_width=True, hide_index=True)
        st.bar_chart(df.set_index("Engine")["Time (s)"])

        # ---------- Download all ----------
        combined = "\n\n".join(
            f"===== {e} ({timings[e]}s) =====\n{results[e]}" for e in selected
        )
        st.download_button(
            "⬇ Download All Results (.txt)",
            combined,
            file_name="ocr_results.txt",
        )
