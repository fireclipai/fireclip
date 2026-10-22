import io
import tempfile
import time
import urllib.parse

import imageio
import numpy as np
import requests
import streamlit as st
from PIL import Image

st.set_page_config(page_title="FireClip", page_icon="🔥", layout="centered")
st.title("🔥 FireClip")
st.caption("Text → Image → Video, on free tiers.")

SIZES = {
    "Landscape": {"Low": (768, 432), "Medium": (1024, 576)},
    "Portrait": {"Low": (432, 768), "Medium": (576, 1024)},
}

# ---------- Sidebar settings ----------
with st.sidebar:
    st.header("⚙️ Settings")
    orientation = st.radio("Orientation", ["Landscape", "Portrait"])
    quality = st.selectbox("Resolution", ["Low", "Medium"], index=1)
    W, H = SIZES[orientation][quality]
    st.caption(f"Output: {W}×{H}")
    st.divider()
    st.subheader("Image engine")
    engine = st.selectbox(
        "Engine",
        ["Pollinations (no key)", "Hugging Face – SDXL", "Hugging Face – FLUX.1 schnell"],
    )
    hf_token = st.text_input("Hugging Face token (optional)", type="password")
    st.divider()
    st.subheader("Video engine")
    backend_url = st.text_input(
        "Colab/Kaggle AI-video backend URL (optional)",
        placeholder="https://xxxx.ngrok-free.app",
    )

# ---------- Generators ----------
def gen_pollinations(prompt, w, h):
    url = (
        "https://image.pollinations.ai/prompt/"
        + urllib.parse.quote(prompt)
        + f"?width={w}&height={h}&model=flux&nologo=true&seed={int(time.time())}"
    )
    for _ in range(3):  # anonymous tier is rate limited; retry gently
        r = requests.get(url, timeout=120)
        if r.status_code == 200 and r.headers.get("content-type", "").startswith("image"):
            return Image.open(io.BytesIO(r.content)).convert("RGB")
        time.sleep(16)
    raise RuntimeError("Pollinations is busy or rate-limited. Try again in a minute.")


def gen_hf(prompt, w, h, model, token):
    from huggingface_hub import InferenceClient

    if not token:
        raise RuntimeError("Add a Hugging Face token in the sidebar.")
    client = InferenceClient(api_key=token)
    img = client.text_to_image(prompt, model=model, width=w, height=h)
    return img.convert("RGB")


def text_to_image(prompt):
    if engine.startswith("Pollinations"):
        return gen_pollinations(prompt, W, H)
    model = (
        "stabilityai/stable-diffusion-xl-base-1.0"
        if "SDXL" in engine
        else "black-forest-labs/FLUX.1-schnell"
    )
    try:
        return gen_hf(prompt, W, H, model, hf_token)
    except Exception as e:
        st.warning(f"{engine} failed ({e}). Falling back to Pollinations.")
        return gen_pollinations(prompt, W, H)


def fit(img, w, h):
    """Resize + center-crop to exactly w×h."""
    scale = max(w / img.width, h / img.height)
    img = img.resize((int(img.width * scale) + 1, int(img.height * scale) + 1), Image.LANCZOS)
    l, t = (img.width - w) // 2, (img.height - h) // 2
    return img.crop((l, t, l + w, t + h))


def ken_burns(img, seconds, fps, motion):
    """Instant, free 'fake' image-to-video using zoom/pan."""
    base = fit(img, W, H)
    big = base.resize((int(W * 1.3), int(H * 1.3)), Image.LANCZOS)
    n = seconds * fps
    path = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
    with imageio.get_writer(path, fps=fps, codec="libx264", pixelformat="yuv420p") as wr:
        for i in range(n):
            t = i / max(n - 1, 1)
            if motion == "Zoom in":
                z, cx = 1.0 + 0.3 * t, big.width / 2
            elif motion == "Zoom out":
                z, cx = 1.3 - 0.3 * t, big.width / 2
            else:  # Pan left to right
                z, cx = 1.15, big.width * (0.4 + 0.2 * t)
            cy = big.height / 2
            cw, ch = big.width / z, big.height / z
            l = max(0, min(cx - cw / 2, big.width - cw))
            tp = max(0, min(cy - ch / 2, big.height - ch))
            frame = big.crop((int(l), int(tp), int(l + cw), int(tp + ch))).resize((W, H), Image.LANCZOS)
            wr.append_data(np.asarray(frame))
    return path


def ai_video(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    r = requests.post(
        backend_url.rstrip("/") + "/video",
        files={"file": ("in.png", buf.getvalue(), "image/png")},
        headers={"ngrok-skip-browser-warning": "1"},
        timeout=900,
    )
    r.raise_for_status()
    return r.content


# ---------- Tabs ----------
tab_img, tab_vid = st.tabs(["🖼️ Image", "🎬 Video"])

with tab_img:
    prompt = st.text_area("Describe your image", placeholder="A fox made of fire running through a snowy forest, cinematic")
    if st.button("Generate image", type="primary"):
        if not prompt.strip():
            st.error("Please enter a prompt.")
        else:
            with st.spinner("Generating…"):
                try:
                    st.session_state["img"] = text_to_image(prompt.strip())
                except Exception as e:
                    st.error(str(e))
    if "img" in st.session_state:
        st.image(st.session_state["img"], use_container_width=True)
        buf = io.BytesIO()
        st.session_state["img"].save(buf, format="PNG")
        st.download_button("Download PNG", buf.getvalue(), "fireclip.png", "image/png")

with tab_vid:
    up = st.file_uploader("Reference image (or use the one from the Image tab)", type=["png", "jpg", "jpeg", "webp"])
    src = Image.open(up).convert("RGB") if up else st.session_state.get("img")
    if src is not None:
        st.image(src, width=240)
    mode = st.radio(
        "Mode",
        ["Quick clip (instant, zoom/pan)", "AI video (needs Colab backend, slow)"],
    )
    if mode.startswith("Quick"):
        motion = st.selectbox("Motion", ["Zoom in", "Zoom out", "Pan"])
        secs = st.slider("Seconds", 2, 8, 4)
    if st.button("Make video", type="primary"):
        if src is None:
            st.error("Generate or upload an image first.")
        else:
            with st.spinner("Rendering…"):
                try:
                    if mode.startswith("Quick"):
                        path = ken_burns(src, secs, 24, motion)
                        data = open(path, "rb").read()
                    else:
                        if not backend_url:
                            raise RuntimeError("Paste your Colab backend URL in the sidebar.")
                        data = ai_video(src)
                    st.session_state["video"] = data
                except Exception as e:
                    st.error(str(e))
    if "video" in st.session_state:
        st.video(st.session_state["video"])
        st.download_button("Download MP4", st.session_state["video"], "fireclip.mp4", "video/mp4")
