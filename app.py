"""
app.py
------
Streamlit web app to demo the pipeline visually — this is the easiest way
to "see the model running" without touching the command line.

Run locally:
    streamlit run app.py

Then open the URL it prints (usually http://localhost:8501) in your browser.
"""

import time
import tempfile
import os

import cv2
import streamlit as st

from src.pipeline import CrowdPoseActionPipeline

st.set_page_config(page_title="Crowd Pose & Action Recognition", layout="wide")
st.title("🧍‍♂️🧍‍♀️ Human Pose Estimation & Action Recognition in Crowded Scenes")
st.caption("YOLOv8-Pose (multi-person) + built-in tracking + per-person action classification")


@st.cache_resource(show_spinner="Loading model (first run downloads weights)...")
def get_pipeline(config_path="config.yaml"):
    return CrowdPoseActionPipeline(config_path=config_path)


with st.sidebar:
    st.header("Settings")
    source_type = st.radio("Input source", ["Upload video", "Webcam (snapshot)"])
    st.markdown("---")
    st.markdown(
        "**Model status**\n\n"
        "- Pose model: YOLOv8-Pose ✅\n"
        "- Tracker: ByteTrack ✅\n"
        "- Action recognizer: rule-based (instant) or trained LSTM if a checkpoint exists\n"
    )

pipeline = get_pipeline()

if source_type == "Upload video":
    uploaded = st.file_uploader("Upload a video (mp4/avi/mov)", type=["mp4", "avi", "mov"])
    if uploaded is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded.read())
        video_path = tfile.name

        st.video(video_path)
        run_btn = st.button("▶ Run pose estimation + action recognition")

        if run_btn:
            cap = cv2.VideoCapture(video_path)
            frame_placeholder = st.empty()
            stats_placeholder = st.empty()
            progress_bar = st.progress(0)

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
            action_tally = {}
            frame_idx = 0
            t0 = time.time()

            out_path = os.path.join(tempfile.gettempdir(), "annotated_output.mp4")
            writer = None

            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                annotated, results = pipeline.process_frame(frame)

                if writer is None:
                    h, w = annotated.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(out_path, fourcc, 20, (w, h))
                writer.write(annotated)

                for r in results:
                    action_tally[r["action"]] = action_tally.get(r["action"], 0) + 1

                frame_idx += 1
                if frame_idx % 3 == 0:  # update UI every few frames for speed
                    frame_placeholder.image(
                        cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                        channels="RGB",
                        caption=f"Frame {frame_idx}/{total_frames}",
                    )
                    stats_placeholder.write({"People currently tracked": len(results),
                                              "Action counts so far": action_tally})
                progress_bar.progress(min(frame_idx / total_frames, 1.0))

            cap.release()
            if writer is not None:
                writer.release()

            elapsed = time.time() - t0
            st.success(f"Done! Processed {frame_idx} frames in {elapsed:.1f}s "
                       f"({frame_idx / max(elapsed,1e-6):.2f} FPS).")
            st.subheader("Final action tally")
            st.bar_chart(action_tally)

            with open(out_path, "rb") as f:
                st.download_button("⬇ Download annotated video", f,
                                    file_name="annotated_output.mp4", mime="video/mp4")

else:
    st.info("Take a snapshot from your webcam to test pose estimation on a single frame.")
    img_file = st.camera_input("Capture a frame")
    if img_file is not None:
        import numpy as np
        from PIL import Image

        image = Image.open(img_file)
        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        annotated, results = pipeline.process_frame(frame)
        st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), caption="Result")
        st.write(f"People detected: {len(results)}")
        for r in results:
            st.write(f"ID {r['track_id']}: **{r['action']}** (conf={r['conf']:.2f})")
