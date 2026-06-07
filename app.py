import os
import json
import tempfile
from io import BytesIO

import av
import cv2
import numpy as np
import requests
import streamlit as st
from streamlit_webrtc import VideoProcessorBase, WebRtcMode, webrtc_streamer


# --- Page config ---
st.set_page_config(page_title="Computerpreter", layout="wide")

# --- Compact top navigation ---
nav_col, _ = st.columns([0.18, 0.82])
with nav_col:
    page = st.selectbox(
        "Navigate",
        ["Computerpreter", "About us"],
        index=0,
        key="page_nav",
        label_visibility="collapsed",
    )

if page == "About us":
    st.header("About Computerpreter")
    st.write(
        "Computerpreter is an app that bridges the Deaf world and hearing world. "
        "Victor Young and Forest Young, high school students from Skyline High School in SLC, Utah, "
        "have created this app through their knowledge and passion for AI and its uses for benefitting communities worldwide."
    )
    st.markdown(
        "Here's a video that explains how Computerpreter works as well as why Computerpreter stands out: "
        "https://youtu.be/jIPOeskewME!"
    )
    st.markdown(
        "Here's the slideshow used in the video: "
        "https://docs.google.com/presentation/d/1tQ2sYysGiEH8UfrWaAOD-ggDFwBp7mRGoyiaZViza7E/edit?usp=sharing!"
    )
    st.markdown(
        "Here's a research paper written about Computerpreter: "
        "https://docs.google.com/document/d/1W1qUcp0b5JO8nVbPvNwcDe_E6gxKtxKbMr_LEGp5OMQ/edit?usp=sharing!"
    )
    st.markdown(
        "Here's our Github code that runs Computerpreter on the cloud: "
        "https://github.com/FooStoch/ASL_Signs_Detection!"
    )

    st.header("Accolades and Awards")
    st.markdown(
        "Computerpreter was named the Congressional App Challenge (CAC) Top Apps West Region Winner: "
        "https://www.congressionalappchallenge.us/2025-winners/!"
    )
    st.markdown(
        "Computerpreter won 5th place in the World AI Competition for Youth (WAICY): "
        "https://waicy-cdn.wholeren.cn/wp-content/uploads/2025/12/WAICY-2025-Winner-Announcement-Global-4.pdf!"
    )
    st.markdown(
        "Computerpreter won 1st place in the Youth Entrepreneurship Challenge (YEC) by the Chinese Association of Science and Technology (CAST): "
        "https://drive.google.com/file/d/1OFue5TGpxAFIHckojYR5WGYabPgmK3DZ/view?usp=sharing!"
    )
    st.markdown(
        "Computerpreter won 3rd place in the High School Utah Entrepreneurship Challenge (HSUEC) and $3,100: "
        "https://lassonde.utah.edu/hsuec!"
    )
    st.caption("Navigation: use the top-left menu to return to the main app.")
    st.stop()

st.title("Computerpreter")
left_col, right_col = st.columns([1, 1])

# -------------------------
# Session state
# -------------------------
def init_state():
    defaults = {
        "playing_finger": False,
        "playing_dynamic": False,
        "current_mode": None,
        "switching": False,
        "chat_history": [],
        "audio_data": None,
        "dynamic_sequence": [],
        "fingerspelling_raw": [],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_state()

# -------------------------
# Cached resources
# -------------------------
@st.cache_resource
def load_asl_module():
    import asl_inference
    return asl_inference

@st.cache_resource
def load_finger_module():
    import fingerspelling_inference
    return fingerspelling_inference

@st.cache_resource
def load_finger_model():
    finger = load_finger_module()
    return finger.load_model("MLP_3.pt")

@st.cache_resource
def load_whisper_model():
    import whisper
    return whisper.load_model("base")

asl = load_asl_module()
finger = load_finger_module()
finger_model = load_finger_model()

# -------------------------
# Constants
# -------------------------
FINGER_LABEL_MAP = {
    0: "A",
    1: "B",
    2: "K",
    3: "L",
    4: "M",
    5: "N",
    6: "O",
    7: "P",
    8: "Q",
    9: "R",
    10: "S",
    11: "T",
    12: "C",
    13: "U",
    14: "V",
    15: "W",
    16: "X",
    17: "Y",
    18: "Z",
    19: "D",
    20: "E",
    21: "F",
    22: "G",
    23: "H",
    24: "I",
    25: "J",
}

rtc_conf = {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}


# -------------------------
# Fingerspelling processor
# -------------------------
def create_finger_processor():
    class FingerProcessor(VideoProcessorBase):
        def __init__(self):
            self.predicted_letters = []
            self.last_letter = ""
            self.mp_drawing = finger.mp_drawing
            self.mp_drawing_styles = finger.mp_drawing_styles
            self.mp_hands = finger.mp_hands

        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            try:
                img = frame.to_ndarray(format="bgr24")
                display = img.copy()

                sample, results = finger.preprocess_hand(img)

                if sample is not None and results is not None:
                    if results.multi_hand_landmarks:
                        for hand_landmarks in results.multi_hand_landmarks:
                            self.mp_drawing.draw_landmarks(
                                display,
                                hand_landmarks,
                                self.mp_hands.HAND_CONNECTIONS,
                                self.mp_drawing_styles.get_default_hand_landmarks_style(),
                                self.mp_drawing_styles.get_default_hand_connections_style(),
                            )

                    pred_idx = finger.predict(finger_model, sample)
                    pred_label = FINGER_LABEL_MAP.get(int(pred_idx), "?")
                    self.last_letter = pred_label
                    self.predicted_letters.append(pred_label)

                    try:
                        if "fingerspelling_raw" in st.session_state:
                            st.session_state["fingerspelling_raw"].append(pred_label)
                    except Exception:
                        pass

                    cv2.putText(
                        display,
                        pred_label,
                        (50, 80),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        2,
                        (0, 255, 0),
                        5,
                    )
                else:
                    cv2.putText(
                        display,
                        "No hand detected",
                        (35, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        2,
                        (0, 0, 255),
                        4,
                    )

                return av.VideoFrame.from_ndarray(display, format="bgr24")
            except Exception:
                return frame

    return FingerProcessor


# -------------------------
# Dynamic sign processor
# -------------------------
def create_dynamic_processor():
    class DynamicProcessor(VideoProcessorBase):
        def __init__(self):
            self.holistic = asl.mp_holistic.Holistic(
                static_image_mode=False,
                min_detection_confidence=0.7,
                min_tracking_confidence=0.7,
            )
            self.buffer = []
            self.max_frames = 30
            self.last_text = ""
            self.display_count = 0
            self.predicted_signs = []

        def __del__(self):
            try:
                if hasattr(self, "holistic") and self.holistic:
                    self.holistic.close()
            except Exception:
                pass

        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            try:
                img = frame.to_ndarray(format="bgr24")
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                _ = self.holistic.process(rgb)

                landmarks = asl.extract_landmarks(img)
                if landmarks is not None:
                    self.buffer.append(landmarks)
                    img = asl.draw_landmarks(img, landmarks)

                if len(self.buffer) >= self.max_frames:
                    sign, conf = asl.predict_sign(self.buffer, asl.model, asl.device)
                    self.last_text = f"{sign} ({conf * 100:.1f}%)"
                    self.display_count = self.max_frames
                    self.predicted_signs.append(sign)

                    try:
                        if "dynamic_sequence" in st.session_state:
                            st.session_state["dynamic_sequence"].append(sign)
                    except Exception:
                        pass

                    self.buffer.clear()

                if self.display_count > 0:
                    cv2.putText(
                        img,
                        self.last_text,
                        (10, img.shape[0] - 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA,
                    )
                    self.display_count -= 1

                return av.VideoFrame.from_ndarray(img, format="bgr24")
            except Exception:
                return frame

    return DynamicProcessor


# -------------------------
# Mode switching callbacks
# -------------------------
def start_fingerspelling():
    st.session_state["fingerspelling_raw"] = []
    st.session_state["playing_dynamic"] = False
    st.session_state["playing_finger"] = True
    st.session_state["current_mode"] = "Fingerspelling"


def stop_fingerspelling():
    st.session_state["playing_finger"] = False

    collected = []
    try:
        ctx = st.session_state.get("webrtc_ctx_finger")
        if ctx is not None and getattr(ctx, "video_processor", None) is not None:
            collected += getattr(ctx.video_processor, "predicted_letters", []) or []
    except Exception:
        pass

    try:
        collected = (st.session_state.get("fingerspelling_raw", []) or []) + collected
    except Exception:
        pass

    history = [c for c in collected if c]

    window_size = 10
    threshold = 6
    result = []
    prev_main = None

    for i in range(len(history) - window_size + 1):
        window = history[i:i + window_size]
        counts = {}
        for letter in window:
            counts[letter] = counts.get(letter, 0) + 1
        main_letter = max(counts, key=counts.get)
        if counts[main_letter] >= threshold and main_letter != prev_main:
            result.append(main_letter)
            prev_main = main_letter

    if not result and history:
        compressed = []
        prev = None
        for letter in history:
            if letter != prev:
                compressed.append(letter)
                prev = letter
        result = compressed

    result_string = "".join(result)
    st.session_state["chat_history"].append({"role": "assistant", "text": result_string})
    st.session_state["fingerspelling_raw"] = []


def start_dynamic():
    st.session_state["dynamic_sequence"] = []
    st.session_state["playing_finger"] = False
    st.session_state["playing_dynamic"] = True
    st.session_state["current_mode"] = "Dynamic Sign"


def stop_dynamic():
    st.session_state["playing_dynamic"] = False

    try:
        ctx_dyn = st.session_state.get("webrtc_ctx_dynamic")
        if ctx_dyn is not None:
            stop_fn = getattr(ctx_dyn, "stop", None)
            if callable(stop_fn):
                try:
                    stop_fn()
                except Exception:
                    pass
            try:
                setattr(ctx_dyn, "desired_playing_state", False)
            except Exception:
                pass
    except Exception:
        pass

    collected = []
    try:
        ctx = st.session_state.get("webrtc_ctx_dynamic")
        if ctx is not None and getattr(ctx, "video_processor", None) is not None:
            collected += getattr(ctx.video_processor, "predicted_signs", []) or []
    except Exception:
        pass

    try:
        collected = (st.session_state.get("dynamic_sequence", []) or []) + collected
    except Exception:
        pass

    final_seq = [s for s in collected if s]
    sentence = " ".join(final_seq).strip()

    if final_seq:
        content_prompt = (
            f"Here's American Sign Language gloss: {final_seq}. "
            "Please turn it into an English sentence with periods, question marks, capital letters, etc. "
            "Please output nothing else."
        )

        api_key = None
        try:
            api_key = st.secrets["openrouter"]["api_key"]
        except Exception:
            api_key = st.secrets.get("OPENROUTER_API_KEY") or os.environ.get("OPENROUTER_API_KEY")

        if not api_key:
            st.warning(
                "OpenRouter API key not found in st.secrets or environment variable 'OPENROUTER_API_KEY'. "
                "Falling back to showing raw ASL gloss."
            )
            st.session_state["chat_history"].append({"role": "assistant", "text": sentence})
        else:
            try:
                with st.spinner("Translating ASL to English..."):
                    response = requests.post(
                        url="https://openrouter.ai/api/v1/chat/completions",
                        headers={
                            "Authorization": f"Bearer {api_key}",
                            "Content-Type": "application/json",
                        },
                        data=json.dumps(
                            {
                                "model": "nvidia/nemotron-3-super-120b-a12b:free",
                                "messages": [{"role": "user", "content": content_prompt}],
                            }
                        ),
                        timeout=30,
                    )
                    if response.status_code == 200:
                        result = response.json()
                        assistant_response = result["choices"][0]["message"]["content"]
                        st.session_state["chat_history"].append(
                            {"role": "assistant", "text": assistant_response}
                        )
                    else:
                        st.session_state["chat_history"].append({"role": "assistant", "text": sentence})
            except Exception:
                st.session_state["chat_history"].append({"role": "assistant", "text": sentence})
    else:
        st.session_state["chat_history"].append({"role": "assistant", "text": ""})

    st.session_state["dynamic_sequence"] = []


# -------------------------
# Left column UI
# -------------------------
with left_col:
    mode = st.selectbox("Select mode:", ["Fingerspelling", "Dynamic Sign"])

if st.session_state["current_mode"] is not None and st.session_state["current_mode"] != mode and not st.session_state["switching"]:
    st.session_state["switching"] = True
    st.session_state["playing_finger"] = mode == "Fingerspelling"
    st.session_state["playing_dynamic"] = mode == "Dynamic Sign"
    st.session_state["current_mode"] = mode
    st.experimental_rerun()

if st.session_state["current_mode"] is None:
    st.session_state["current_mode"] = mode

if st.session_state["switching"]:
    st.session_state["switching"] = False

with left_col:
    if mode == "Fingerspelling":
        ctx_f = webrtc_streamer(
            key="finger",
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=create_finger_processor(),
            media_stream_constraints={
                "video": {"frameRate": {"ideal": 10, "max": 15}},
                "audio": False,
            },
            async_processing=True,
            rtc_configuration=rtc_conf,
            desired_playing_state=st.session_state["playing_finger"],
        )
        st.session_state["webrtc_ctx_finger"] = ctx_f

        cols = st.columns([1, 1])
        with cols[0]:
            st.button("Start Fingerspelling", key="start_finger_btn", on_click=start_fingerspelling)
        with cols[1]:
            st.button("Stop Fingerspelling", key="stop_finger_btn", on_click=stop_fingerspelling)

    else:
        ctx_d = webrtc_streamer(
            key="dynamic",
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=create_dynamic_processor(),
            media_stream_constraints={
                "video": {"frameRate": {"ideal": 10, "max": 15}},
                "audio": False,
            },
            async_processing=True,
            rtc_configuration=rtc_conf,
            desired_playing_state=st.session_state["playing_dynamic"],
        )
        st.session_state["webrtc_ctx_dynamic"] = ctx_d

        cols = st.columns([1, 1])
        with cols[0]:
            st.button("Start Dynamic Sign", key="start_dynamic_btn", on_click=start_dynamic)
        with cols[1]:
            st.button("Stop Dynamic Sign", key="stop_dynamic_btn", on_click=stop_dynamic)


# -------------------------
# Right column: speech-to-text
# -------------------------
with right_col:
    st.markdown("### Speech-to-Text")
    st.markdown("**Hit $\color{red}{\boxed{\text{Reset}}}$ when transcription is finished to save memory!**")

    try:
        from st_audiorec import st_audiorec
        record_result = st_audiorec()
    except Exception:
        try:
            import streamlit.components.v1 as components
            st_audiorec_comp = components.declare_component("st_audiorec", path="st_audiorec/frontend/build")
            record_result = st_audiorec_comp()
        except Exception as e:
            record_result = None
            st.warning(
                "Audio recorder component not available. Install streamlit-audio-recorder in requirements "
                "or include the component frontend build. Error: {}".format(e)
            )

    wav_bytes = None

    if isinstance(record_result, dict) and "arr" in record_result:
        with st.spinner("processing audio…"):
            ind, raw = zip(*record_result["arr"].items())
            ind = np.array(ind, dtype=int)
            raw = np.array(raw, dtype=int)
            sorted_bytes = raw[ind]
            stream = BytesIO(bytearray(int(v) & 0xFF for v in sorted_bytes))
            wav_bytes = stream.read()
    elif isinstance(record_result, (bytes, bytearray)):
        wav_bytes = bytes(record_result)

    if wav_bytes is not None:
        st.session_state["audio_data"] = wav_bytes
        st.success("Recording captured")

    col_t1, col_t2 = st.columns([1, 1])
    with col_t1:
        if st.button("Transcribe Audio"):
            if st.session_state.get("audio_data", None) is None:
                st.error("No recording found!")
            else:
                tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                tmp.write(st.session_state["audio_data"])
                tmp.flush()
                tmp_path = tmp.name
                tmp.close()

                model = load_whisper_model()
                with st.spinner("Transcribing..."):
                    transcription = model.transcribe(tmp_path)

                text = transcription.get("text", "").strip()
                if text:
                    st.session_state["chat_history"].append({"role": "user", "text": text})
                    st.success("Transcription added to chat")
                else:
                    st.info("No text recognized")


# -------------------------
# Chat area
# -------------------------
st.markdown("---")

cols_header = st.columns([1, 8])
with cols_header[0]:
    st.markdown("## Chat")
with cols_header[1]:
    if st.button("Clear History", key="clear_history_btn"):
        st.session_state["chat_history"] = []
        st.success("Chat history cleared")

user_input = st.chat_input("Send a message (or speak then transcribe):")
if user_input:
    st.session_state["chat_history"].append({"role": "user", "text": user_input})

for entry in st.session_state["chat_history"]:
    if entry["role"] == "user":
        st.chat_message("user").write(entry["text"])
    else:
        st.chat_message(entry.get("role", "assistant")).write(entry["text"])

st.caption(
    "AI predictions may be inaccurate. Please refer to professional interpreters for important situations such as medical or legal emergencies."
)
