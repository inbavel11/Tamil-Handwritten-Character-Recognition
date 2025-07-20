import streamlit as st
import numpy as np
import cv2
import re
import tensorflow as tf
from streamlit_drawable_canvas import st_canvas

# Load the trained model (ensure compile=False to avoid optimizer errors)
model = tf.keras.models.load_model("tamil_model_classonly_v2.h5", compile=False)

# Tamil label map (cleaned and deduplicated from raw)
raw_text = """
ா,அ	ஆ	இ	ஈ	உ	ஊ	எ	ஏ	ஐ	ஒ	ஓ	ஔ ஃ க்		க	கி	கீ	கு	கூ ச்	ச	சி	சீ	சு	சூ ங்	ங	ஙி	ஙீ	ஙு	ஙூ ஞ்		ஞ		ஞி	ஞீ	ஞு	ஞூ ட்		ட		டி	டீ	டு	டூ ண்		ண		ணி	ணீ	ணு	ணூ த்	த		தி	தீ	து	தூ ந்		ந		நி	நீ	நு	நூ ப்		ப		பி	பீ	பு	பூ ம்		ம		மி	மீ	மு	மூ	ய்	ய		யி	யீ	யு	யூ ர்		ர		ரி	ரீ	ரு	ரூ ல்		ல		லி	லீ	லு	லூ	ள்		ள		ளி	ளீ	ளு	ளூ ற்		ற		றி	றீ	று	றூ வ்		வ		வி	வீ	வு	வூ ழ்		ழ		ழி	ழீ	ழு	ழூ ன்		ன		னி	னீ	னு	ஷி	ஷீ	ஷு	ஷூ க்ஷ	க்ஷ்	க்ஷி	க்ஷீ ஜி	ஜீ	ஹ	ஹ்		ஹி	ஹீ	ஹு	ஹூ		ஸ	ஸ்	ஸி	ஸீ	ஸு	ஸூ ஷ ஷ் னூ ஸ்ரீ shu ஜ	ஜ்	ji jii srii  	ெ  ே  ை
"""
cleaned = re.sub(r"[\t\s]+", ",", raw_text.strip())
char_list = [c for c in cleaned.split(",") if c != ""]
char_list = list(dict.fromkeys(char_list))
label_map = {i: char for i, char in enumerate(char_list)}

# Preprocess function
def preprocess_image(img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, img = cv2.threshold(img, 250, 255, cv2.THRESH_BINARY_INV)
    img = cv2.resize(img, (64, 64))
    img = img / 255.0
    return img.reshape(1, 64, 64, 1)

# Predict function
def predict_single_character(img):
    processed = preprocess_image(img)
    pred = model.predict(processed)
    idx = np.argmax(pred)
    return label_map.get(idx, "?")

# UI
st.title("✍️ Tamil Handwritten Character Recognition (Single Character Only)")
st.write("Draw a single Tamil character below:")

canvas_result = st_canvas(
    fill_color="#FFFFFF",
    stroke_width=12,
    stroke_color="#000000",
    background_color="#FFFFFF",  # whiteboard
    height=300,
    width=300,
    drawing_mode="freedraw",
    key="canvas"
)

if canvas_result.image_data is not None:
    image = np.array(canvas_result.image_data).astype(np.uint8)
    st.image(image, caption="Your Drawing", use_column_width=False)

    if st.button("🔍 Predict Character"):
        pred_char = predict_single_character(image)
        st.success(f"🔤 Predicted Tamil Character: **{pred_char}**")
        st.balloons()