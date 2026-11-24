
import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from PIL import Image

st.set_page_config(
    page_title="Brain Tumor MRI Classifier",
    page_icon="🧠",
    layout="centered"
)

CLASS_NAMES = ["Meningioma", "Glioma", "Pituitary Tumor"]
CHECKPOINT_PATH = "models/brain_tumor_cnn_weighted_fold1_epoch2.pth"

class BrainTumorCNN(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))

@st.cache_resource
def load_model():
    model = BrainTumorCNN()
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model

st.title("🧠 Brain Tumor MRI Classifier")
st.write(
    "Upload a brain MRI image to obtain an experimental prediction "
    "among three tumor categories."
)

st.info(
    "Supported formats: PNG, JPG, JPEG, BMP and TIFF. "
    "Upload a single 2D MRI slice."
)

uploaded_file = st.file_uploader(
    "Choose an MRI image",
    type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"]
)

if uploaded_file is not None:
    try:
        original_image = Image.open(uploaded_file).convert("L")

        if st.button("Analyze MRI", type="primary", use_container_width=True):
            with st.spinner("Analyzing MRI image..."):
                resized = original_image.resize(
                    (224, 224),
                    resample=Image.Resampling.BILINEAR
                )

                image_array = np.asarray(resized, dtype=np.float32)
                minimum = image_array.min()
                maximum = image_array.max()

                normalized = (
                    (image_array - minimum) /
                    (maximum - minimum + 1e-8)
                )

                input_tensor = torch.from_numpy(normalized).unsqueeze(0).unsqueeze(0)

                model = load_model()

                with torch.no_grad():
                    logits = model(input_tensor)
                    probabilities = torch.softmax(logits, dim=1)[0]

                predicted_index = int(torch.argmax(probabilities))
                predicted_class = CLASS_NAMES[predicted_index]

            st.success("Analysis completed")

            st.subheader("Prediction")
            st.markdown(f"## {predicted_class}")

            st.subheader("Model confidence scores")
            for index, class_name in enumerate(CLASS_NAMES):
                score = float(probabilities[index])
                st.write(f"**{class_name}: {score * 100:.2f}%**")
                st.progress(score)

            st.caption(
                "These scores are model outputs, not calibrated diagnostic "
                "probabilities or medical certainty."
            )

    except Exception as error:
        st.error(f"Could not process this image: {error}")

st.divider()

st.warning(
    "**Research demonstration only.** This experimental model was trained "
    "to choose among Meningioma, Glioma and Pituitary Tumor. It has no "
    "normal/no-tumor class and cannot identify other conditions. "
    "It may produce a tumor prediction even for a normal or unsuitable image. "
    "Do not use its output for diagnosis, treatment or medical decisions."
)

st.caption(
    "Model: experimental Fold 1 development checkpoint | "
    "Input: 224 × 224 grayscale image | Device: CPU"
)
