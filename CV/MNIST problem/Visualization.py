import streamlit as st
import streamlit_drawable_canvas as draw
from streamlit_drawable_canvas import st_canvas
import torch
import numpy as np
from torchvision.transforms import Resize
import torch.nn.functional as F
import importlib
import os
import sys
import argparse

resize = Resize((28, 28))

# Specify canvas parameters in application
drawing_mode = st.sidebar.selectbox(
    "Drawing tool:", ("freedraw", "line", "rect", "circle", "transform")
)

stroke_width = st.sidebar.slider("Stroke width: ", 1, 25, 15)
if drawing_mode == "point":
    point_display_radius = st.sidebar.slider("Point display radius: ", 1, 25, 3)
stroke_color = st.sidebar.color_picker("Stroke color hex: ", "#fff")
bg_color = st.sidebar.color_picker("Background color hex: ", "#000")


# Create a canvas component
canvas_result = st_canvas(
    fill_color="rgba(0, 0, 0, 0.5)",  # Fixed fill color with some opacity
    stroke_width=stroke_width,
    stroke_color=stroke_color,
    background_color=bg_color,
    background_image=None,
    height=400,
    width=400,
    drawing_mode=drawing_mode,
    point_display_radius=point_display_radius if drawing_mode == "point" else 0,
    key="canvas",
)


def get_tensor(image):
    image = np.moveaxis(image, [0, 1, 2], [1, 2, 0])
    image = resize(torch.tensor(image))[0]
    return image[None, :, :]


def import_model(path):
    model = torch.load(path, weights_only=False)
    return model


def running_model(model, device):
    button = st.button("Send Image to Model")
    if button:
        image = canvas_result.image_data
        image = get_tensor(image).half().to(device)
        with torch.inference_mode():
            preds = model(image).detach().cpu()
        st.write(f"Predicted: {torch.argmax(preds, dim=1).item()}")
        st.write(f"Prediction probs: {(F.softmax(preds.squeeze(), dim=-1)).round(decimals=2).tolist()}")


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_lib(lib_path, class_name):
    module_dir = os.path.dirname(lib_path)
    if module_dir not in sys.path:
        sys.path.append(module_dir)
    module_name = os.path.splitext(os.path.basename(lib_path))[0]
    globals()[class_name] = getattr(importlib.import_module(module_name), class_name)



def main(args):
    model_classname = args.ModelClassName
    model_class = args.ClassPath
    get_lib(model_class, model_classname)
    path = args.model
    device = get_device()
    model = import_model(path).to(device)
    running_model(model, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Web interface to interact with model and draw numbers"
    )
    parser.add_argument(
        "model", help="Input file with full model, saved like torch.save(model, path)"
    )
    parser.add_argument(
        "ClassPath", help="path to python file with current model class"
    )
    parser.add_argument("ModelClassName", help="Model class name in python file")
    args = parser.parse_args()
    main(args)
