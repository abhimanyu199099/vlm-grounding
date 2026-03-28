"""
demo/app.py — Gradio web demo.

Run:
    python demo/app.py --ckpt checkpoints/baseline/best.pt [--port 7860]

The interface accepts an uploaded image and a text query.
It calls Grounder.predict(), overlays the predicted bounding box using
draw_grounding_result(), and displays the annotated image alongside
the confidence score and box coordinates.
"""

import argparse
from pathlib import Path

import torch
from PIL import Image

from demo.inference import Grounder
from eval.visualize import draw_grounding_result


def build_demo(ckpt_path: str, device: str = "cpu"):
    try:
        import gradio as gr
    except ImportError:
        raise ImportError(
            "Gradio is required for the demo. Install it with:\n"
            "  pip install gradio"
        )

    grounder = Grounder(ckpt_path=ckpt_path, device=device)

    def predict(image: Image.Image, query: str):
        if image is None:
            return None, "Please upload an image."
        if not query or not query.strip():
            return None, "Please enter a referring expression."

        box, conf = grounder.predict(image, query.strip())

        annotated = draw_grounding_result(
            phrase=query.strip(),
            pred_box=torch.tensor(box),
            image=image,          # pass PIL image directly — no disk lookup needed
            gt_box=None,          # GT not available for user-uploaded images
            iou_score=None,
        )

        result_str = (
            f"Confidence : {conf:.3f}\n"
            f"Box (x1,y1,x2,y2) : {[round(v) for v in box]}"
        )
        return annotated, result_str

    demo = gr.Interface(
        fn=predict,
        inputs=[
            gr.Image(type="pil", label="Upload image"),
            gr.Textbox(
                label="Referring expression",
                placeholder="e.g.  the woman in a red dress",
                lines=1,
            ),
        ],
        outputs=[
            gr.Image(type="pil",  label="Grounded region"),
            gr.Textbox(label="Result", lines=3),
        ],
        title="Visual grounding demo",
        description=(
            "Upload an image and describe an object or region in it. "
            "The model will draw a bounding box around the described entity."
        ),
        allow_flagging="never",
    )
    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt",   required=True,      help="Path to best.pt checkpoint")
    parser.add_argument("--port",   type=int, default=7860)
    parser.add_argument("--device", default="cpu",      help="'cpu' or 'cuda'")
    parser.add_argument("--share",  action="store_true", help="Create a public Gradio link")
    args = parser.parse_args()

    demo = build_demo(args.ckpt, device=args.device)
    demo.launch(server_port=args.port, share=args.share)