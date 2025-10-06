import gradio as gr
import torch
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
yolo_model = YOLO('yolov8n.pt')

def process_image(image):
    # Resize to max 1280px
    img_pil = Image.fromarray(image).convert('RGB')
    max_size = 1280
    if max(img_pil.size) > max_size:
        ratio = max_size / max(img_pil.size)
        new_size = tuple(int(dim * ratio) for dim in img_pil.size)
        img_pil = img_pil.resize(new_size, Image.LANCZOS)
    
    # Detection only
    results = yolo_model(img_pil, device='mps')[0]
    result_img = np.array(img_pil).copy()
    
    detection_count = len(results.boxes)
    detected_objects = []
    
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])
        obj_name = yolo_model.names[cls_id]
        label = f"{obj_name} {conf:.2f}"
        
        cv2.rectangle(result_img, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(result_img, label, (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        detected_objects.append(obj_name)
    
    # Summary
    if detection_count > 0:
        unique_objects = list(set(detected_objects))
        info = f"Detected {detection_count} objects: {', '.join(unique_objects)}"
    else:
        info = "No objects detected"
    
    return result_img, info

css = """
#title {
    text-align: center;
    font-size: 2.5em;
    font-weight: bold;
    margin-bottom: 10px;
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
#subtitle {
    text-align: center;
    font-size: 1.2em;
    color: #666;
    margin-bottom: 20px;
}
"""

with gr.Blocks(css=css, theme=gr.themes.Soft()) as demo:
    gr.Markdown("<h1 id='title'>AI Street Vision V1</h1>")
    gr.Markdown("<p id='subtitle'>Real-Time Object Detection for Street Scenes | YOLOv8n</p>")
    
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(label="Upload Street Scene", type="numpy")
            process_btn = gr.Button("Analyze Scene", variant="primary", size="lg")
        
        with gr.Column():
            output_image = gr.Image(label="Detection Results")
            output_text = gr.Textbox(label="Detection Summary", interactive=False)
    
    gr.Markdown("### Features:")
    gr.Markdown("- Detects: vehicles, pedestrians, traffic lights, bicycles, motorcycles, buses, trucks")
    gr.Markdown("- Powered by YOLOv8n with Apple M3 MPS acceleration")
    gr.Markdown("- **V2 Coming:** Semantic segmentation for drivable area detection")
    
    process_btn.click(
        fn=process_image,
        inputs=input_image,
        outputs=[output_image, output_text]
    )

if __name__ == "__main__":
    demo.launch(share=True)