import cv2
import numpy as np
import sys
sys.path.append("..")
from segment_anything import SamPredictor, sam_model_registry


checkpoint = "../models/sam_vit_h_4b8939.pth"
model_type = "vit_h"
sam = sam_model_registry[model_type](checkpoint=checkpoint)
sam.to(device='mps')
predictor = SamPredictor(sam)

image = cv2.imread('src/assets/data/truck.jpg')
predictor.set_image(image)
image_embedding = predictor.get_image_embedding().cpu().numpy()
np.save("src/assets/data/truck_embedding.npy", image_embedding)

# run from within scripts folder 
# python export_onnx_model.py 
# --checkpoint ../models/sam_vit_h_4b8939.pth 
# --output ../demo/model/sam_onnx_example.onnx 
# --model-type vit_h 
# --quantize-out ../demo/model/sam_onnx_quantized_example.onnx