import os
from dataclasses import dataclass

import torch
from PIL import Image
import open_clip
import numpy as np

import supervision as sv
from autodistill.detection import CaptionOntology, DetectionBaseModel

HOME = os.path.expanduser("~")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from transformers import CLIPProcessor, CLIPModel

@dataclass
class MetaCLIP(DetectionBaseModel):
    ontology: CaptionOntology
    
    def __init__(self, ontology: CaptionOntology):
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.ontology = ontology

    def predict(self, input: str, confidence: int = 0.5) -> sv.Classifications:
        prompts = self.ontology.prompts()

        image = Image.open(input)

        inputs = self.processor(text=prompts, images=image, return_tensors="pt", padding=True)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits_per_image = outputs.logits_per_image 
            probs = logits_per_image.softmax(dim=1).tolist()

            # create dictionary of prompt: probability
            probs = list(zip(prompts, probs[0]))
            
            # filter out prompts with confidence less than the threshold
            probs = [i for i in probs if i[1] > confidence]

            return sv.Classifications(
                class_id=np.array([prompts.index(i[0]) for i in probs]),
                confidence=np.array([i[1] for i in probs]),
            )