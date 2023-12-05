import os
from dataclasses import dataclass

import numpy as np
import supervision as sv
import torch
from autodistill.detection import CaptionOntology, DetectionBaseModel
from PIL import Image
import open_clip

HOME = os.path.expanduser("~")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if not os.path.exists(f"{HOME}/.cache/autodistill/open_clip/b32_400m.pt"):
    os.makedirs(f"{HOME}/.cache/autodistill/open_clip", exist_ok=True)
    os.system(
        f"wget https://dl.fbaipublicfiles.com/MMPT/metaclip/b32_400m.pt -P {HOME}/.cache/autodistill/open_clip"
    )


@dataclass
class MetaCLIP(DetectionBaseModel):
    ontology: CaptionOntology

    def __init__(self, ontology: CaptionOntology):
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32-quickgelu",
            pretrained=f"{HOME}/.cache/autodistill/open_clip/b32_400m.pt",
        )

        self.ontology = ontology

    def predict(self, input: str, confidence: int = 0.5) -> sv.Classifications:
        prompts = self.ontology.prompts()

        image = self.preprocess(Image.open(input)).unsqueeze(0).to(DEVICE)

        text = open_clip.tokenize(prompts)

        with torch.no_grad():
            # cosine similarity as logits
            image_features = self.model.encode_image(image)
            text_features = self.model.encode_text(text)

            probs = (image_features @ text_features.T).softmax(dim=-1)

            # create dictionary of prompt: probability
            probs = list(zip(prompts, probs[0]))

            # filter out prompts with confidence less than the threshold
            probs = [i for i in probs if i[1] > confidence]

            return sv.Classifications(
                class_id=np.array([prompts.index(i[0]) for i in probs]),
                confidence=np.array([i[1] for i in probs]),
            )

    def embed_image(self, input: str) -> torch.Tensor:
        image = Image.open(input)

        inputs = self.processor(images=image, return_tensors="pt", padding=True)

        with torch.no_grad():
            outputs = self.model.get_image_features(**inputs)
            return outputs

    def embed_text(self, input: str) -> torch.Tensor:
        inputs = self.processor(text=input, return_tensors="pt", padding=True)

        with torch.no_grad():
            outputs = self.model.get_text_features(**inputs)
            return outputs

    def compare(self, embed1: torch.Tensor, embed2: torch.Tensor) -> float:
        return torch.cosine_similarity(embed1, embed2).item()
