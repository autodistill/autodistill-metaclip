<div align="center">
  <p>
    <a align="center" href="" target="_blank">
      <img
        width="850"
        src="https://media.roboflow.com/open-source/autodistill/autodistill-banner.png"
      >
    </a>
  </p>
</div>

# Autodistill MetaCLIP Module

This repository contains the code supporting the MetaCLIP base model for use with [Autodistill](https://github.com/autodistill/autodistill).

[MetaCLIP](https://github.com/facebookresearch/MetaCLIP), developed by Meta AI Research, is a computer vision model trained using pairs of images and text. The model was described in the [Demystifying CLIP Data](https://arxiv.org/abs/2309.16671) paper. You can use MetaCLIP with autodistill for image classification.

Read the full [Autodistill documentation](https://autodistill.github.io/autodistill/).

Read the [MetaCLIP Autodistill documentation](https://autodistill.github.io/autodistill/base_models/metaclip/).

## Installation

To use MetaCLIP with autodistill, you need to install the following dependency:

```bash
pip3 install autodistill-metaclip
```

## Quickstart

### get predictions

```python
from autodistill_clip import MetaCLIP

# define an ontology to map class names to our MetaCLIP prompt
# the ontology dictionary has the format {caption: class}
# where caption is the prompt sent to the base model, and class is the label that will
# be saved for that caption in the generated annotations
# then, load the model
base_model = MetaCLIP(
    ontology=CaptionOntology(
        {
            "person": "person",
            "a forklift": "forklift"
        }
    )
)

results = base_model.predict("./image.png")
print(results)
```

### calculate and compare embeddings

```python
from autodistill_metaclip import MetaCLIP

base_model = MetaCLIP(None)

text = base_model.embed_text("coffee")
image = base_model.embed_image("coffeeshop.jpg")

print(base_model.compare(text, image))
```

## License

This project was licensed under a Creative Commons [Attribution-NonCommercial 4.0 International](LICENSE).

## 🏆 Contributing

We love your input! Please see the core Autodistill [contributing guide](https://github.com/autodistill/autodistill/blob/main/CONTRIBUTING.md) to get started. Thank you 🙏 to all our contributors!
