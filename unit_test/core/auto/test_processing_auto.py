import unittest

import requests
from PIL import Image
from unit_test import TEST_FLAGS

TEST_FLAGS = ["falbert"]


class TestProcessingAuto(unittest.TestCase):
    @unittest.skipUnless(
        "all" in TEST_FLAGS or "original_clip" in TEST_FLAGS, "just do it"
    )
    def test_original_clip(self):
        from transformers import AutoProcessor, CLIPModel, CLIPProcessor

        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        processor = AutoProcessor.from_pretrained(
            "openai/clip-vit-base-patch32"
        )

        url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        image = Image.open(requests.get(url, stream=True).raw)

        inputs = processor(
            text=["a photo of a cat", "a photo of a dog", "a photo of kitty"],
            images=image,
            return_tensors="pt",
            padding=True,
        )

        outputs = model(**inputs)
        logits_per_image = (
            outputs.logits_per_image
        )  # this is the image-text similarity score
        probs = logits_per_image.softmax(
            dim=1
        )  # we can take the softmax to get the label probabilitiesx = 1

    @unittest.skipUnless(
        "all" in TEST_FLAGS or "clip" in TEST_FLAGS, "just do it"
    )
    def test_clip(self):
        from easyguard import AutoModel, AutoProcessor

        path = "openai/clip-vit-base-patch32"
        model = AutoModel.from_pretrained(path)
        processor = AutoProcessor.from_pretrained(path)
        url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        image = Image.open(requests.get(url, stream=True).raw)

        inputs = processor(
            text=["a photo of a cat", "a photo of a dog", "a photo of kitty"],
            images=image,
            return_tensors="pt",
            padding=True,
        )

        outputs = model(**inputs)
        logits_per_image = (
            outputs.logits_per_image
        )  # this is the image-text similarity score
        probs = logits_per_image.softmax(
            dim=1
        )  # we can take the softmax to get the label probabilitiesx = 1

    @unittest.skipUnless(
        "all" in TEST_FLAGS or "falbert" in TEST_FLAGS, "just do it"
    )
    def test_falbert(self):
        from easyguard import AutoModel, AutoProcessor

        archive = "falbert-hq-live"
        processor = AutoProcessor.from_pretrained(archive)


if __name__ == "__main__":
    unittest.main()