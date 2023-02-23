import unittest
import torch

from unit_test import TEST_FLAGS

TEST_FLAGS = ["all"]


class TestXLMR(unittest.TestCase):
    @unittest.skipUnless(
        "all" in TEST_FLAGS or "xlmr" in TEST_FLAGS, "just do it"
    )
    def test_xlmr(self):
        from easyguard import AutoModel, AutoTokenizer

        text = "good good study, day day up!"
        archive = "fashion-deberta-ner"
        tokenizer = AutoTokenizer.from_pretrained(archive)
        inputs = tokenizer(text)
        print("inputs:", inputs)
        model = AutoModel.from_pretrained(archive)
        new_inputs = dict()
        new_inputs["input_ids"] = torch.tensor([[1, 11749, 11749, 9127, 11914, 11897, 11918, 115, 10706, 10706, 9143, 104, 2]])
        new_inputs["token_type_ids"] = torch.tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
        new_inputs["attention_mask"] = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
        ouputs = model(**new_inputs)
        print(ouputs)


if __name__ == "__main__":
    unittest.main()
