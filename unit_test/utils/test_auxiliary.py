import unittest

from unit_test import TEST_FLAGS

TEST_FLAGS = ["all"]


class TestAuxiliary(unittest.TestCase):
    @unittest.skipUnless(
        "all" in TEST_FLAGS or "sha256" in TEST_FLAGS, "just do it"
    )
    def test_sha256(self) -> str:
        from easyguard.utils.auxiliary_utils import sha256

        data = "bert"
        result = sha256(data)
        print(result)

    @unittest.skipUnless(
        "all" in TEST_FLAGS or "cache_file" in TEST_FLAGS, "just do it"
    )
    def test_cache_file(self):
        from easyguard.utils.auxiliary_utils import cache_file

        print(
            cache_file(
                "test",
                set(["vocab.txt", "vocab1.txt", "vocab2.txt"]),
                model_type="deberta",
                remote_url="hdfs://haruna/home/byte_ecom_govern/user/yangzheming/asr_model/zh_deberta_base_l6_emd_20210720/",
            )
        )

    @unittest.skipUnless(
        "all" in TEST_FLAGS or "list_pretrained_models" in TEST_FLAGS,
        "just do it",
    )
    def test_list_pretrained_models(self):
        from easyguard.utils.auxiliary_utils import list_pretrained_models

        list_pretrained_models()

    @unittest.skipUnless(
        "all" in TEST_FLAGS or "hf_name_or_path_check" in TEST_FLAGS,
        "just do it",
    )
    def test_hf_name_or_path_check(self):
        from easyguard.utils.auxiliary_utils import hf_name_or_path_check

        name_or_path = "fashion-deberta-ccr-order"
        model_url = "hdfs://haruna/home/byte_ecom_govern/easyguard/models/fashion_deberta_ccr_order"
        file_name = "vocab.txt"
        model_type = "debert"
        print(hf_name_or_path_check(name_or_path, model_url, model_type))

    @unittest.skipUnless(
        "all" in TEST_FLAGS or "convert_model_weight" in TEST_FLAGS,
        "just do it",
    )
    def test_convert_model_weight(self):
        from easyguard.utils.auxiliary_utils import convert_model_weights

        path = "/root/.cache/easyguard/models/fashionxlm_moe/6c2f5988fb7ea932b4914cf0fc6c1acb2460de2ee93f2a31370fa9d45f070f37/pytorch_model_old.bin"
        # convert_model_weights(path, "backbone.", remove_old=False)


if __name__ == "__main__":
    unittest.main()
