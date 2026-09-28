import unittest

from verifier_rl.sft_data import collate_rows, dataset_manifest, encode_example, families, validate_dataset


class FakeTokenizer:
    eos_token_id = 999

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        prefix = "USER:" + messages[0]["content"] + "\nASSISTANT:"
        return prefix if add_generation_prompt else prefix + messages[1]["content"] + "~"

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [999 if c == "~" else ord(c) for c in text]}


class SFTDataTests(unittest.TestCase):
    def test_targets_and_family_split(self):
        result = validate_dataset()
        self.assertTrue(result["passed"])
        self.assertEqual(result["target_checks"], 40)
        data = dataset_manifest()
        self.assertEqual(len(data["examples"]), 40)
        self.assertEqual(len({x["id"] for x in data["examples"]}), 40)
        self.assertEqual(len({x["source_hash"] for x in data["examples"]}), 10)
        self.assertEqual(len([x for x in data["examples"] if x["split"] == "train"]), 32)
        self.assertEqual(len(families()), 10)

    def test_masks_prompt_not_assistant_or_real_eos(self):
        tokenizer = FakeTokenizer()
        sample = {"prompt": "short", "completion": "def f(): return []"}
        row = encode_example(tokenizer, sample)
        n = len("USER:short\nASSISTANT:")
        self.assertEqual(row["labels"][:n], [-100] * n)
        self.assertEqual(row["labels"][n:], row["input_ids"][n:])
        self.assertEqual(row["labels"][-1], tokenizer.eos_token_id)
        shorter = encode_example(tokenizer, {"prompt": "x", "completion": "y"})
        batch = collate_rows([row, shorter], tokenizer.eos_token_id)
        self.assertEqual(batch["labels"][1][len(shorter["labels"]) - 1], tokenizer.eos_token_id)
        self.assertEqual(batch["labels"][1][len(shorter["labels"]):], [-100] * (len(row["labels"]) - len(shorter["labels"])))
        self.assertEqual(sum(batch["attention_mask"][1]), len(shorter["labels"]))

    def test_refuses_truncation_missing_eos_and_template_mismatch(self):
        sample = {"prompt": "x", "completion": "y"}
        with self.assertRaises(ValueError): encode_example(FakeTokenizer(), sample, max_tokens=2)
        bad = FakeTokenizer()
        bad.eos_token_id = None
        with self.assertRaises(ValueError): encode_example(bad, sample)

        class BadTemplate(FakeTokenizer):
            def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
                return "wrong" if add_generation_prompt else "other~"

        with self.assertRaises(ValueError): encode_example(BadTemplate(), sample)

    def test_refuses_token_prefix_mismatch(self):
        class BadTokens(FakeTokenizer):
            def __call__(self, text, add_special_tokens=False):
                ids = super().__call__(text, add_special_tokens)["input_ids"]
                if text.endswith("~"):
                    ids[0] += 1
                return {"input_ids": ids}

        with self.assertRaises(ValueError): encode_example(BadTokens(), {"prompt": "x", "completion": "y"})
