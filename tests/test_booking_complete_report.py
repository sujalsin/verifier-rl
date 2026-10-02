import ast
import hashlib
import json
from pathlib import Path
import re
import unittest

from verifier_rl import booking_complete_report as report


def fenced_blocks(text):
    blocks = []
    delimiter, language, body = None, None, []
    for line in text.splitlines(keepends=True):
        if delimiter is None:
            match = re.fullmatch(r"(`{3,})([a-z]*)\n", line)
            if match:
                delimiter, language = match.groups()
                body = []
        elif line.rstrip("\n") == delimiter:
            value = "".join(body)
            if value.endswith("\n"):
                value = value[:-1]
            blocks.append((language, value))
            delimiter = None
        else:
            body.append(line)
    if delimiter is not None:
        raise AssertionError("unclosed code fence")
    return blocks


class FormattingTests(unittest.TestCase):
    def test_candidate_markdown_stays_literal(self):
        value = "```python\nprint(1)\n```\n### A false heading\n````\n"
        self.assertEqual(fenced_blocks(report.fence(value)), [("text", value)])

    def test_table_escaping_and_field_count(self):
        self.assertIn(r"a\|b", report.table(["first", "second"], [["a|b", "x\ny"]]))
        with self.assertRaisesRegex(ValueError, "width"):
            report.table(["one"], [[1,2]])

    def test_missing_data_bounds_not_flattened(self):
        self.assertEqual(report.bounds([3,4]), "3 to 4")
        self.assertEqual(report.bounds([0,0]), "0")
        self.assertEqual(report.bounds([.25,.5], percentage=True), "25.00% to 50.00%")

    def test_no_execution_or_network_primitives(self):
        tree = ast.parse(Path(report.__file__).read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                self.assertNotIn(node.func.id, {"exec", "eval", "compile", "__import__"})
            if isinstance(node, ast.Import):
                self.assertFalse({a.name.split(".")[0] for a in node.names} & {"socket", "subprocess", "modal", "requests"})


@unittest.skipUnless(
    report.OUTPUT.exists() and report.MANIFEST.exists()
    and (report.LOCAL / "download-manifest.json").exists(),
    "complete evidence comparison requires the optional private runs/ bundle; use make evidence-check",
)
class CompleteEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.text = report.OUTPUT.read_text()
        cls.manifest = json.loads(report.MANIFEST.read_text())
        cls.blocks = fenced_blocks(cls.text)
        cls.json_blocks = [(i,json.loads(value)) for i,(kind,value) in enumerate(cls.blocks) if kind == "json"]
        cls.inventory = json.loads((report.LOCAL/"modal-results/inventory.json").read_text())
        cls.result = json.loads((report.LOCAL/"modal-results/result.json").read_text())
        cls.samples = {s["sample_id"]:s for a in cls.inventory["saved_arms"].values()
                       for e in a["evaluations"].values() for s in e["samples"]}
        baseline = json.loads((report.LOCAL/"failure-evidence/baseline-generation.json").read_text())
        cls.samples.update((s["sample_id"],s) for s in baseline["samples"])
        groups = [cls.inventory["baseline"], *cls.result["policies"].values()]
        cls.rows = {r["sample_id"]:r for g in groups for r in g["programs"]}

    def test_manifest_matches_every_record_count(self):
        self.assertEqual(hashlib.sha256(self.text.encode()).hexdigest(), self.manifest["report_sha256"])
        self.assertEqual(len(self.text.encode()), self.manifest["report_bytes"])
        counts = self.manifest["counts"]
        for key,value in {"evaluation_cohorts":25, "evaluation_responses":2048, "training_updates":288,
                          "checkpoint_receipts":288, "post_training_batches":480, "protected_input_records":10906,
                          "logical_test_cases":288, "unique_test_inputs":287, "new_cloud_calls":0}.items():
            self.assertEqual(counts[key],value)
        self.assertTrue(report.validate_rendered(self.text,counts))

    def test_every_generation_source_and_score_preserved(self):
        seen = set()
        for i,record in self.json_blocks:
            if not isinstance(record,dict) or record.get("record_type") != "evaluation_response":
                continue
            sid = record["sample_id"]
            self.assertNotIn(sid,seen)
            seen.add(sid)
            sample = self.samples[sid]
            self.assertEqual(record["scores"],self.rows[sid])
            self.assertEqual(record["generation"],{k:v for k,v in sample.items() if k not in {"source","raw"}})
            self.assertEqual(self.blocks[i+1],("python",sample["source"]))
            self.assertEqual(self.blocks[i+2],("text",sample["raw"]))
        self.assertEqual(seen,set(self.samples))

    def test_every_checkpoint_receipt_preserved(self):
        seen = set()
        for _,record in self.json_blocks:
            if isinstance(record,dict) and record.get("record_type") == "checkpoint_receipt":
                identity = (record["policy"],record["update"])
                self.assertNotIn(identity,seen)
                seen.add(identity)
                original = self.inventory["saved_arms"][identity[0]]["boundaries"][str(identity[1])]
                self.assertEqual(record["receipt"],original)
        self.assertEqual(len(seen),288)

    def test_every_training_log_field_preserved(self):
        logs = [v for _,v in self.json_blocks if isinstance(v,dict) and "log_history" in v and "global_step" in v]
        self.assertEqual(len(logs),12)
        for arm in self.inventory["saved_arms"].values():
            self.assertIn(arm["metrics"],logs)

    def test_all_targeted_raw_documents_preserved(self):
        seen = set()
        for _,record in self.json_blocks:
            if not isinstance(record,dict) or "document" not in record or "sample_id" not in record:
                continue
            sid = record["sample_id"]
            seen.add(sid)
            original = json.loads((report.LOCAL/"failure-evidence"/(sid+".json")).read_text())
            self.assertEqual(record["document"],original)
        self.assertEqual(len(seen),38)

    def test_cases_and_pilot_unknowns_are_retained(self):
        lists = [value for _,value in self.json_blocks if isinstance(value,list)]
        cases = next(v for v in lists if len(v)==288 and isinstance(v[0],dict) and "case_id" in v[0])
        self.assertEqual(len({c["input_hash"] for c in cases}),287)
        pilot = next(v for _,v in self.json_blocks if isinstance(v,dict) and v.get("status")=="completed_with_uncertainty")
        self.assertEqual(pilot["summary"]["policies"]["reference-24"]["audit"]["full_pass_bounds"],[3,4])
        self.assertIn("raw training rollout/reward journals",self.text)

    def test_source_fingerprints_match_local_files(self):
        for name,entry in self.manifest["sources"].items():
            raw = (report.ROOT/name).read_bytes()
            self.assertEqual(hashlib.sha256(raw).hexdigest(),entry["sha256"],name)
            self.assertEqual(len(raw),entry["bytes"],name)

    def test_complete_configuration_and_cohort_summaries_preserved(self):
        records = [value for _,value in self.json_blocks if isinstance(value,dict)]
        config = json.loads((report.LOCAL/"modal-results/config.json").read_text())
        self.assertIn(config,records)
        for group in [self.inventory["baseline"],*self.result["policies"].values()]:
            self.assertIn(group["summary"],records)

    def test_navigation_outside_literal_evidence(self):
        prose, delimiter = [], None
        for line in self.text.splitlines():
            if delimiter is None:
                opening = re.fullmatch(r"(`{3,})[a-z]*",line)
                if opening:
                    delimiter = opening[1]
                else:
                    prose.append(line)
            elif line == delimiter:
                delimiter = None
        self.assertIsNone(delimiter)
        anchors = {re.sub(r"[^\w\- ]", "", line.lstrip("# ").lower()).replace(" ","-")
                   for line in prose if line.startswith("#")}
        links = re.findall(r"\[[^\]]+\]\(([^)]+)\)","\n".join(prose))
        self.assertGreater(len(links),22)
        for target in links:
            if target.startswith("#"):
                self.assertIn(target[1:],anchors,target)
            elif not target.startswith(("https://","http://")):
                path = target.split("#",1)[0]
                self.assertTrue((report.OUTPUT.parent/path).exists(),target)


if __name__ == "__main__":
    unittest.main()
