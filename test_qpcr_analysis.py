import unittest

import pandas as pd

from qpcr_analysis import QPCRInputError, analyze


def example():
    rows = []
    for sample, condition, reference, target in [
        ("C1", "Control", 20, 25),
        ("C2", "Control", 20, 25),
        ("T1", "Treatment", 20, 24),
        ("T2", "Treatment", 20, 24),
    ]:
        for gene, ct in [("Ref", reference), ("Target", target)]:
            for replicate in (1, 2, 3):
                rows.append((sample, condition, gene, replicate, ct))
    return pd.DataFrame(rows, columns=["Sample", "Condition", "Gene", "Replicate", "Ct"])


class TestQPCR(unittest.TestCase):
    def test_known_fold_change(self):
        result = analyze(example(), "Ref", "Control")
        treated = result.summary.query("Condition == 'Treatment' and Gene == 'Target'")
        self.assertAlmostEqual(treated["Geometric_Mean_FC"].iloc[0], 2.0)
        self.assertEqual(treated["Biological_N"].iloc[0], 2)

    def test_single_replicate_fails(self):
        data = example().query("not (Sample == 'C1' and Gene == 'Target' and Replicate != 1)")
        with self.assertRaisesRegex(QPCRInputError, "QC 실패"):
            analyze(data, "Ref", "Control")

    def test_missing_reference_fails(self):
        data = example().query("not (Sample == 'T1' and Gene == 'Ref')")
        with self.assertRaisesRegex(QPCRInputError, "누락"):
            analyze(data, "Ref", "Control")

    def test_sample_in_two_conditions_fails(self):
        data = example()
        data.loc[data.Sample == "T1", "Sample"] = "C1"
        with self.assertRaisesRegex(QPCRInputError, "여러 Condition"):
            analyze(data, "Ref", "Control")

    def test_nonnumeric_ct_fails(self):
        data = example()
        data["Ct"] = data["Ct"].astype(str)
        data.loc[0, "Ct"] = "Undetermined"
        with self.assertRaisesRegex(QPCRInputError, "Ct는"):
            analyze(data, "Ref", "Control")


if __name__ == "__main__":
    unittest.main()
