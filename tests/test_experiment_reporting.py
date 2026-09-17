import tempfile
import unittest
from pathlib import Path

from experiments.reporting import write_markdown_report


class ExperimentReportingTest(unittest.TestCase):
    def test_writes_protocol_and_metric_tables(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "summary.md"
            write_markdown_report(
                output,
                "测试报告",
                {"device": "CPU"},
                {"samples": 2},
                (("计算效率", ("方法", "延迟"), (("itoh", 1.25),)),),
                ("smoke only",),
            )

            text = output.read_text(encoding="utf-8")
            self.assertIn("# 测试报告", text)
            self.assertIn("samples", text)
            self.assertIn("itoh", text)
            self.assertIn("smoke only", text)
