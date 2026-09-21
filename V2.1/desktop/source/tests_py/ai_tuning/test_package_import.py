import unittest


class PackageImportTests(unittest.TestCase):
    def test_package_exports_version(self):
        import vcsel_ai_tuning

        self.assertEqual(vcsel_ai_tuning.__version__, "1.1.0")


if __name__ == "__main__":
    unittest.main()
