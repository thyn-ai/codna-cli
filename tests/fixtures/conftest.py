# Fixture trees are sample data for the extractor/memory tests, NOT a pytest suite.
# Keep pytest from collecting files like mini_repo/test_refund.py as real tests.
collect_ignore_glob = ["*"]
