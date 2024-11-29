import pytest
from scripts.preprocess import small_data

def test_small_data():
    # Mock paths and parameters
    directory_y = "mock/targets"
    directory_x = "mock/images"
    substring = "test"
    num_files = 10

    # Assert output shapes
    X, y = small_data(directory_y, directory_x, substring, num_files)
    assert X.shape[0] == y.shape[0], "Mismatch in X and y sample sizes"
