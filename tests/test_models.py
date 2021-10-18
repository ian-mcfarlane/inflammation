"""Tests for statistics functions within the Model layer."""

import numpy as np
import numpy.testing as npt
from unittest.mock import patch
import pytest

@pytest.mark.parametrize(
    "test, expected",
    [
        ([[0, 0], [0, 0], [0, 0]], [0, 0]),
        ([[1, 2], [3, 4], [5, 6]], [3, 4]),
    ])
def test_daily_mean(test, expected):
    """Test mean function works for array of zeroes and positive integers."""
    from inflammation.models import daily_mean
    npt.assert_array_equal(np.array(expected), daily_mean(np.array(test)))


@pytest.mark.parametrize(
    "test, expected",
    [
        ([[1, 6], [3, 4], [5, 2]], [5, 6]),
        ([[-1, -2], [-3, -4], [-5, -6]], [-1, -2]),
    ])
def test_daily_max(test, expected):
    """Test that max function works for an array of positive or negative integers."""
    from inflammation.models import daily_max
    npt.assert_array_equal(np.array(expected), daily_max(np.array(test)))

@pytest.mark.parametrize(
    "test, expected",
    [
        ([[1, 6], [3, 4], [5, 2]], [1, 2]),
        ([[-1, -2], [-3, -4], [-5, -6]], [-5, -6]),
    ])
def test_daily_min(test, expected):
    """Test that min function works for an array of positive integers."""
    from inflammation.models import daily_min
    npt.assert_array_equal(np.array(expected), daily_min(np.array(test)))


def test_daily_min_string():
    """Test for TypeError when passing strings"""
    from inflammation.models import daily_min

    with pytest.raises(TypeError):
        error_expected = daily_min([['Hello', 'there'], ['General', 'Kenobi']])


@patch('inflammation.models.get_data_dir', return_value='/data_dir')
def test_load_csv(mock_get_data_dir):
    from inflammation.models import load_csv
    with patch('numpy.loadtxt') as mock_loadtxt:
        load_csv('test.csv')
        name, args, kwargs = mock_loadtxt.mock_calls[0]
        assert kwargs['fname'] == '/data_dir/test.csv'
        load_csv('/test.csv')
        name, args, kwargs = mock_loadtxt.mock_calls[1]
        assert kwargs['fname'] == '/test.csv'

# TODO(lesson-mocking) Implement a unit test for the load_csv function
