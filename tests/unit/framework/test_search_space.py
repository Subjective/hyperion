"""Tests for search space primitives."""

from hyperion.framework.search_space import Bool, Choice, Float, Int, sample_space


def test_uniform_sampling():
    """Uniform should sample float values in range."""
    space = Float(min=0.0, max=1.0)

    # Sample multiple times
    samples = [sample_space(space) for _ in range(100)]

    # All should be in range
    assert all(0.0 <= s <= 1.0 for s in samples)
    # Should have variety (very unlikely to get same value twice)
    assert len(set(samples)) > 50


def test_uniform_log_sampling():
    """Uniform with log=True should sample in log space."""
    space = Float(min=0.001, max=1.0, log=True)

    samples = [sample_space(space) for _ in range(100)]

    # All should be in range
    assert all(0.001 <= s <= 1.0 for s in samples)
    # Should have more samples near the lower end in log space
    below_0_1 = sum(1 for s in samples if s < 0.1)
    # In log space, we expect roughly equal distribution
    assert below_0_1 > 20  # Should have substantial samples in lower range


def test_int_sampling():
    """Int should sample integer values in range."""
    space = Int(min=1, max=10)

    samples = [sample_space(space) for _ in range(100)]

    # All should be integers in range
    assert all(isinstance(s, int) for s in samples)
    assert all(1 <= s <= 10 for s in samples)
    # Should cover the range
    assert len(set(samples)) >= 8  # Should hit most values


def test_choice_sampling():
    """Choice should sample from options."""
    space = Choice(options=["sgd", "adam", "rmsprop"])

    samples = [sample_space(space) for _ in range(100)]

    # All should be from options
    assert all(s in ["sgd", "adam", "rmsprop"] for s in samples)
    # Should sample all options
    assert set(samples) == {"sgd", "adam", "rmsprop"}


def test_bool_sampling():
    """Bool should sample True/False."""
    space = Bool()

    samples = [sample_space(space) for _ in range(100)]

    # Should be booleans
    assert all(isinstance(s, bool) for s in samples)
    # Should have both values
    assert True in samples
    assert False in samples
    # Should be roughly balanced
    true_count = sum(samples)
    assert 30 < true_count < 70  # Roughly 50/50


def test_nested_space_sampling():
    """Should sample from nested spaces."""
    space = {
        "lr": Float(0.001, 0.1, log=True),
        "batch_size": Choice([16, 32, 64, 128]),
        "optimizer": Choice(["sgd", "adam"]),
        "use_dropout": Bool(),
        "layers": Int(1, 5),
    }

    params = sample_space(space)

    # Check structure
    assert "lr" in params
    assert "batch_size" in params
    assert "optimizer" in params
    assert "use_dropout" in params
    assert "layers" in params

    # Check types and ranges
    assert 0.001 <= params["lr"] <= 0.1
    assert params["batch_size"] in [16, 32, 64, 128]
    assert params["optimizer"] in ["sgd", "adam"]
    assert isinstance(params["use_dropout"], bool)
    assert 1 <= params["layers"] <= 5
    assert isinstance(params["layers"], int)


def test_sample_space_with_plain_values():
    """Should pass through plain values unchanged."""
    space = {"fixed_value": 42, "fixed_string": "constant", "sampled": Float(0, 1)}

    params = sample_space(space)

    # Fixed values unchanged
    assert params["fixed_value"] == 42
    assert params["fixed_string"] == "constant"
    # Sampled value in range
    assert 0 <= params["sampled"] <= 1
