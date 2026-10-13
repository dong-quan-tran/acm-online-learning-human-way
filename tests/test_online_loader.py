from src.data import get_online_loader


def test_online_loader_size():
    loader = get_online_loader(seed=42)

    assert len(loader.dataset) == 10000


def test_online_loader_labels():
    loader = get_online_loader(seed=42)

    labels = []

    for _, batch_labels in loader:
        labels.extend(batch_labels.tolist())

    assert set(labels) == set(range(20))


def test_same_seed_same_order():
    loader1 = get_online_loader(seed=42)
    loader2 = get_online_loader(seed=42)

    order1 = loader1.dataset.indices
    order2 = loader2.dataset.indices

    assert order1 == order2


def test_different_seed_different_order():
    loader1 = get_online_loader(seed=42)
    loader2 = get_online_loader(seed=43)

    order1 = loader1.dataset.indices
    order2 = loader2.dataset.indices

    assert order1 != order2


def test_batch_shape():
    loader = get_online_loader(seed=42)

    images, labels = next(iter(loader))

    assert images.shape == (32, 3, 32, 32)
    assert labels.shape == (32,)


def test_final_batch_not_dropped():
    loader = get_online_loader(seed=42)

    total_examples = 0

    for images, _ in loader:
        total_examples += len(images)

    assert total_examples == 10000
