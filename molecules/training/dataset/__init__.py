import itertools as it
from .qm9 import QM9Dataset, QM9GraphicalStructure, qm9_datasets_to_kwargs, qm9_kwargs_gettable_from_dataset


datasets_to_kwargs = {
    d.__name__: kwargs for d, kwargs in it.chain(
        qm9_datasets_to_kwargs.items(),
    )
}
kwargs_gettable_from_dataset = {
    d.__name__: kwargs for d, kwargs in it.chain(
        qm9_kwargs_gettable_from_dataset.items(),
    )
}