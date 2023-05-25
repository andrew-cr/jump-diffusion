import itertools as it
from .image import *
from .video import *
from .embedded import *
from .matrix_factorization import *


datasets_to_kwargs = {
    d.__name__: kwargs for d, kwargs in it.chain(
        image_datasets_to_kwargs.items(),
        video_datasets_to_kwargs.items(),
        embedded_datasets_to_kwargs.items(),
        matrix_factorization_datasets_to_kwargs.items(),
    )
}
kwargs_gettable_from_dataset = {
    d.__name__: kwargs for d, kwargs in it.chain(
        image_kwargs_gettable_from_dataset.items(),
        video_kwargs_gettable_from_dataset.items(),
        embedded_kwargs_gettable_from_dataset.items(),
        matrix_factorization_kwargs_gettable_from_dataset.items(),
    )
}