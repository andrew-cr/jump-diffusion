import itertools as it
from .image import *
from .video import *


datasets_to_kwargs = {
    d.__name__: kwargs for d, kwargs in it.chain(
        image_datasets_to_kwargs.items(),
        video_datasets_to_kwargs.items(),
    )
}
kwargs_gettable_from_dataset = {
    d.__name__: kwargs for d, kwargs in it.chain(
        image_kwargs_gettable_from_dataset.items(),
        video_kwargs_gettable_from_dataset.items(),
    )
}