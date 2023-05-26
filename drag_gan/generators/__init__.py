from .base import BaseGenerator  # noqa
from .styleganv2 import StyleGANv2Generator  # noqa

REGISTERED_GENERATORS = {StyleGANv2Generator.__name__: StyleGANv2Generator}
