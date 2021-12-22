from importlib import import_module
from inspect import isclass
from pathlib import Path
from pkgutil import iter_modules

from .base_feature import FeatureType

# iterate through the modules in the current package
features = []
for (_, module_name, _) in iter_modules([Path(__file__).resolve().parent]):
    # import the module and iterate through its attributes
    module = import_module(f"{__name__}.{module_name}")
    for attribute_name in dir(module):
        attribute = getattr(module, attribute_name)

        if (
            isclass(attribute)
            and issubclass(attribute, FeatureType)
            and attribute_name != "FeatureType"
        ):
            # Add the class to this package's variables
            globals()[attribute_name] = attribute
            features.append(attribute_name)
