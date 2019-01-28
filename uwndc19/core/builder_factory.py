from importlib import import_module


def create_builder(model_type: str, config: dict):
    package_name = __package__.split('.')[0]
    builder_class = getattr(import_module('%s.models.%s.builder' % (package_name, model_type)), 'Builder')
    return builder_class(config)
