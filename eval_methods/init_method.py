import importlib

def create_method(args, *argv):
    name_class_method = args.name_class
    module = importlib.import_module('.'+name_class_method, package='eval_methods')
    constructor_method = getattr(module, name_class_method)
    method = constructor_method(args, argv)
    return method