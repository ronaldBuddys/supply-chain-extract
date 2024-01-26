import os


def get_path(*sub_dir):
    return os.path.join(os.path.dirname(__file__), *sub_dir)


def get_parent_path(*sub_dir):
    return os.path.join(os.path.dirname(get_path()), *sub_dir)


def get_configs_path(*sub_dir):
    return get_parent_path("configs", *sub_dir)


def get_data_path(*sub_dir):
    return get_parent_path("data", *sub_dir)


def get_image_path(*sub_dir):
    return get_parent_path("images", *sub_dir)


def get_plots_path(*sub_dir):
    return get_parent_path("plots", *sub_dir)
