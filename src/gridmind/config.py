_save_dir: str | None = None


def set_save_dir(path: str) -> None:
    """Set the project-wide default directory for saving policies and TensorBoard logs."""
    global _save_dir
    _save_dir = path


def get_save_dir() -> str | None:
    return _save_dir
