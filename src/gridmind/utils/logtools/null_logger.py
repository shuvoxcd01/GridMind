class NullWriter:
    """No-op logger used when write_summary=False or torch is unavailable."""

    def add_scalar(self, *args, **kwargs):
        pass

    def close(self):
        pass
