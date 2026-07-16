__all__ = ["PXDPOModel"]


def __getattr__(name):
    if name == "PXDPOModel":
        from PXdpo.modeling import PXDPOModel

        return PXDPOModel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
