from .unke import unkeHyperParams, apply_unke_to_model


UNKE_DICT = {
    "unke": (unkeHyperParams, apply_unke_to_model),
}

ALG_DICT = {
    **UNKE_DICT,
}

__all__ = [
    "UNKE_DICT"
]