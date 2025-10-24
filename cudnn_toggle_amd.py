import torch

from cudnn_wrapper import _is_amd_like


class AnyType(str):
    """A special class that is always equal in not equal comparisons. Credit to pythongosssss"""

    def __ne__(self, __value: object) -> bool:
        return False


anyType = AnyType("*")


class CUDNNToggleAMD:
    NAME = "cuDNN Toggle (AMD-aware)"
    DESCRIPTION = (
        """
        Toggle torch.backends.cudnn.enabled ON or OFF,
        but only applies the change when an AMD-like environment is detected (AMD/ZLUDA/HIP).
        On NVIDIA (or when AMD is not detected), this node will not modify cuDNN and will
        print a notice. If you care about preserving the original state of cuDNN, connect
        the prev_cudnn output of the disabling node to the enable_cudnn input of the enabling
        node. To ensure correct behavior, connect the any_input input and any_output output
        of the node to intercept an IMAGE or LATENT immediately before and after the node you
        wish to affect.
        
        The most common use case is to disable cuDNN for VAE Decode and VAE Encode nodes, which
        can receive significant speedups on AMD-like setups.
        """
    ).strip()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                "any_input": (anyType, {}),
            },
            "required": {
                "enable_cudnn": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = (anyType, "BOOLEAN")
    RETURN_NAMES = ("any_output", "prev_cudnn")
    FUNCTION = "toggle"
    CATEGORY = "ovum/cudnn-wrapper"

    def toggle(self, enable_cudnn, any_input=None):
        # Read previous states (guarded)
        try:
            prev_cudnn = torch.backends.cudnn.enabled
        except Exception:
            prev_cudnn = False

        if not _is_amd_like():
            # Do not modify cuDNN on non-AMD systems
            print(
                f"[OVUM_CUDDN_TOGGLE] AMD GPU not detected; torch.backends.cudnn.enabled still set to {prev_cudnn}"
            )
            return (any_input, prev_cudnn)

        # Apply requested toggle on AMD-like systems
        torch.backends.cudnn.enabled = enable_cudnn

        if enable_cudnn != prev_cudnn:
            print(f"[OVUM_CUDDN_TOGGLE] torch.backends.cudnn.enabled set to {enable_cudnn} (was {prev_cudnn})")
        else:
            print(f"[OVUM_CUDDN_TOGGLE] torch.backends.cudnn.enabled still set to {enable_cudnn}")

        return (any_input, prev_cudnn)


CLAZZES = [CUDNNToggleAMD]
