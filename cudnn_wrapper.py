from typing import Callable, Optional, Type, Any
import types
import inspect

# noinspection PyPackageRequirements
import torch
# noinspection PyPackageRequirements
from nodes import NODE_CLASS_MAPPINGS

_FLAG = '_is_cudnn_wrapped'
_FN_PREFIX = '_cudnn_wrapped_'
_CATEGORY_DEFAULT = 'ovum/cudnn_wrapped_nodes'

essential_doc = (
    "Temporarily disables torch.backends.cudnn.* during node execution for AMD/ZLUDA users, "
    "restoring the previous settings afterward, and printing the same console output format "
    "as CUDNNToggleOvum."
)


def _detect_gpu_vendor_str() -> str:
    """
    Lightweight detection similar to AmdNvidiaIfElseOvum._detect_gpu_vendor
    Returns a lower-signal string that may contain vendor hints.
    """
    try:
        s = ""
        try:
            if torch.cuda.is_available():
                s = torch.cuda.get_device_name(0)
        except Exception:
            pass
        # ZLUDA env hint
        import os
        if os.environ.get("ZLUDA", "") or os.environ.get("ZLUDA_ROOT", ""):
            s = (s + " ZLUDA").strip()
        return s
    except Exception:
        return ""

def is_hip():
    if torch.version.hip:
        return True
    return False

def _is_amd_like() -> bool:
    vstr = _detect_gpu_vendor_str()
    vlow = vstr.lower()
    # mirror logic used by AmdNvidiaIfElseOvum
    return any(x in vlow for x in ("radeon", "amd ", "zluda")) or is_hip

def _print_cudnn_change(target_value: bool, prev_enabled: bool):
    # Match CUDNNToggleOvum console output for 'enabled' flag
    if target_value != prev_enabled:
        print(f"[OVUM_CUDDN_TOGGLE] torch.backends.cudnn.enabled set to {target_value} (was {prev_enabled})")
    else:
        print(f"[OVUM_CUDDN_TOGGLE] torch.backends.cudnn.enabled still set to {target_value}")


def _extract_extra_options_from_call(callable_fn: Callable, node, args, kwargs) -> dict:
    """Mimic VideoCombine's way of pulling extra options from extra_pnginfo.
    It tries to bind args/kwargs to the callable's signature and then read
    the 'extra_pnginfo' parameter, if present, and extract workflow.extra.
    """
    try:
        sig = inspect.signature(callable_fn)
        try:
            bound = sig.bind_partial(node, *args, **kwargs)
        except TypeError:
            bound = sig.bind_partial(*args, **kwargs)
        extra_pnginfo = bound.arguments.get('extra_pnginfo', None)
        if extra_pnginfo is not None:
            return extra_pnginfo.get('workflow', {}).get('extra', {}) or {}
    except Exception:
        pass
    # Fallback: check kwargs directly
    try:
        extra_pnginfo = kwargs.get('extra_pnginfo', None)
        if extra_pnginfo is not None:
            return extra_pnginfo.get('workflow', {}).get('extra', {}) or {}
    except Exception:
        pass
    return {}


def _wrap_function_with_cudnn_disable(callable_fn: Callable) -> Callable:
    def wrapped(node, *args, **kwargs):
        # Check workflow extra options toggle first. If disabled, do not change cudnn at all.
        extra_options = _extract_extra_options_from_call(callable_fn, node, args, kwargs)
        try:
            cur_enabled = torch.backends.cudnn.enabled
        except Exception:
            cur_enabled = False
        if extra_options.get('ovum.cudnn-wrapper-enabled') == False:
            print(
                f"[OVUM_CUDDN_TOGGLE] Disabled by global setting: ovum.cudnn-wrapper-enabled=False; cudnn settings unchanged (enabled={cur_enabled})"
            )
            return callable_fn(node, *args, **kwargs)

        if not _is_amd_like():
            try:
                prev_enabled = torch.backends.cudnn.enabled
            except Exception:
                prev_enabled = False
            print(
                f"[OVUM_CUDDN_TOGGLE] AMD GPU not detected; cudnn settings unchanged "
                f"(enabled={prev_enabled})"
            )
            return callable_fn(node, *args, **kwargs)

        # Save current state
        prev_enabled = torch.backends.cudnn.enabled

        # Disable while running
        torch.backends.cudnn.enabled = False
        _print_cudnn_change(False, prev_enabled)

        try:
            return callable_fn(node, *args, **kwargs)
        finally:
            # Restore
            cur_enabled = torch.backends.cudnn.enabled
            torch.backends.cudnn.enabled = prev_enabled
            # Print messages reflecting the change back to original
            _print_cudnn_change(prev_enabled, cur_enabled)

    return wrapped


def convert_to_cudnn_wrapped_inplace(class_key: str) -> bool:
    """Replace NODE_CLASS_MAPPINGS[class_key] with a cudnn-wrapped version in-place."""
    if getattr(class_key, _FLAG, False):
        return False
    class_to_wrap = NODE_CLASS_MAPPINGS[class_key]
    converted = create_cudnn_wrapped_node(class_to_wrap, new_category=class_to_wrap.CATEGORY)
    if converted:
        try:
            desc = getattr(converted, 'DESCRIPTION', None)
            if not desc:
                setattr(converted, 'DESCRIPTION', essential_doc)
        except Exception:
            pass
        NODE_CLASS_MAPPINGS[class_key] = converted
        return True
    return False


def is_cudnn_wrapped(class_key: str) -> bool:
    cls = NODE_CLASS_MAPPINGS[class_key]
    return getattr(cls, _FLAG, False)


def create_cudnn_wrapped_node(class_to_wrap: Type,
                              new_name: Optional[str] = None,
                              new_category: Optional[str] = None) -> Optional[Type]:
    """
    Returns a new Type which subclasses `class_to_wrap` and whose FUNCTION method
    is wrapped to temporarily disable cudnn for AMD users during the call, then restore.
    """
    if getattr(class_to_wrap, _FLAG, False):
        # print(f"[CUDNNWrapper] {class_to_wrap.__name__} already wrapped")
        # already wrapped
        return None

    new_name = new_name or f"cudnn_wrapped_{class_to_wrap.__name__}"
    if new_name in NODE_CLASS_MAPPINGS:
        print(
            f"[CUDNNWrapper] '{class_to_wrap.__name__}' => '{new_name}' name collision; avoid duplicate registration (already wrapped?)")
        # name collision; avoid duplicate registration
        return None

    new_class = types.new_class(new_name, (class_to_wrap,))
    function_name = getattr(new_class, 'FUNCTION')
    original_fn = getattr(new_class, function_name)
    wrapped_function = _wrap_function_with_cudnn_disable(original_fn)

    setattr(new_class, f"{_FN_PREFIX}{function_name}", wrapped_function)
    setattr(new_class, 'FUNCTION', f"{_FN_PREFIX}{function_name}")
    setattr(new_class, 'CATEGORY', new_category or _CATEGORY_DEFAULT)

    # Ensure INPUT_TYPES has a hidden.extra_pnginfo input so workflow extras are available
    orig_input_types = getattr(new_class, 'INPUT_TYPES', None)

    def _ensure_hidden_extra_pnginfo(dct: Any):
        try:
            if not isinstance(dct, dict):
                return dct
            hidden = dct.get('hidden', None)
            if hidden is None:
                dct['hidden'] = {"extra_pnginfo": "EXTRA_PNGINFO"}
            elif isinstance(hidden, dict):
                if 'extra_pnginfo' not in hidden:
                    hidden['extra_pnginfo'] = "EXTRA_PNGINFO"
            # If hidden is not a dict (e.g., a special helper), we leave it untouched
            return dct
        except Exception:
            return dct

    if callable(orig_input_types):
        original_input_types_callable = orig_input_types
        def _cudnn_wrapped_INPUT_TYPES(cls):
            try:
                base = original_input_types_callable()
            except TypeError:
                # Some classmethods may expect the class parameter
                base = original_input_types_callable(cls)
            return _ensure_hidden_extra_pnginfo(base)
        setattr(new_class, 'INPUT_TYPES', classmethod(_cudnn_wrapped_INPUT_TYPES))
    else:
        # Provide a minimal INPUT_TYPES if none exists
        def _cudnn_wrapped_INPUT_TYPES(cls):
            return {"required": {}, "hidden": {"extra_pnginfo": "EXTRA_PNGINFO"}}
        setattr(new_class, 'INPUT_TYPES', classmethod(_cudnn_wrapped_INPUT_TYPES))

    setattr(new_class, _FLAG, True)

    return new_class
