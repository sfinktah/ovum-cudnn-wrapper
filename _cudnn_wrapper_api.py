from __future__ import annotations
from aiohttp import web
from nodes import NODE_CLASS_MAPPINGS
from server import PromptServer
import os
import re
import inspect
# noinspection PyUnresolvedReferences,PyPackageRequirements
import torch

from cudnn_wrapper import (
    convert_to_cudnn_wrapped_inplace,
    is_cudnn_wrapped,
)


@PromptServer.instance.routes.post("/ovum-cudnn-wrapper/cudnn_wrap_request")
async def ovum_cudnn_wrap_request(request: web.Request):
    try:
        data = await request.post()
        class_key = data.get("type")
        if not class_key:
            return web.json_response({"response": False})
        ok = bool(convert_to_cudnn_wrapped_inplace(class_key))
        return web.json_response({"response": ok})
    except Exception:
        return web.json_response({"response": False})


@PromptServer.instance.routes.post("/ovum-cudnn-wrapper/cudnn_wrap_query")
async def ovum_cudnn_wrap_query(request: web.Request):
    try:
        data = await request.post()
        class_key = data.get("type")
        return web.json_response({"response": bool(is_cudnn_wrapped(class_key))})
    except Exception:
        return web.json_response({"response": False})


@PromptServer.instance.routes.post("/ovum-cudnn-wrapper/cudnn_wrap_query_bulk")
async def ovum_cudnn_wrap_query_bulk(request: web.Request):
    # Accepts JSON body: { "types": ["ClassA", "ClassB", ...] }
    try:
        try:
            data = await request.json()
        except Exception:
            data = await request.post()
        types = data.get("types") or []
        if isinstance(types, str):
            types = [t.strip() for t in types.split(",") if t.strip()]
        elif not isinstance(types, (list, tuple)):
            types = []
        result = {}
        for t in types:
            try:
                result[t] = bool(is_cudnn_wrapped(t))
            except Exception:
                result[t] = False
        sorted_result = {k: result[k] for k in sorted(result)}
        return web.json_response({"response": sorted_result})
    except Exception:
        return web.json_response({"response": {}})


@PromptServer.instance.routes.post("/ovum-cudnn-wrapper/cudnn_wrap_init")
async def ovum_cudnn_wrap_init(request: web.Request):
    """
    Auto-convert classes listed in classes_to_cudnn_wrap.txt.
    Supports exact class keys and regex patterns.
    Regex syntax:
      - Lines starting with "re:" treat the remainder as a Python regex.
        Flags can be specified as: re:i:pattern (letters among imsxauL), or using inline (?i) etc.
      - Lines wrapped in slashes like "/.../" are also treated as regex. Trailing flags are allowed: /pattern/imx
    Other non-empty, non-comment lines are treated as exact NODE_CLASS_MAPPINGS keys.

    Exclusions:
      - Any line prefixed with '-' denotes an exclusion and follows the same syntax as inclusions
        (e.g., -VAEDecode, -re:^VAE, -/vae.*/i). Exclusions are applied after inclusions and take priority.
    """
    try:
        root_cfg = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..', 'classes_to_cudnn_wrap.txt'))
        module_cfg = os.path.join(os.path.dirname(__file__), 'classes_to_cudnn_wrap.txt')
        cfg_path = root_cfg if os.path.isfile(root_cfg) else module_cfg
        print(f"[CUDNNWrapper] reading config from {cfg_path}")
        exact_keys = []
        regex_patterns = []  # list of tuples (pattern_str_display, compiled_pattern)
        exact_excludes = []
        regex_excludes = []  # list of tuples (pattern_str_display, compiled_pattern)

        def _flags_from_letters(letters: str) -> int:
            flag_map = {
                'i': re.IGNORECASE,
                'm': re.MULTILINE,
                's': re.DOTALL,
                'x': re.VERBOSE,
                'a': re.ASCII,
                'u': re.UNICODE,
                'L': re.LOCALE,
            }
            flags = 0
            for ch in letters:
                if ch in flag_map:
                    flags |= flag_map[ch]
            return flags

        with open(cfg_path, 'r', encoding='utf-8', errors='ignore') as f:
            for raw in f.readlines():
                if raw.startswith('#'):
                    continue
                line = raw.strip()
                if not line:
                    continue
                # Handle exclusions prefixed with '-'
                is_exclude = line.startswith('-')
                if is_exclude:
                    line = line[1:].strip()
                    if not line:
                        continue
                pat_disp = None
                if line.startswith('re:'):
                    rest = line[3:].strip()
                    flags_val = 0
                    # Support re:flags:pattern where flags are letters [imsxauL]
                    if ':' in rest:
                        maybe_flags, _, remainder = rest.partition(':')
                        if maybe_flags and all(c.isalpha() for c in maybe_flags):
                            flags_val = _flags_from_letters(maybe_flags)
                            pat = remainder
                            pat_disp = f"re:{maybe_flags}:{pat}"
                        else:
                            pat = rest
                    else:
                        pat = rest
                    # Also support re:/pattern/flags
                    if pat.startswith('/') and pat.endswith('/') and len(pat) >= 2:
                        # no trailing flags here because of endswith('/'), treat as pure /.../
                        inner = pat[1:-1]
                        pat_disp = pat_disp or f'/{inner}/'
                        try:
                            (regex_excludes if is_exclude else regex_patterns).append((pat_disp, re.compile(inner, flags_val)))
                        except re.error as e:
                            print(f"CUDNNWrapper: invalid regex {pat_disp}: {e}")
                    else:
                        pat_disp = pat_disp or f're:{pat}'
                        try:
                            (regex_excludes if is_exclude else regex_patterns).append((pat_disp, re.compile(pat, flags_val)))
                        except re.error as e:
                            print(f"CUDNNWrapper: invalid regex {pat_disp}: {e}")
                elif line.startswith('/') and len(line) >= 2:
                    # Support /pattern/flags (flags optional)
                    last_slash = line.rfind('/')
                    if last_slash == 0:
                        # malformed like '/...'
                        continue
                    pat = line[1:last_slash]
                    trailing = line[last_slash+1:]
                    flags_val = _flags_from_letters(trailing)
                    pat_disp = f'/{pat}/{trailing}' if trailing else f'/{pat}/'
                    try:
                        (regex_excludes if is_exclude else regex_patterns).append((pat_disp, re.compile(pat, flags_val)))
                    except re.error as e:
                        print(f"CUDNNWrapper: invalid regex {pat_disp}: {e}")
                else:
                    if is_exclude:
                        exact_excludes.append(line)
                    else:
                        exact_keys.append(line)
        # Build a unique processing list to avoid wrapping the same key multiple times
        unique_targets = []
        seen = set()

        # Add exact matches first (preserve order)
        for key in exact_keys:
            if key not in seen:
                unique_targets.append((key, None))
                seen.add(key)

        # Collect regex matches, skipping keys already scheduled
        mapping_keys = list(NODE_CLASS_MAPPINGS.keys())
        for disp, pat in regex_patterns:
            matched_any = False
            for key in mapping_keys:
                try:
                    if pat.search(key):
                        matched_any = True
                        if key not in seen:
                            unique_targets.append((key, disp))
                            seen.add(key)
                except Exception as e:
                    print(f"CUDNNWrapper: Error testing regex {disp} on '{key}': {type(e).__name__}")
            if not matched_any:
                print(f"CUDNNWrapper: regex {disp} matched no classes")

        # Build exclusion set (priority over inclusions)
        excluded = set(exact_excludes)
        for disp, pat in regex_excludes:
            matched_any = False
            for key in mapping_keys:
                try:
                    if pat.search(key):
                        matched_any = True
                        excluded.add(key)
                except Exception:
                    print(f"CUDNNWrapper: Error testing exclude regex {disp} on '{key}'")
            if not matched_any:
                # print(f"CUDNNWrapper: exclude regex {disp} matched no classes")
                pass

        if excluded:
            # Filter unique_targets by exclusions
            before = len(unique_targets)
            unique_targets = [(k, o) for (k, o) in unique_targets if k not in excluded]
            removed = before - len(unique_targets)
            if removed:
                print(f"CUDNNWrapper: excluded {removed} target(s) due to exclusions")

        # Process each key at most once
        for key, origin in unique_targets:
            try:
                wrapped = bool(convert_to_cudnn_wrapped_inplace(key))
                if wrapped:
                    if origin is None:
                        print(f"CUDNNWrapper: Wrapped '{key}'")
                    else:
                        print(f"CUDNNWrapper: Wrapped via regex {origin} -> '{key}'")
                elif not is_cudnn_wrapped(key):
                    if origin is None:
                        print(f"CUDNNWrapper: Failed wrapping '{key}'")
                    else:
                        print(f"CUDNNWrapper: Failed wrapping via regex {origin} -> '{key}'")
            except KeyError:
                if origin is None:
                    print(f"CUDNNWrapper: '{key}' not found to wrap")
                else:
                    print(f"CUDNNWrapper: regex target '{key}' from {origin} not found to wrap")
            except Exception as e:
                if origin is None:
                    print(f"CUDNNWrapper: Failed to wrap '{key}' because {type(e).__name__}")
                else:
                    print(f"CUDNNWrapper: Failed regex-wrap '{key}' for {origin} because {type(e).__name__}")
    except Exception:
        print("CUDNNWrapper: problem reading classes_to_cudnn_wrap.txt")
    return web.json_response({"response": True})

@PromptServer.instance.routes.get('/ovum-cudnn-wrapper/cudnn-status')
async def status(d):
    try:
        # Detect AMD-like environment (AMD HIP build or ZLUDA) similar to cudnn_wrapper logic
        amd_like = False
        try:
            vstr = ""
            try:
                if torch.cuda.is_available():
                    vstr = torch.cuda.get_device_name(0)
            except Exception:
                pass
            try:
                hip = getattr(getattr(torch, "version", object()), "hip", None)
                if hip:
                    vstr = (vstr + " AMD ").strip()
            except Exception:
                pass
            import os
            vlow = vstr.lower()
            if ("amd" in vlow) or ("zluda" in vlow) or os.environ.get("ZLUDA") or os.environ.get("ZLUDA_ROOT"):
                amd_like = True
        except Exception:
            amd_like = False
        return web.json_response({
            "torch.backends.cudnn.enabled": torch.backends.cudnn.enabled,
            "torch.backends.cudnn.benchmark": torch.backends.cudnn.benchmark,
            "amd_like": amd_like
        })
    except Exception as e:
        return web.json_response({"error": True, "message": str(e)})


# -------------------- Debug helpers and routes --------------------

def _flags_from_letters_debug(letters: str) -> int:
    flag_map = {
        'i': re.IGNORECASE,
        'm': re.MULTILINE,
        's': re.DOTALL,
        'x': re.VERBOSE,
        'a': re.ASCII,
        'u': re.UNICODE,
        'L': re.LOCALE,
    }
    flags = 0
    for ch in letters or "":
        if ch in flag_map:
            flags |= flag_map[ch]
    return flags


def _node_info_for_key(key: str) -> dict:
    info = {"key": key, "exists": key in NODE_CLASS_MAPPINGS}
    if not info["exists"]:
        return info
    cls = NODE_CLASS_MAPPINGS.get(key)
    info["class_repr"] = repr(cls)
    info["class_name"] = getattr(cls, "__name__", None)
    info["module"] = getattr(cls, "__module__", None)
    info["qualname"] = getattr(cls, "__qualname__", None)
    info["category"] = getattr(cls, "CATEGORY", None)
    info["function"] = getattr(cls, "FUNCTION", None)
    try:
        src_file = inspect.getsourcefile(cls) or inspect.getfile(cls)
        info["file"] = src_file
        try:
            src_lines, start_line = inspect.getsourcelines(cls)
            info["line"] = start_line
            info["lines_of_code"] = len(src_lines)
        except Exception:
            pass
    except Exception:
        pass
    try:
        info["is_wrapped"] = bool(is_cudnn_wrapped(key))
    except Exception:
        info["is_wrapped"] = False
    return info


@PromptServer.instance.routes.get("/ovum-cudnn-wrapper/debug/node-mappings")
async def debug_node_mappings(request: web.Request):
    """
    Query params:
      - q: substring filter (case-insensitive)
      - re: regex pattern to filter keys
      - flags: regex flags letters among [imsxauL]
      - details: "1" to include detailed class info
      - limit: integer to limit returned items (after sorting/filtering)
    """
    try:
        params = request.rel_url.query
        substr = (params.get("q") or "").strip()
        pattern = (params.get("re") or "").strip()
        flags_letters = (params.get("flags") or "").strip()
        details = (params.get("details") or "0").strip() == "1"
        limit_str = (params.get("limit") or "").strip()
        limit = int(limit_str) if limit_str.isdigit() else None

        keys = sorted(NODE_CLASS_MAPPINGS.keys())
        if substr:
            s = substr.lower()
            keys = [k for k in keys if s in k.lower()]

        if pattern:
            try:
                pat = re.compile(pattern, _flags_from_letters_debug(flags_letters))
                keys = [k for k in keys if pat.search(k)]
            except re.error as e:
                return web.json_response({"error": True, "message": f"Invalid regex: {e}"})
        total = len(keys)
        if limit is not None:
            keys = keys[:max(0, limit)]
        if details:
            data = [_node_info_for_key(k) for k in keys]
        else:
            data = keys
        return web.json_response({
            "count": total,
            "returned": len(keys),
            "filters": {"q": substr, "re": pattern, "flags": flags_letters, "details": details, "limit": limit},
            "data": data
        })
    except Exception as e:
        return web.json_response({"error": True, "message": str(e)})


@PromptServer.instance.routes.get("/ovum-cudnn-wrapper/debug/node-details")
async def debug_node_details(request: web.Request):
    """
    Query params:
      - key: exact NODE_CLASS_MAPPINGS key
    """
    try:
        key = (request.rel_url.query.get("key") or "").strip()
        if not key:
            return web.json_response({"error": True, "message": "Missing 'key' parameter"})
        info = _node_info_for_key(key)
        if not info.get("exists"):
            # Provide basic suggestions to help spotting typos
            suggestions = []
            k_lower = key.lower()
            for k in sorted(NODE_CLASS_MAPPINGS.keys()):
                if k_lower in k.lower() or k.lower() in k_lower:
                    suggestions.append(k)
                if len(suggestions) >= 20:
                    break
            return web.json_response({"exists": False, "key": key, "suggestions": suggestions})
        return web.json_response(info)
    except Exception as e:
        return web.json_response({"error": True, "message": str(e)})


@PromptServer.instance.routes.get("/ovum-cudnn-wrapper/debug/config")
async def debug_config(request: web.Request):
    """
    Parses classes_to_cudnn_wrap.txt and previews how it would match current NODE_CLASS_MAPPINGS.
    Returns exact targets, regex entries (display), and which keys each regex would match.
    """
    try:
        # Resolve config path similarly to init
        root_cfg = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'classes_to_cudnn_wrap.txt'))
        module_cfg = os.path.join(os.path.dirname(__file__), 'classes_to_cudnn_wrap.txt')
        cfg_path = root_cfg if os.path.isfile(root_cfg) else module_cfg

        exact_keys = []
        regex_entries = []  # list of tuples (display, compiled)
        raw_lines = []

        def add_regex(display: str, pattern: str, flags_val: int):
            try:
                regex_entries.append((display, re.compile(pattern, flags_val)))
            except re.error as e:
                regex_entries.append((f"{display} [INVALID: {e}]", None))

        with open(cfg_path, 'r', encoding='utf-8', errors='ignore') as f:
            for raw in f.readlines():
                raw_lines.append(raw.rstrip("\n"))
                if raw.startswith('#'):
                    continue
                line = raw.strip()
                if not line:
                    continue
                if line.startswith('re:'):
                    rest = line[3:].strip()
                    flags_val = 0
                    if ':' in rest:
                        maybe_flags, _, remainder = rest.partition(':')
                        if maybe_flags and all(c.isalpha() for c in maybe_flags):
                            flags_val = _flags_from_letters_debug(maybe_flags)
                            pat = remainder
                            disp = f"re:{maybe_flags}:{pat}"
                        else:
                            pat = rest
                            disp = f"re:{pat}"
                    else:
                        pat = rest
                        disp = f"re:{pat}"
                    if pat.startswith('/') and pat.endswith('/') and len(pat) >= 2:
                        inner = pat[1:-1]
                        disp = f'/{inner}/'
                        add_regex(disp, inner, flags_val)
                    else:
                        add_regex(disp, pat, flags_val)
                elif line.startswith('/') and len(line) >= 2:
                    last_slash = line.rfind('/')
                    if last_slash == 0:
                        continue
                    pat = line[1:last_slash]
                    trailing = line[last_slash+1:]
                    flags_val = _flags_from_letters_debug(trailing)
                    disp = f'/{pat}/{trailing}' if trailing else f'/{pat}/'
                    add_regex(disp, pat, flags_val)
                else:
                    exact_keys.append(line)

        mapping_keys = sorted(NODE_CLASS_MAPPINGS.keys())
        regex_matches = {}
        for disp, comp in regex_entries:
            if comp is None:
                regex_matches[disp] = {"error": True, "matches": []}
                continue
            hits = []
            for key in mapping_keys:
                try:
                    if comp.search(key):
                        hits.append(key)
                except Exception:
                    pass
            regex_matches[disp] = {"error": False, "count": len(hits), "matches": hits}

        missing_exact = [k for k in exact_keys if k not in NODE_CLASS_MAPPINGS]

        return web.json_response({
            "config_path": cfg_path,
            "raw_lines": raw_lines,
            "exact_keys": exact_keys,
            "missing_exact": missing_exact,
            "regex_count": len(regex_entries),
            "regex_matches": regex_matches,
            "mapping_count": len(mapping_keys)
        })
    except FileNotFoundError:
        return web.json_response({"error": True, "message": "classes_to_cudnn_wrap.txt not found"})
    except Exception as e:
        return web.json_response({"error": True, "message": str(e)})
