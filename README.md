# cuDNN Wrapper for AMD VAE Workloads

Dynamically wrap ComfyUI node types so that, when running on AMD-like GPUs, cuDNN is temporarily disabled during
VAE-related encode/decode operations. On NVIDIA GPUs the wrapper goes inert and will not modify cuDNN, so it’s safe to
enable everywhere.

This custom node set doesn’t add new nodes; instead it lets you add a cuDNN-disabling wrapper to existing node types 
(primarily useful for VAE encoding/decoding on AMD).

## How to use

- Right‑click on a node and choose: `Disable cuDNN (wrapper)`.
- The node’s title will not change. Instead, an AMD logo is added to the node’s title bar:
  - Green: AMD detected
  - Blue: AMD detected and cuDNN is currently disabled (or the node is running)
  - Any other color: no AMD detected
  Hover the logo to see a status tooltip.
- Until you restart the server, all instances of this node type (in all workflows) will be wrapped.
- When you restart the server, the wrapping is removed and you’ll need to re‑add it (see below to make it automatic).

## Important caveats

Please read these before reporting issues:

- The modification happens in the backend and applies to every instance of that node type until the server is restarted.
- If you add wrapping to (say) a VAE Decode node, all VAE Decode nodes become wrapped across all workflows for this
  server session.
- If execution stops or is interrupted while a wrapped node is running, cuDNN may remain disabled (because the node
  didn’t get a chance to restore cuDNN settings). It’s recommended to add a small “enable cuDNN” step near the start of
  your workflow to ensure the environment is in a known good state at the beginning of runs.
- On NVIDIA GPUs (or when AMD-like hardware is not detected), wrapping becomes a no‑op and will not change cuDNN.

## Picking node types to always wrap

If you want particular node types to always be wrapped, add them to `classes_to_cudnn_wrap.txt`.
When you reload the webpage, all node types named in this file will have wrapping added if they haven’t already.

Recommendation: if you plan to customize this list, copy `classes_to_cudnn_wrap.txt` to your ComfyUI root directory and
edit it there. The file ships with sensible defaults that should work for most setups, so you may not need to change
anything.