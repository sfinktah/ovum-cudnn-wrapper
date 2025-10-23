import {app} from "../../../scripts/app.js";
import {api} from "../../../scripts/api.js";
import {chainCallback} from "../01/utility.js";

// Debounced bulk query machinery for cudnn wrap status
const BULK_QUERY_ROUTE = "/ovum-cudnn-wrapper/cudnn_wrap_query_bulk";
const SINGLE_QUERY_ROUTE = "/ovum-cudnn-wrapper/cudnn_wrap_query";

const wrapStatusCache = new Map(); // type -> boolean
const pendingTypes = new Set(); // Set<string>
const pendingResolvers = new Map(); // type -> Array<(v:boolean)=>void>
let bulkTimer = null;

// Schedule a flush of pending types to the backend
function scheduleBulkFlush() {
    if (bulkTimer != null) return;
    bulkTimer = setTimeout(flushPendingTypes, 0);
}

async function flushPendingTypes() {
    const types = Array.from(pendingTypes);
    pendingTypes.clear();
    bulkTimer = null;

    if (types.length === 0) return;

    // Helper to resolve and cleanup
    const resolveForType = (t, value) => {
        wrapStatusCache.set(t, value);
        const resolvers = pendingResolvers.get(t) || [];
        pendingResolvers.delete(t);
        for (const r of resolvers) {
            try {
                r(value);
            } catch {}
        }
    };

    try {
        // Try bulk first
        const data = await call_server_bulk(types, BULK_QUERY_ROUTE);
        const resp = data?.response;
        if (resp && typeof resp === "object") {
            for (const t of types) {
                resolveForType(t, Boolean(resp[t]));
            }
            return;
        }
        // If response malformed, fall through to single fallback
        throw new Error("Malformed bulk response");
    } catch (_) {
        // Fallback to single queries if bulk route isn't available
        await Promise.all(
            types.map(async (t) => {
                try {
                    const data = await call_server(t, SINGLE_QUERY_ROUTE);
                    resolveForType(t, Boolean(data?.response));
                } catch {
                    resolveForType(t, false);
                }
            })
        );
    }
}

function requestWrapStatus(type) {
    if (!type) return Promise.resolve(false);
    if (wrapStatusCache.has(type)) return Promise.resolve(wrapStatusCache.get(type));

    return new Promise((resolve) => {
        const arr = pendingResolvers.get(type) || [];
        arr.push(resolve);
        pendingResolvers.set(type, arr);
        pendingTypes.add(type);
        scheduleBulkFlush();
    });
}

// UI extension mirroring result-wrapper, but for cudnn wrapper
// Global status of environment
let AMD_LIKE = false;
let CUDNN_ENABLED = true;
let STATUS_TIMER = null;
// Warning state when cuDNN failed to be disabled on AMD
let CUDNN_DISABLE_WARNING = false;
let CUDNN_DISABLE_WARNING_MSG = "";

async function fetch_status() {
    try {
        const res = await api.fetchApi('/ovum-cudnn-wrapper/cudnn-status', { method: 'GET' });
        const json = await res.json();
        if (json && typeof json === 'object') {
            if (typeof json["amd_like"] === 'boolean') AMD_LIKE = json["amd_like"];
            if (typeof json["torch.backends.cudnn.enabled"] === 'boolean') CUDNN_ENABLED = json["torch.backends.cudnn.enabled"];
        }
        // If we are on AMD and cuDNN is confirmed disabled, clear any warning state
        if (AMD_LIKE && CUDNN_ENABLED === false) {
            CUDNN_DISABLE_WARNING = false;
            CUDNN_DISABLE_WARNING_MSG = '';
        }
    } catch {}
}
// def is_amd():
//     global cpu_state
//     if cpu_state == CPUState.GPU:
//         if torch.version.hip:
function drawAmdLogo(ctx, x, y, size, color) {
    // Precise AMD corner logo rendered from the provided SVG path
    // Path from user-provided SVG:
    //   <path d="M 93.813042,137.04914 80.062117,123.29583 h 50.059963 v 50.06181 l -13.75093,-13.75172 v -22.55678 z m -0.01614,2.7559 -14.157324,14.15679 v 19.81703 h 19.814381 l 14.156531,-14.1568 H 93.796902 Z" />
    const pathStr = 'M 93.813042,137.04914 80.062117,123.29583 h 50.059963 v 50.06181 l -13.75093,-13.75172 v -22.55678 z m -0.01614,2.7559 -14.157324,14.15679 v 19.81703 h 19.814381 l 14.156531,-14.1568 H 93.796902 Z';
    // Precomputed bounds (original SVG coordinate space)
    const minX = 79.639578;
    const minY = 123.29583;
    const width = 50.482502;
    const height = 50.48303;

    // Build and cache Path2D from SVG string if supported
    try {
        if (!drawAmdLogo._path) {
            drawAmdLogo._path = new Path2D(pathStr);
        }
    } catch (e) {
        drawAmdLogo._path = null;
    }

    // If Path2D with SVG is not supported, fallback to the simple representation
    if (!drawAmdLogo._path) {
        const t = Math.max(2, size * 0.28); // thickness
        ctx.fillStyle = color;
        ctx.beginPath();
        // Horizontal bar (left -> right)
        ctx.rect(x, y + size - t, size, t);
        // Vertical bar (bottom -> top)
        ctx.rect(x + size - t, y, t, size);
        ctx.fill();
        // Cutout square in inner corner to resemble arrow head
        const cut = t * 1.1;
        ctx.clearRect(x + size - cut, y + size - cut, cut, cut);
        return;
    }

    // Draw scaled/translated precise logo
    ctx.save();
    ctx.fillStyle = color;
    const s = size / Math.max(width, height);
    ctx.translate(x, y);
    ctx.scale(s, s);
    ctx.translate(-minX, -minY);
    ctx.fill(drawAmdLogo._path);
    ctx.restore();
}

function drawNvidiaLogo(ctx, x, y, size) {
    // NVIDIA logo drawn from provided SVG path with white background on the left side band
    // SVG viewBox: 0 0 271.7 179.7
    const pathStr = 'M101.3 53.6V37.4c1.6-.1 3.2-.2 4.8-.2 44.4-1.4 73.5 38.2 73.5 38.2S148.2 119 114.5 119c-4.5 0-8.9-.7-13.1-2.1V67.7c17.3 2.1 20.8 9.7 31.1 27l23.1-19.4s-16.9-22.1-45.3-22.1c-3-.1-6 .1-9 .4m0-53.6v24.2l4.8-.3c61.7-2.1 102 50.6 102 50.6s-46.2 56.2-94.3 56.2c-4.2 0-8.3-.4-12.4-1.1v15c3.4.4 6.9.7 10.3.7 44.8 0 77.2-22.9 108.6-49.9 5.2 4.2 26.5 14.3 30.9 18.7-29.8 25-99.3 45.1-138.7 45.1-3.8 0-7.4-.2-11-.6v21.1h170.2V0H101.3zm0 116.9v12.8c-41.4-7.4-52.9-50.5-52.9-50.5s19.9-22 52.9-25.6v14h-.1c-17.3-2.1-30.9 14.1-30.9 14.1s7.7 27.3 31 35.2M27.8 77.4s24.5-36.2 73.6-40V24.2C47 28.6 0 74.6 0 74.6s26.6 77 101.3 84v-14c-54.8-6.8-73.5-67.2-73.5-67.2z';
    const vw = 271.7, vh = 179.7;

    // Scale so the vertical size matches AMD logo height exactly
    const scale = size / vh; // fit by height
    const dw = vw * scale;
    const dh = size;
    const dx = x + (size - dw) / 2; // horizontally center within square
    const dy = y;

    // Build Path2D if possible
    try {
        if (!drawNvidiaLogo._path) {
            drawNvidiaLogo._path = new Path2D(pathStr);
        }
    } catch (e) {
        drawNvidiaLogo._path = null;
    }

    // Background: white band limited to the logo's vertical extremities (now full height)
    ctx.save();
    // ctx.fillStyle = '#ffffff';
    // ctx.fillRect(x, y, size, size);

    // If Path2D is not supported, draw a simple fallback (green rectangle with diagonal notch)
    if (!drawNvidiaLogo._path) {
        ctx.fillStyle = '#76b900';
        const pad = size * 0.12;
        ctx.fillRect(x + pad, dy + pad, size - pad * 2, dh - pad * 2);
        ctx.fillStyle = '#ffffff';
        ctx.beginPath();
        ctx.moveTo(x + size * 0.35, dy + dh * 0.65);
        ctx.lineTo(x + size * 0.55, dy + dh * 0.45);
        ctx.lineTo(x + size * 0.75, dy + dh * 0.65);
        ctx.closePath();
        ctx.fill();
        ctx.restore();
        return;
    }

    // Draw the scaled logo centered within the square button
    ctx.translate(dx, dy);
    ctx.scale(scale, scale);
    ctx.fillStyle = '#76b900';
    ctx.fill(drawNvidiaLogo._path);
    ctx.restore();
}

function roundedRect(ctx, x, y, w, h, r) {
    if (ctx.roundRect) {
        ctx.beginPath();
        ctx.roundRect(x, y, w, h, r);
        return;
    }
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.arcTo(x + w, y, x + w, y + h, r);
    ctx.arcTo(x + w, y + h, x, y + h, r);
    ctx.arcTo(x, y + h, x, y, r);
    ctx.arcTo(x, y, x + w, y, r);
}

app.registerExtension({
    name: "ovum.cudnn_wrapper",

    setup() {
        // Re-init on websocket reconnection
        function onSocketOpen() {
            try { call_server(null, "/ovum-cudnn-wrapper/cudnn_wrap_init"); } catch (e) {}
            try { fetch_status(); } catch (e) {}
            if (!STATUS_TIMER) {
                STATUS_TIMER = setInterval(fetch_status, 60000);
            }
        }
        // Clear status polling when socket disconnects/closes
        function onSocketClose() {
            if (STATUS_TIMER) {
                clearInterval(STATUS_TIMER);
                STATUS_TIMER = null;
            }
        }
        // When a node starts executing, verify cuDNN disabled state for wrapped nodes
        function onExecuting(e) {
            const id = e?.detail;
            const g = app?.graph;
            let node = null;
            try {
                if (g?.getNodeById) node = g.getNodeById(id);
                else if (g?._nodes_by_id) node = g._nodes_by_id[id];
                else if (Array.isArray(g?._nodes)) node = g._nodes.find(n => n?.id === id);
            } catch {}
            if (node && node._is_cudnn_wrapped) {
                setTimeout(async () => {
                    try { await fetch_status(); } catch {}
                    if (AMD_LIKE && CUDNN_ENABLED !== false) {
                        CUDNN_DISABLE_WARNING = true;
                        CUDNN_DISABLE_WARNING_MSG = 'Warning: cuDNN was unable to be disabled';
                    } else {
                        CUDNN_DISABLE_WARNING = false;
                        CUDNN_DISABLE_WARNING_MSG = '';
                    }
                    try { app.graph?.canvas?.setDirty?.(true, true); } catch {}
                }, 500);
            } else {
                // Reset warning when a non-wrapped node starts
                CUDNN_DISABLE_WARNING = false;
                CUDNN_DISABLE_WARNING_MSG = '';
            }
        }

        // Register listeners
        api.addEventListener('reconnected', onSocketOpen);
        api.addEventListener('executing', onExecuting);
        // Attempt to listen for disconnection via API event and raw socket close
        try { api.addEventListener('disconnected', onSocketClose); } catch {}
        try { api.socket?.addEventListener?.('close', onSocketClose); } catch {}
    },

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        const getExtraMenuOptions = nodeType.prototype.getExtraMenuOptions;
        nodeType.prototype.getExtraMenuOptions = function (_, options) {
            getExtraMenuOptions?.apply(this, arguments);
            if (!this._is_cudnn_wrapped) {
                options.push({
                    content: "🥚 Disable cuDNN (wrapper)",
                    callback: async () => {
                        const data = await call_server(this.type, "/ovum-cudnn-wrapper/cudnn_wrap_request");
                        if (data?.response) {
                            nodeType.prototype._is_cudnn_wrapped = true;
                            app.graph.nodes
                                .filter((node) => node.type === this.type)
                                .forEach((node) => {
                                    node._is_cudnn_wrapped = true;
                                });
                        } else {
                            console.log(`CUDNNWrapper: Failed to wrap '${this.type}'`);
                        }
                    },
                });
            }
        };

        // Bulk-query backend to mark class as already wrapped
        try {
            const isWrapped = await requestWrapStatus(nodeType.comfyClass);
            if (isWrapped) nodeType.prototype._is_cudnn_wrapped = true;
        } catch (e) {
            // Silently ignore if backend route not available
        }
    },

    async nodeCreated(node) {
        // const original_getTitle = node.getTitle;
        // We don't need to do this is we are adding a freakin' AMD logo to the titlebar.
        // node.getTitle = function () {
        //     const t = original_getTitle ? original_getTitle.call(node) : node.title || node.type;
        //     if (node._is_cudnn_wrapped) return `${t} (cudnn)`;
        //     else return t;
        // };

        // Track running state
        chainCallback(node, 'onExecute', function () {
            this._ov_cudnn_running = true;
        });
        chainCallback(node, 'onExecuted', function () {
            this._ov_cudnn_running = false;
        });
        // TODO: Handle execution errors and interruptions in our cuDNN toggled nodes, and restore cuDNN.
        // execution_error: ExecutionErrorWsMessage;
        // execution_interrupted: ExecutionInterruptedWsMessage;

        // Hover tracking over logo
        chainCallback(node, 'onMouseMove', function (e, pos, canvas) {
            if (!this._is_cudnn_wrapped) return;
            const r = this._ov_cudnn_logo_rect;
            let hovered = false;
            if (r && pos) {
                // pos is in local space of the node
                hovered = (pos[0] >= r.x && pos[0] <= r.x + r.w && pos[1] >= r.y && pos[1] <= r.y + r.h);
            }
            if (hovered !== this._ov_cudnn_hover) {
                this._ov_cudnn_hover = hovered;
                app.graph?.canvas?.setDirty?.(true, true);
            }
        });

        chainCallback(node, 'onDrawForeground', function (ctx) {
            if (!this._is_cudnn_wrapped) return;
            const titleHeight = LiteGraph.NODE_TITLE_HEIGHT;
            const cWidth = this._collapsed_width || LiteGraph.NODE_COLLAPSED_WIDTH;
            const buttonWidth = cWidth - titleHeight - 6;
            let cx = (this.flags.collapsed ? cWidth : this.size[0]) - buttonWidth - 6;

            // Draw bg rectangle similar to halt_toggle
            ctx.fillStyle = this.color || LiteGraph.NODE_DEFAULT_COLOR;
            ctx.beginPath();
            ctx.rect(cx, 2 - titleHeight, buttonWidth, titleHeight - 4);
            ctx.fill();

            // Center of button area
            cx += buttonWidth / 2;

            const size = 7.2 * 2; // 14.4 similar to halt square
            const x0 = cx - size / 2;
            const y0 = -titleHeight / 2 - size / 2;

            // Determine color
            let color = null;
            if (!AMD_LIKE) {
                if (this._ov_cudnn_running) color = '#a84444';
                else color = this.mouseOver ? LiteGraph.NODE_SELECTED_TITLE_COLOR : (this.boxcolor || LiteGraph.NODE_DEFAULT_BOXCOLOR);
            } else {
                // AMD detected
                if (CUDNN_DISABLE_WARNING) color = '#A8003B'; // warning red if failed to disable
                else if (CUDNN_ENABLED) color = '#00A86B'; // AMD green
                else color = '#3378FF'; // blue when disabled
            }

            if (!AMD_LIKE) {
                // NVIDIA GPU detected: draw NVIDIA logo with its native colors
                drawNvidiaLogo(ctx, x0, y0, size);
            } else {
                // AMD detected: draw AMD logo with dynamic color
                // drawNvidiaLogo(ctx, x0, y0, size);
                drawAmdLogo(ctx, x0, y0, size, color);
            }
            this._ov_cudnn_logo_rect = { x: x0, y: y0, w: size, h: size };

            // Tooltip when hovering
            if (this._ov_cudnn_hover) {
                ctx.save();
                let msg;
                let bg = '#00A86B'; // default AMD green background for tooltip
                if (!AMD_LIKE) {
                    msg = `AMD not detected: cuDNN will not be modified (currently ${CUDNN_ENABLED ? 'enabled' : 'disabled'})`;
                } else if (CUDNN_DISABLE_WARNING) {
                    msg = CUDNN_DISABLE_WARNING_MSG || 'Warning: cuDNN was unable to be disabled';
                    bg = '#A8003B'; // warning red background
                } else {
                    msg = `AMD detected: cuDNN is currently ${CUDNN_ENABLED ? 'enabled' : 'disabled'}`;
                }
                const padding = 6;
                ctx.font = (LiteGraph.NODE_TEXT_SIZE * 0.7) + 'px Arial';
                const metrics = ctx.measureText(msg);
                const tw = Math.ceil(metrics.width) + padding * 2;
                const th = LiteGraph.NODE_TEXT_SIZE * 0.7 + padding * 1.2;
                const rx = cx - tw / 2;
                const ry = -titleHeight - th - 4;
                ctx.globalAlpha = 0.9;
                ctx.fillStyle = bg;
                roundedRect(ctx, rx, ry, tw, th, 6);
                ctx.fill();
                ctx.globalAlpha = 1;
                ctx.fillStyle = '#ffffff';
                ctx.textAlign = 'center';
                ctx.textBaseline = 'middle';
                ctx.fillText(msg, cx, ry + th / 2);
                ctx.restore();
            }
        });
    },

    async init() {
        // Initialize cudnn wrapper by applying any configured class conversions on the backend.
        try {
            await call_server(null, "/ovum-cudnn-wrapper/cudnn_wrap_init");
        } catch (e) {}
        // Fetch environment status and keep it fresh periodically
        try {
            await fetch_status();
        } catch {}
        if (!STATUS_TIMER) {
            STATUS_TIMER = setInterval(fetch_status, 60000);
        }
    },
});

async function call_server(type, method) {
    const body = new FormData();
    if (type) body.append("type", type);
    const response = await api.fetchApi(method, { method: "POST", body });
    return await response.json();
}

async function call_server_bulk(types, method) {
    const response = await api.fetchApi(method, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ types }),
    });
    return await response.json();
}
