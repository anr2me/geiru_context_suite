// Geiru Keystone Attention (node id GeiruKeystoneAttention): pink like the other Geiru nodes. block_mode = "lighting blocks" greys out the blocks box and shows the lighting blocks in it;
// "custom blocks" makes the box editable again (with whatever was typed there last).
import { app } from "../../scripts/app.js";

const LIGHTING = "31-33, 36-44, 47";

app.registerExtension({
    name: "clown.context.suite.keystone_attention",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "GeiruKeystoneAttention") return;

        const get = (node, name) => node.widgets?.find((w) => w.name === name);

        const sync = (node) => {
            const mode = get(node, "block_mode");
            const box = get(node, "blocks");
            if (!mode || !box) return;
            if (mode.value === "lighting blocks") {
                if (box.value !== LIGHTING) node.__geiruCustomBlocks = box.value;
                box.value = LIGHTING;
                box.disabled = true;
            } else {
                if (box.disabled && node.__geiruCustomBlocks != null) box.value = node.__geiruCustomBlocks;
                box.disabled = false;
            }
            node.setDirtyCanvas?.(true, true);
        };

        const hook = (node) => {
            const w = get(node, "block_mode");
            if (!w || w.__geiruHooked) return;
            const prevCb = w.callback;
            w.callback = function () { const out = prevCb?.apply(this, arguments); sync(node); return out; };
            w.__geiruHooked = true;
        };

        const onCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onCreated?.apply(this, arguments);
            this.color = "#c2407a";      // Geiru pink (same as the Geiru Context Suite nodes)
            this.bgcolor = "#6e2347";
            hook(this);
            sync(this);
            return r;
        };

        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const r = onConfigure?.apply(this, arguments);
            hook(this);
            sync(this);
            return r;
        };
    },
});
