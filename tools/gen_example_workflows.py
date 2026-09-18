"""Regenerate example_workflows/*.json from the live node schemas.

    cd ComfyUI && env/bin/python custom_nodes/ComfyUI-DiffHDR/tools/gen_example_workflows.py

Run this after ANY change to a node's INPUT_TYPES. Widget values in a saved ComfyUI
workflow are a positional array, so adding, removing or reordering a widget silently
shifts every later value in the saved examples. Generating from the schema, and
asserting the array lengths at the end, removes that whole class of bug.
"""
import importlib.util, json, os, sys

PACK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COMFY = os.path.dirname(os.path.dirname(PACK))
sys.path.insert(0, COMFY)
spec = importlib.util.spec_from_file_location("CD", os.path.join(PACK, "__init__.py"),
                                              submodule_search_locations=[PACK])
pack = importlib.util.module_from_spec(spec); sys.modules["CD"] = pack
spec.loader.exec_module(pack)
N = pack.NODE_CLASS_MAPPINGS

NON_WIDGET = {"IMAGE", "MASK", "LATENT", "MODEL", "CLIP", "VAE", "CONDITIONING",
              "DIFFHDR_PIPELINE", "DIFFHDR_COND"}


def widgets(cls, overrides=None):
    """Ordered (name, default) for every widget input, honouring overrides."""
    overrides = overrides or {}
    out = []
    for name, spec_ in cls.INPUT_TYPES().get("required", {}).items():
        t = spec_[0]
        opts = spec_[1] if len(spec_) > 1 else {}
        if isinstance(t, str) and t in NON_WIDGET:
            continue
        if name in overrides:
            out.append((name, overrides[name]))
        elif isinstance(t, list):
            out.append((name, t[0]))
        else:
            out.append((name, opts.get("default", "")))
    return out


def node(nid, key, pos, size, values, order):
    cls = N[key]
    return {
        "id": nid, "type": key, "pos": pos, "size": size, "flags": {},
        "order": order, "mode": 0, "inputs": [], "outputs": [],
        "properties": {"Node name for S&R": key},
        "widgets_values": [v for _, v in values],
    }


def build(title, lora, mode, w, h, loader_over=None, enc_over=None, smp_over=None):
    loader_w = widgets(N["DiffHDR_LoadModel"], {
        "wan_vace_model": "diffusion_models/WAN21/wan2.1_vace_14B_fp16.safetensors",
        "text_encoder": "text_encoders/umt5-xxl-enc-bf16.safetensors",
        "vae": "vae/wan_2.1_vae.safetensors",
        "lora_name": f"diffhdr/{lora}",
        "tokenizer_dir": "auto", "lora_strength": 1.0,
        "dit_dtype": "fp8_e4m3fn", "vram_mode": "auto", "vram_limit_gb": 0.0,
        **(loader_over or {})})
    enc_w = widgets(N["DiffHDR_Encode"], {"mode": mode, "width": w, "height": h,
                                          **(enc_over or {})})
    smp_w = widgets(N["DiffHDR_Sample"], {"steps": 10, "cfg_scale": 1.0, "seed": 10,
                                          **(smp_over or {})})

    nodes = [
        node(1, "DiffHDR_LoadModel", [40, 40], [430, 260], loader_w, 0),
        node(2, "DiffHDR_Encode", [40, 360], [400, 320], enc_w, 2),
        node(3, "DiffHDR_Sample", [520, 40], [420, 420], smp_w, 3),
    ]
    # Wire them up: LoadImage -> Encode -> Sample, LoadModel -> Sample.
    nodes[0]["outputs"] = [{"name": "pipeline", "type": "DIFFHDR_PIPELINE", "links": [10], "slot_index": 0}]
    nodes[1]["inputs"] = [{"name": "images", "type": "IMAGE", "link": 11}]
    nodes[1]["outputs"] = [
        {"name": "conditioning", "type": "DIFFHDR_COND", "links": [12], "slot_index": 0},
        {"name": "over_mask", "type": "MASK", "links": None, "slot_index": 1},
        {"name": "under_mask", "type": "MASK", "links": None, "slot_index": 2},
    ]
    nodes[2]["inputs"] = [
        {"name": "pipeline", "type": "DIFFHDR_PIPELINE", "link": 10},
        {"name": "conditioning", "type": "DIFFHDR_COND", "link": 12},
    ]
    nodes[2]["outputs"] = [{"name": "images", "type": "IMAGE", "links": [13], "slot_index": 0}]

    loader_img = {
        "id": 4, "type": "LoadImage", "pos": [40, 720], "size": [320, 314], "flags": {},
        "order": 1, "mode": 0, "inputs": [],
        "outputs": [{"name": "IMAGE", "type": "IMAGE", "links": [11], "slot_index": 0},
                    {"name": "MASK", "type": "MASK", "links": None, "slot_index": 1}],
        "properties": {"Node name for S&R": "LoadImage"},
        "widgets_values": ["example.png", "image"],
    }
    preview = {
        "id": 5, "type": "PreviewImage", "pos": [990, 40], "size": [300, 300], "flags": {},
        "order": 4, "mode": 0,
        "inputs": [{"name": "images", "type": "IMAGE", "link": 13}],
        "outputs": [], "properties": {"Node name for S&R": "PreviewImage"},
        "widgets_values": [],
    }
    nodes = [nodes[0], loader_img, nodes[1], nodes[2], preview]
    links = [
        [10, 1, 0, 3, 0, "DIFFHDR_PIPELINE"],
        [11, 4, 0, 2, 0, "IMAGE"],
        [12, 2, 0, 3, 1, "DIFFHDR_COND"],
        [13, 3, 0, 5, 0, "IMAGE"],
    ]
    return {
        "id": title, "last_node_id": 5, "last_link_id": 13,
        "nodes": nodes, "links": links, "groups": [], "config": {},
        "extra": {"note": NOTES[title]},
        "version": 0.4,
    }


_COMMON = ("The PreviewImage node clamps to [0,1], so the result will look blown out -- "
           "that is the preview, not the data. Wire `images` into an EXR writer "
           "(in the ETC suite: BAT Video Combine, format image/exr-sequence) to keep "
           "the real scene-linear values. ")
NOTES = {
    "diffhdr_video": _COMMON + "Swap LoadImage for a video/sequence loader to process a "
        "clip; up to 33 frames run in a single pass.",
    "diffhdr_panorama": _COMMON + "Equirectangular HDRI: note Encode mode=panorama and "
        "the DiffHDR_Pano LoRA on the loader -- both are required, they go together.",
    "diffhdr_long_video": _COMMON + "For clips longer than window_size. Swap LoadImage "
        "for a video/sequence loader; with fewer than 34 frames this behaves exactly "
        "like the plain video workflow. Keep use_prev_window_reference on, or "
        "consecutive windows invent different highlight content and the cross-fade "
        "shows.",
}

OUT = os.path.join(PACK, "example_workflows")
specs = [
    ("diffhdr_video.json", "DiffHDR.safetensors", "video", 1280, 720, None, None, None),
    ("diffhdr_panorama.json", "DiffHDR_Pano.safetensors", "panorama", 2048, 1024, None, None, None),
    ("diffhdr_long_video.json", "DiffHDR.safetensors", "video", 1280, 720, None, None,
     {"window_size": 33, "window_stride": 16}),
]
for name, lora, mode, w, h, lo, eo, so in specs:
    wf = build(name[:-5], lora, mode, w, h, lo, eo, so)
    with open(os.path.join(OUT, name), "w") as f:
        json.dump(wf, f, indent=2)
    n_nodes = len(wf["nodes"])
    print(f"wrote {name}: {n_nodes} nodes, {len(wf['links'])} links")

# Sanity: widget arrays must be the right length for each node type.
for name in os.listdir(OUT):
    wf = json.load(open(os.path.join(OUT, name)))
    for nd in wf["nodes"]:
        if nd["type"] in N:
            want = len(widgets(N[nd["type"]]))
            got = len(nd["widgets_values"])
            assert want == got, f"{name}/{nd['type']}: {got} widget values, schema wants {want}"
    print(f"  {name}: widget arrays match the schemas")
