"""Regenerate example_workflows/*.json from the live node schemas.

    cd ComfyUI && env/bin/python custom_nodes/ComfyUI-DiffHDR/tools/gen_example_workflows.py

Run this after ANY change to a node's INPUT_TYPES. Widget values in a saved ComfyUI
workflow are a positional array, so adding, removing or reordering a widget silently
shifts every later value in the saved examples. Generating from the schema, and
asserting the array lengths at the end, removes that whole class of bug -- provided
the generator lays the array out exactly as the frontend does, which means:

* optional widgets count too, after the required ones;
* an INT with a `control_after_generate` widget -- every input called `seed` gets one
  by default -- serialises an extra value straight after its own. The first release
  of this script missed that, and every Sample value after `seed` landed one widget
  late (sigma_shift=33, window_size=16, tea_cache=1.0) in both the classic and the
  Nodes 2.0 renderer.

tests/test_nodes.py re-checks the shipped files against the same rules.
"""
import importlib.util, json, os, sys, uuid

import torch

PACK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COMFY = os.path.dirname(os.path.dirname(PACK))
sys.path.insert(0, COMFY)
if not torch.cuda.is_available():  # comfy.model_management would otherwise init CUDA
    import comfy.cli_args
    comfy.cli_args.args.cpu = True
spec = importlib.util.spec_from_file_location("CD", os.path.join(PACK, "__init__.py"),
                                              submodule_search_locations=[PACK])
pack = importlib.util.module_from_spec(spec); sys.modules["CD"] = pack
spec.loader.exec_module(pack)
N = pack.NODE_CLASS_MAPPINGS

WIDGET_TYPES = {"INT", "FLOAT", "STRING", "BOOLEAN"}


def _is_widget(t) -> bool:
    return isinstance(t, list) or t in WIDGET_TYPES


def _has_control(name, t, opts) -> bool:
    return t == "INT" and bool(opts.get("control_after_generate", name in ("seed", "noise_seed")))


def widgets(cls, overrides=None):
    """Ordered (name, value) for every serialised widget value, as the frontend lays them out."""
    overrides = overrides or {}
    out = []
    schema = cls.INPUT_TYPES()
    for section in ("required", "optional"):
        for name, spec_ in schema.get(section, {}).items():
            t = spec_[0]
            opts = spec_[1] if len(spec_) > 1 else {}
            if not _is_widget(t):
                continue
            if name in overrides:
                out.append((name, overrides[name]))
            elif isinstance(t, list):
                out.append((name, opts.get("default", t[0])))
            else:
                out.append((name, opts.get("default", "")))
            if _has_control(name, t, opts):
                ctl = opts.get("control_after_generate")
                out.append(("control_after_generate", ctl if isinstance(ctl, str) else "randomize"))
    return out


def sockets(cls):
    """Ordered socket (non-widget) inputs."""
    out = []
    schema = cls.INPUT_TYPES()
    for section in ("required", "optional"):
        for name, spec_ in schema.get(section, {}).items():
            if not _is_widget(spec_[0]):
                out.append((name, spec_[0]))
    return out


def node(nid, key, pos, size, order, values=(), links_in=None, outputs=()):
    """A serialised node. ``links_in`` maps socket name -> link id."""
    links_in = links_in or {}
    cls = N.get(key)
    inputs = ([{"name": n, "type": t, "link": links_in.get(n)} for n, t in sockets(cls)]
              if cls else [{"name": n, "type": t, "link": l} for n, (t, l) in links_in.items()])
    return {
        "id": nid, "type": key, "pos": pos, "size": size, "flags": {},
        "order": order, "mode": 0, "inputs": inputs,
        "outputs": [{"name": n, "type": t, "links": l, "slot_index": i}
                    for i, (n, t, l) in enumerate(outputs)],
        "properties": {"Node name for S&R": key},
        "widgets_values": [v for _, v in values],
    }


def build(title, lora, mode, w, h, loader_over=None, enc_over=None, smp_over=None):
    loader_w = widgets(N["DiffHDR_LoadModel"], {
        "wan_vace_model": "diffusion_models/wan2.1_vace_14B_fp16.safetensors",
        "text_encoder": "text_encoders/umt5-xxl-enc-bf16.safetensors",
        "vae": "vae/wan_2.1_vae.safetensors",
        "lora_name": f"loras/{lora}",
        "tokenizer_dir": "auto", "lora_strength": 1.0,
        "dit_dtype": "fp8_e4m3fn", "vram_mode": "auto", "vram_limit_gb": 0.0,
        **(loader_over or {})})
    enc_w = widgets(N["DiffHDR_Encode"], {"mode": mode, "width": w, "height": h,
                                          **(enc_over or {})})
    smp_w = widgets(N["DiffHDR_Sample"], {"steps": 10, "cfg_scale": 1.0, "seed": 10,
                                          **(smp_over or {})})

    # Laid out left to right in the order data flows, with room for the taller Nodes
    # 2.0 rendering -- the first release stacked Load Model on top of Encode, and in
    # Nodes 2.0 they overlapped.
    nodes = [
        node(4, "LoadImage", [40, 40], [340, 380], 0,
             values=[("image", "example.png"), ("upload", "image")],
             outputs=[("IMAGE", "IMAGE", [11]), ("MASK", "MASK", None)]),
        {
            "id": 8, "type": "Note", "pos": [40, 460], "size": [340, 420], "flags": {},
            "order": 1, "mode": 0, "inputs": [], "outputs": [],
            "properties": {}, "widgets_values": [NOTES[title]],
            "color": "#432", "bgcolor": "#653",
        },
        node(2, "DiffHDR_Encode", [420, 40], [420, 460], 2, enc_w,
             links_in={"images": 11},
             outputs=[("conditioning", "DIFFHDR_COND", [12]), ("over_mask", "MASK", None),
                      ("under_mask", "MASK", None), ("mask", "MASK", [14])]),
        node(6, "MaskToImage", [420, 540], [260, 60], 3,
             links_in={"mask": ("MASK", 14)}, outputs=[("IMAGE", "IMAGE", [15])]),
        node(7, "PreviewImage", [420, 640], [340, 320], 4,
             links_in={"images": ("IMAGE", 15)}),
        node(1, "DiffHDR_LoadModel", [880, 40], [460, 460], 5, loader_w,
             outputs=[("pipeline", "DIFFHDR_PIPELINE", [10])]),
        node(3, "DiffHDR_Sample", [1380, 40], [440, 640], 6, smp_w,
             links_in={"pipeline": 10, "conditioning": 12},
             outputs=[("images", "IMAGE", [13])]),
        node(5, "PreviewImage", [1860, 40], [380, 420], 7,
             links_in={"images": ("IMAGE", 13)}),
    ]
    links = [
        [10, 1, 0, 3, 0, "DIFFHDR_PIPELINE"],
        [11, 4, 0, 2, 0, "IMAGE"],
        [12, 2, 0, 3, 1, "DIFFHDR_COND"],
        [13, 3, 0, 5, 0, "IMAGE"],
        [14, 2, 3, 6, 0, "MASK"],
        [15, 6, 0, 7, 0, "IMAGE"],
    ]
    return {
        # The frontend's workflow schema requires a UUID here. Derived from the name so
        # regenerating does not churn the files.
        "id": str(uuid.uuid5(uuid.NAMESPACE_URL, f"ComfyUI-DiffHDR/{title}")),
        "revision": 0,
        "last_node_id": 8, "last_link_id": 15,
        "nodes": nodes, "links": links, "groups": [], "config": {},
        # Above 0.6, or the classic renderer drops to its low-detail mode and hides widgets.
        "extra": {"ds": {"scale": 0.65, "offset": [60, 60]}},
        "version": 0.4,
    }


_COMMON = ("Preview Image clamps to [0,1], so the result will look blown out -- that "
           "is the preview, not the data. Wire `images` into an EXR writer (e.g. BAT "
           "Video Combine, format image/exr-sequence) to keep the real scene-linear "
           "values.\n\n"
           "The small preview under Encode shows the mask the model will reconstruct. "
           "Tune over_threshold until it covers the clipped areas and nothing else, or "
           "connect your own MASK to Encode.\n\n"
           "Pick your own files in the Load Model node -- the names here are "
           "placeholders.\n\n")
NOTES = {
    "diffhdr_video": _COMMON + "Swap Load Image for a video/sequence loader to process "
        "a clip; up to window_size frames run in a single pass. On a 24 GB card at "
        "1280x720 set window_size to 9 on the Sample node.",
    "diffhdr_panorama": _COMMON + "Equirectangular HDRI: note Encode mode=panorama and "
        "the DiffHDR_Pano LoRA on the loader -- both are required, they go together. "
        "A batch of panoramas is processed one at a time.",
    "diffhdr_long_video": _COMMON + "For clips longer than window_size. Swap Load Image "
        "for a video/sequence loader; with fewer frames than window_size this behaves "
        "exactly like the plain video workflow. Keep use_prev_window_reference on, or "
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
        json.dump(wf, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(f"wrote {name}: {len(wf['nodes'])} nodes, {len(wf['links'])} links")

# Sanity: widget arrays must be the right length for each node type.
for name in sorted(os.listdir(OUT)):
    wf = json.load(open(os.path.join(OUT, name)))
    for nd in wf["nodes"]:
        if nd["type"] in N:
            want = len(widgets(N[nd["type"]]))
            got = len(nd["widgets_values"])
            assert want == got, f"{name}/{nd['type']}: {got} widget values, schema wants {want}"
    print(f"  {name}: widget arrays match the schemas")
