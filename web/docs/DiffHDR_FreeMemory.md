# DiffHDR Free Memory

Releases the cached DiffHDR pipeline and returns the memory.

You rarely need this. `DiffHDR Load Model` already holds at most one pipeline and
evicts it before building a replacement, so switching models does not stack. This node
is for when you want the card (and the system RAM) back for something else without
restarting ComfyUI.

## Wire it after Sample

Connect Sample's `images` through this node's `images` input, and take the output on to
your saver:

```
DiffHDR Sample ──▶ DiffHDR Free Memory ──▶ EXR writer
```

That connection is what makes the release happen *after* sampling. With nothing
connected the node has no dependencies, and ComfyUI runs output nodes like this one
first — before the pipeline has even been used.

The next run rebuilds the pipeline automatically (a cold load, so expect the Load Model
time again). You do not need to touch the loader.

It always runs when the graph runs — the point is the side effect, so it is not cached.
Set `trigger` to false to make it a no-op (images still pass through) without removing
it from the graph.
