# DiffHDR Free Memory

Releases the cached DiffHDR pipeline and returns the VRAM.

You rarely need this. `DiffHDR Load Model` already holds at most one pipeline and
evicts it before building a replacement, so switching models does not stack. This node
is for when you want the card back for something else without restarting ComfyUI.

It always runs when the graph runs — the point is the side effect, so it is not cached.

Set `trigger` to false to make it a no-op without removing it from the graph.
