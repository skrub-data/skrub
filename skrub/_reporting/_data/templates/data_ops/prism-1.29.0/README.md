The files in this directory are a subset of the [Prism.js](https://prismjs.com/)
project (v1.29.0), used to provide client-side syntax highlighting and line
highlighting/scrolling for the source code pages linked from the DataOps
report (`skb.full_report()`), without requiring a network connection.

Files downloaded from the official CDN build:
https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/

- `prism.min.css` (core theme)
- `prism-line-numbers.min.css` / `prism-line-numbers.min.js` (line-numbers plugin)
- `prism-line-highlight.min.css` / `prism-line-highlight.min.js` (line-highlight plugin)
- `prism-core.min.js` (core)
- `prism-python.min.js` (Python language grammar)

Prism.js is distributed under the MIT license (see `LICENSE` in this
directory). Do not edit those files manually; download a newer version from
the CDN above (updating the version number) if an update is needed.
