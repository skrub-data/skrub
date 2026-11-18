"""
This script is used to generate the plots that show the embedding similarities
produced by the LLMEncoder that are shown in the home page of the website.

It saves two transparent PNG files, following the same approach as
``demo_periodic_features.py``:

- ``llm_encoder_light.png``: dark text, for light backgrounds
- ``llm_encoder_dark.png``: light text, for dark backgrounds

The images are written to ``doc/_static`` so that they can be referenced
directly from the landing page template.
"""

# %%
from pathlib import Path

# import cmocean
import matplotlib.pyplot as plt
import pandas as pd

from skrub import LLMEncoder

STATIC_DIR = Path(__file__).parent / "_static"

# %%
df = pd.DataFrame(
    {
        "text": ["airplane", "flight", "airport", "apple", "fruit"],
    }
)

# %%
encoder = LLMEncoder(model_name="all-MiniLM-L6-v2", n_components=None)

vectors = encoder.fit_transform(df["text"])

# %%
# Similarity between the embeddings of each pair of entries.
similarity = vectors @ vectors.T

cmap = "viridis"
# cmap = cmocean.cm.thermal  # requires the cmocean package

plt.rcParams.update({"font.size": 36})

fig, ax = plt.subplots(figsize=(18, 12), layout="constrained")
image = ax.imshow(similarity, aspect="auto", cmap=cmap)
ax.set_xticks(range(len(df)), labels=df["text"], ha="right")
ax.set_yticks(range(len(df)), labels=df["text"])
suptitle = fig.suptitle("LLMEncoder", fontsize=50)
ax.set_title("Embedding similarity of related words", fontsize=36)
# ax.set_xlabel("Data entries")
# ax.set_ylabel("Data entries")
fig.colorbar(image, ax=ax, shrink=0.8)


# %%
def style_for_background(fig, color, suptitle=None):
    """Make the figure transparent and recolor text and spines."""
    fig.patch.set_alpha(0)
    if suptitle is not None:
        suptitle.set_color(color)
    for ax in fig.axes:
        ax.patch.set_alpha(0)
        ax.title.set_color(color)
        ax.xaxis.label.set_color(color)
        ax.yaxis.label.set_color(color)
        ax.tick_params(axis="both", which="both", colors=color)
        for spine in ax.spines.values():
            spine.set_color(color)


# %%
# Light mode version (dark text for light backgrounds)
style_for_background(fig, "black", suptitle)
fig.savefig(
    STATIC_DIR / "llm_encoder_light.png",
    dpi=300,
    bbox_inches="tight",
    transparent=True,
)

# %%
# Dark mode version (light text for dark backgrounds)
style_for_background(fig, "white", suptitle)
fig.savefig(
    STATIC_DIR / "llm_encoder_dark.png",
    dpi=300,
    bbox_inches="tight",
    transparent=True,
)
# %%
