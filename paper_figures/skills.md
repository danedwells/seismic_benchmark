# Figure guidelines (paper_figures/)

Apply these to every figure generated in this folder. Add new rules at the bottom.

## Rules

1. **Minimal lines.** Drop anything that doesn't carry information: gridlines, extra spines (keep left/bottom only, or none when the axis isn't read), tick marks on axes that aren't read, box frames around legends, redundant reference lines.
2. **Minimal, clearly readable text.** Only labels the viewer needs. Short wording, no sentences. Prefer direct labels on the data over legends when there are few series.
3. **No title by default.** Only add one if there is genuinely no better option. If a label is needed, make it a text annotation inside the axes; otherwise it goes in the paper caption or is said in the presentation. No suptitles.
4. **Clean whitespace separation.** Panels and labels must not overlap or crowd. Set panel positions explicitly (`fig.add_axes` / `AX_POS`), leave clear gaps between panels and around colorbars, and keep annotations off the data (move them into empty space rather than shrinking them). Check the rendered image for overlap before calling a figure done.
5. **Font size no smaller than 8.** Applies to everything: tick labels, legends, annotations, colorbar labels. Enlarge the figure, not shrink the text, to resolve crowding.
6. **Transparent background.** Anything without an explicit fill must be transparent so the figure takes the color of the slide or page behind it. Set `'savefig.transparent': True` in the `plt.rcParams.update({...})` call at the top of every script (this makes the figure and axes backgrounds transparent in both the PNG and PDF). Explicitly filled things stay opaque: map land/ocean, shaded regions, bars, legend boxes (`framealpha`) and text label boxes. Don't add `facecolor='white'` to figures or axes. Text and lines are black by default, so avoid designs that depend on a white background showing through (e.g. white text on an unfilled area).

## Colors (match `mislocation_impact.png` and `improvement_map.png`)

- **Priors:** Uniform `gray`, KDE_Seismicity `darkblue`, ETAS `red`. Other static priors (when shown): Gear1 `tab:green`, NSHM `tab:orange`, Helmstetter `tab:purple`.
- **Maps:** land `lightgray`, ocean `lightcyan`, coastline/state lines thin (`linewidth` 0.4–0.7, state edges `gray`).
- **Diverging (change / improvement, error):** `RdBu_r`, centered on 0 (`TwoSlopeNorm`), blue = improvement, red = worse, with `extend='both'` on the colorbar.
- **Sequential (intensity / MMI):** `Reds`.
- **True / reference location:** gold star with black edge (USGS location); black `x` for small event markers.
- **Highlights:** `crimson` for the single thing a cartoon is pointing at.
- Keep the same prior-to-color mapping across all figures.

## Defaults used in this folder

- `font.family` sans-serif; size 12 base, nothing below 8.
- Save both `.png` (dpi 300) and `.pdf`, `bbox_inches='tight'`, into this folder.
- Scripts are Jupyter-style (`#%%` cells), top-to-bottom runnable, using `PROJECT_ROOT` / `OUTPUT_DIR` from `__file__`.
- Same y-scale across stacked histogram rows; log x-axis for location errors (km).
