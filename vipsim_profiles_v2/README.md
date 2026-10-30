# vipsim_profiles_v2: Table-4 checklist-complete profile drafts

**Status: CALIBRATION DONE, WINNERS LOCKED (2026-07-10). Not yet swapped into
`vipsim_profiles/` (pending: In-Filling gap decision, p4 missing shaders, Opus re-audit).**
The final 7 live here as plain `p1-p7.json` (each carries `renamed_from`); every variant
and rejected experiment is archived in `alternative-profiles/`. Tuner-metric similarity vs
the clean refs: p1 89.4, p2 95.9, p3 96.9, p4 93.3, p5 81.0 (metric-capped, judge by eye),
p6 97.9, p7 87.3 (re-scored 2026-10-29 after the uint8-rounding fix; tuned values unchanged,
previously 89.5 / 95.8 / 97.1 / 93.6 / 80.9 / 98.0 / 87.4).
Steps may carry `_disabled: true` (skipped by the tuner, the renderer,
`run_audit_eval.py` and `abs_diff.py`). Tune with `validation_scripts/profile_tuner.py`
(localhost:8765; see its `/docs` page for every filter + parameter in plain language).

These are corrected recreations of the 7 VIP-Sim participant profiles, rebuilt so each
stack contains **one filter per shader the paper's Table 4 validated** for that participant
(`docs/literature/2507.10479v1.pdf`). They supersede `../vipsim_profiles/p1-p7.json`
(which were tuned by eye to Figure 6 and diverged from the Table-4 checklist), but are
**kept in a separate folder** so nothing is overwritten until they are calibrated.

## Why these exist

At the 2026-07-02 meeting the supervisors + the low-vision advisor could still tell the
paper panels apart from our recreations. Two problems:

1. **Wrong shader sets.** Re-parsing Table 4 exactly showed our v1 stacks diverged: P1 was
   missing *Metamorphosia point*, P5 wrongly included *Nystagmus* (`o` = inappropriate),
   P1/P4/P6 were heavily under-built, and P4 carried two extras (cvd, glare) not in its list.
   **v2 fixes the shader sets.**
2. **Wrong intensities.** The Unity profiles store RAW slider values while the Python port
   is normalized 0-1, so exact strengths can't be reached by eye. **v2 does NOT fix this:**
   the parameter values here are STARTING POINTS. See the repo memory
   `profile-value-conversion-mismatch` and STATUS.md §0 B.

## Table-4 symbol rule

Only **`o` = inappropriate** excludes a shader. `X` (as-is), `-` (adjusted), `*` (added
later), and `!` (validated from memory) all mean **include**.

## What's in each JSON

`table4_mapping` (shader → symbol → our filter → note), `excluded_shaders`,
`removed_from_v1`, `capability_gaps`, `caveat`, and the `filters` stack. `panel_crop` is the
paper reference crop from `vipsim_assets/participants.png`.

## Renders

`renders/p1-p7.png` = paper reference (left) vs our v2 render (right), on the same street
scene the paper used. **All 7 retuned by eye against the paper panels (2026-07-02):** the
original drafts' intensity overshoot (busy/washed panels, black scotomas, buried swirl) is
fixed. Values were still starting points at that date; see the update below.

**Retune principles applied (2026-07-02):**
- Global filters (blur/bloom/cvd/cataracts/distortion) backed off to near-threshold where
  the paper panel stays sharp/colorful (P1) or readable (P2-P4, P6).
- Scotomas gray + semi-transparent, not black (foveal severity ~0.45-0.6), per the meeting.
- Episodic effects (teichopsia, flickering stars, floaters as specks) at near-threshold in
  P4/P6: the paper panels show no rainbow scintillation.
- P1's warp: ONE moderate wide vortex (suction 1.2, radius 0.32, fade_width 0.2) centered
  mid-image; bends lines instead of collapsing them to a point.
- Wash look (P2/P6): in `apply_bcg`, saturation scales with brightness*contrast (linear-space
  multiply before the pivot). A white-out needs a LOW product with the pivot doing the lift
  (e.g. P6: 1.2 * 0.22), not high brightness with moderate contrast.
- P5 mosaic: pixel_size = cells across; 45 matches the paper's block scale, not 70.
- P7 floater cloud recentred to the paper's (x~0.45), tighter core, sparse right side.

**(2026-07-07 UPDATE: the pairing-session framing above is OBSOLETE.** The pairing session was
dropped; calibration is done SOLO by measured matching. Several "residual gaps" listed in old
revisions are now solved by filter extensions: bcg has a real `saturation` knob, the vortex
has an optional swirl term + elliptical field, and the smudges are freely positionable.)

## Solo calibration sprint (2026-07-07 → 07-10)

- **Clean refs:** `renders/ref-Images/p1-p7.png` (whitespace-free panel extractions) are the
  ground truth; `panel_crop` is only a fallback. Each paper panel is a slightly different
  ~92-96% crop of the Trafficscene: per-panel `SCENE_CROP_FRAC` in
  `validation_scripts/render_profiles_v2.py` aligns ours to the ref framing (variants inherit the base
  profile's crop via the `pN` prefix).
- **Interactive tuner:** `.venv/bin/python validation_scripts/profile_tuner.py` → http://localhost:8765.
  Live sliders for every parameter, wipe compare, 🎯 click-to-position, ✏️ sketch overlay
  (with floaters "dots → drawn area"), 📍 per-step influence highlight, save-as-new-profile,
  and a live "ours is X% similar" pill (SSIM / mean-abs-pixel / CIEDE2000: the equivalence-
  validation metric family).
- **Method:** measure the ref (region luminance; object TOP-DECILE saturation, never region
  means; scotoma centroid/extent/core-luminance), sweep the relevant knobs, verify, then
  fine-tune by eye in the tuner. Model prose about colors must be
  pixel-verified before acting on it.
- **Ordering rule:** dark scotoma cores go AFTER the field_loss frost, or the blur washes
  them back out (applied in p4c, p6b).
- **Variants:** `pNb`/`pNc` are experimental copies (caveat marks them); merge the winner
  into `pN`, then delete. Current: p1b/p1c (dot+halo experiments), p3b (= p3c + all-around
  vignette frame + left-side bleed) vs p3c (grey drain + corner gradients/halos), p4b
  (-vortex, desat corner) vs p4c (+tone/cone matching), p5b, p6b (measured scotoma match,
  91% similarity). `p4_alt_flat` is an older fallback.

## Known gaps

- **In-Filling** has no filter in `vipsim_filters.py` (needed by P1 and P6). Add one or
  document as out of scope.
- P3's winner (p3b vs p3c) and P4's winner (p4b vs p4c) not yet picked.

## Regenerate the renders

`.venv/bin/python validation_scripts/render_profiles_v2.py [p1 p3b ...]` (default: p1-p7). It loads each
`pN.json`, takes the clean ref from `renders/ref-Images/` (falls back to `panel_crop`), crops
the Trafficscene to that panel's measured framing, applies the stack, and writes a borderless
ref|ours pair to `renders/pN.png`.

## Next steps

1. Pick the p3 and p4 winners, merge all variant work into `p1-p7.json`, delete variants.
2. Resolve the In-Filling gap.
3. Re-render + re-audit with Opus, then swap `vipsim_profiles_v2/` → `vipsim_profiles/` and
   regenerate the `/audit-compare/profiles` page + Opus compare images.
