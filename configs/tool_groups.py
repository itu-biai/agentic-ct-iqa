"""Tool groups of issue #329 (E13), fixed before any trajectory or rationale was parsed (2026-10-22).

Five groups: noise level, streaks, sharpness, high-frequency residual (hf), visual. Three mappings:

STAGE_B: the 6 tool inputs of Stage B (TOOL_NAMES in colab/vm/vftb_train.py, column order) as the issue lists them
  (high-pass / residual noise and the mean noise score -> noise; streak anisotropy -> streaks; gradient magnitude and
  local gradient deviation -> sharpness; high-frequency residual -> hf). Note: Stage B's prompt calls gradient_p95
  "structure", and noise_score is the mean of the residual_std, gradient_deviation_mean and hf_residual_mean
  percentiles, so it carries part of the sharpness and hf groups too.

AGENT_JSON: every key of the agent's tool_results/img_XX.json (sandboxes test300_ne19, test300_ne26, test300_guided).
  The agent reads that file whole, so one read exposes all four numeric groups at once: reads can only be told apart
  per group for the visual views, not among the numeric groups.

AGENT_VIEWS: every tool_views/img_XX_<view>.png -> visual.

FAITHFULNESS (E16a): per group, the measured percentile the agent saw (compare_to_reference, "_pct" = percentile among
  unprocessed training slices of the raw value, so above 50 = more of it), oriented so that above 50 = more of the
  property the group names: noise level = mean of the high-pass, Immerkaer and NLM-residual percentiles; streaks = the
  air high-pass percentile (the only streak-related percentile shown); sharpness = mean of the Laplacian-variance
  percentile and 100 - the edge-width percentile; hf = mean of the Fourier high-band and NPS high-band percentiles.
"""
GROUPS = ("noise", "streaks", "sharpness", "hf", "visual")

STAGE_B = {"residual_std": "noise", "gradient_deviation_mean": "sharpness", "hf_residual_mean": "hf",
           "noise_score": "noise", "gradient_p95": "sharpness", "streak_anisotropy": "streaks"}
STAGE_B_COLUMNS = {"noise": (0, 3), "sharpness": (1, 4), "hf": (2,), "streaks": (5,)}

AGENT_JSON = {
    # tool -> group; compare_to_reference is split by its keys below
    "high_pass_energy": "noise", "noise_sigma_estimate": "noise", "flat_roi_sigma": "noise",
    "noise_residual_map": "noise", "radial_nps_summary": "hf", "fourier_high_band_fraction": "hf",
    "edge_width": "sharpness", "texture_coherence": "sharpness", "air_artifact_energy": "streaks",
    "streak_anisotropy": "streaks",
    # setup 30 only
    "reference_match": "noise",
}
AGENT_REFERENCE_KEYS = {
    "high_pass_std_pct": "noise", "immerkaer_sigma_pct": "noise", "nlm_residual_std_pct": "noise",
    "laplacian_variance_pct": "sharpness", "edge_width_px_pct": "sharpness", "body_coherence_pct": "sharpness",
    "fourier_high_band_fraction_pct": "hf", "nps_high_band_ratio_pct": "hf", "air_hp_std_pct": "streaks",
}
AGENT_VIEWS = {"window_soft_tissue": "visual", "window_narrow": "visual", "noise_residual_map": "visual",
               "orientation_coherence_map": "visual", "fourier_spectrum": "visual", "mscn_map": "visual",
               "gradient_deviation_map": "visual"}

FAITHFULNESS = {  # group -> (keys averaged, keys entering as 100 - value)
    "noise": (("high_pass_std_pct", "immerkaer_sigma_pct", "nlm_residual_std_pct"), ()),
    "streaks": (("air_hp_std_pct",), ()),
    "sharpness": (("laplacian_variance_pct",), ("edge_width_px_pct",)),
    "hf": (("fourier_high_band_fraction_pct", "nps_high_band_ratio_pct"), ()),
}

# E15 rationale categories (issue #329) and how they meet the tool groups for E16a
CATEGORIES = ("noise", "streaks", "sharpness", "hf", "contrast", "anatomy")
