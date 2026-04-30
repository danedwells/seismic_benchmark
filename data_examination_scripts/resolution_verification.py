#%%
# ---------------------------------------------------------------------------
# Prior resolution diagnostic
# ---------------------------------------------------------------------------
# Verifies that each cached .tt3 prior has the resolution specified in
# config.PRIOR_CONSTRUCTION_PARAMS['target_resolution_deg'], and shows how
# the EPIC C nearest-neighbor lookup samples it relative to the bEPIC grid.
#
# Why run this?  The EPIC C method snaps each 2 km bEPIC grid point to the
# nearest prior cell (round-to-index, no interpolation).  At 0.1° (~11 km)
# several adjacent bEPIC points share the same prior cell value; at 0.02°
# (~2.2 km) they don't.  If the seismicity signal varies at >> 2 km scales,
# both resolutions produce essentially the same posterior — which explains
# the absence of visible location differences between the two experiments.

def verify_prior_resolutions(cache_paths, benchmark_params, target_res_cfg=None):
    """
    Load each cached prior and report its grid dimensions, cell size, and
    how many bEPIC grid points map to each prior cell under nearest-neighbor
    lookup.

    Parameters
    ----------
    cache_paths : dict  {prior_name: tt3_path or None}
    benchmark_params : dict  must contain 'grid_size' and 'grid_km'
    target_res_cfg : dict or None  {prior_name: float or None} from config
    """
    grid_spacing_km = benchmark_params['grid_km'] / benchmark_params['grid_size']
    KM_PER_DEG_LAT  = 111.0    # approximate

    print(f"bEPIC grid spacing : {grid_spacing_km:.1f} km  "
          f"(grid_km={benchmark_params['grid_km']}, "
          f"grid_size={benchmark_params['grid_size']})")
    print(f"{'Prior':<20}  {'shape':>12}  {'dx°':>6}  {'dy°':>6}  "
          f"{'dx km':>7}  {'bEPIC pts/cell':>15}  {'target_res°':>12}")
    print("-" * 85)

    for name, path in cache_paths.items():
        if path is None or not os.path.exists(path):
            target = (target_res_cfg or {}).get(name, '—')
            print(f"  {name:<18}  {'(no file)':>12}  {'—':>6}  {'—':>6}  "
                  f"{'—':>7}  {'—':>15}  {str(target):>12}")
            continue

        p = SeismicPrior.from_tt3(path)
        ny, nx = p.grid.shape
        dx = float(np.diff(p.lons).mean()) if len(p.lons) > 1 else np.nan
        dy = float(np.diff(p.lats).mean()) if len(p.lats) > 1 else np.nan
        dx_km = dx * KM_PER_DEG_LAT
        pts_per_cell = dx_km / grid_spacing_km   # bEPIC grid points per prior cell (approx)
        target = (target_res_cfg or {}).get(name, '—')

        print(f"  {name:<18}  {f'({ny}×{nx})':>12}  {dx:>6.4f}  {dy:>6.4f}  "
              f"{dx_km:>7.2f}  {pts_per_cell:>15.1f}  {str(target):>12}")

    print()
    print("bEPIC pts/cell > 1 means multiple adjacent grid points share the same prior value")
    print("(nearest-neighbor lookup in EPIC_locate_prelim.py:481-483).")
    print("Note: prior_file.py:compute_prior_from_model() has bilinear interpolation but")
    print("is not called by the benchmark — EPIC C samples the raw grid directly.")

verify_prior_resolutions(
    cache_paths      = cache_paths,
    benchmark_params = config.BENCHMARK_PARAMS,
    target_res_cfg   = config.PRIOR_CONSTRUCTION_PARAMS.get('target_resolution_deg'),
)
