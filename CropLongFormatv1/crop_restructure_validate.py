#!/usr/bin/env python3
"""
Issue 1 - Crop restructuring and validation (QWeatherRiskF)

Converts the wide district-year dataset into crop-long format
(one row = one district-year-crop), keeps district / year / weather features,
adds quality flags, and runs validation checks.

The input file is never modified (SHA-256 is verified before and after).

Usage:
  python crop_restructure_validate.py --input merged_cleaned_v1d.csv \
      [--reference merged_district_year_v1c.csv] --outdir .

--reference is optional. It is an earlier version that still contains the
original missing values, and is used only to flag cells that were NaN and
were later filled with 0 (provenance flag).
"""
import argparse, hashlib, json, os, re
import numpy as np, pandas as pd

MEASURES = {'area': 'area_1000_ha', 'production': 'production_1000_tons', 'yield': 'yield_kg_per_ha'}
COL_RE = re.compile(r'^(?P<crop>[a-z0-9_]+?)_(?P<measure>area|production|yield)_(?P<unit>1000_ha|1000_tons|kg_per_ha)$')
ID_COLS = ['district_code', 'district_name', 'year']
AGGREGATES = {  # aggregate crop -> member columns (verified numerically below)
    'sorghum': ['kharif_sorghum', 'rabi_sorghum'],
    'oilseeds': ['groundnut', 'sesamum', 'rapeseed_and_mustard', 'safflower', 'castor', 'linseed', 'sunflower', 'soyabean'],
}
SEASONAL = {'kharif_sorghum': 'kharif', 'rabi_sorghum': 'rabi'}
GROUP_CATEGORY = {'minor_pulses'}  # residual category of pulses, not a verified sum of listed crops
CROP_GROUP = {  # descriptive grouping (assumption, for convenience only)
    'rice': 'cereal', 'wheat': 'cereal', 'maize': 'cereal', 'sorghum': 'millet/sorghum', 'kharif_sorghum': 'millet/sorghum',
    'rabi_sorghum': 'millet/sorghum', 'pearl_millet': 'millet/sorghum', 'finger_millet': 'millet/sorghum',
    'chickpea': 'pulse', 'pigeonpea': 'pulse', 'minor_pulses': 'pulse',
    'groundnut': 'oilseed', 'sesamum': 'oilseed', 'rapeseed_and_mustard': 'oilseed', 'safflower': 'oilseed', 'castor': 'oilseed',
    'linseed': 'oilseed', 'sunflower': 'oilseed', 'soyabean': 'oilseed', 'oilseeds': 'oilseed',
    'sugarcane': 'cash crop', 'cotton': 'cash crop'}
YIELD_TOL = 0.05      # 5 % tolerance between reported yield and production/area*1000
AGG_TOL_REL = 0.05    # 5 % tolerance for aggregate vs sum of members
AGG_TOL_ABS = 0.10    # or 0.10 (thousand units) to allow for rounding
LOW_AREA = 1.0        # thousand ha


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True)
    ap.add_argument('--reference', default=None)
    ap.add_argument('--outdir', default='.')
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True); tdir = os.path.join(a.outdir, 'validation_tables'); os.makedirs(tdir, exist_ok=True)
    h0 = sha256(a.input)
    src = pd.read_csv(a.input)
    checks = []
    def chk(name, ok, detail='', finding=False):
        # structural checks must PASS; data-quality findings are reported as OK / FLAGGED (not script failures)
        checks.append({'type': 'data-quality finding' if finding else 'structural', 'check': name,
                       'result': ('OK' if ok else 'FLAGGED') if finding else ('PASS' if ok else 'FAIL'), 'detail': detail})

    # ---- 1. identify columns
    crop_cols = {}
    for c in src.columns:
        m = COL_RE.match(c)
        if m and '__' not in c:
            crop_cols.setdefault(m['crop'], {})[m['measure']] = c
    complete = {k: v for k, v in crop_cols.items() if set(v) == set(MEASURES)}
    partial = {k: v for k, v in crop_cols.items() if set(v) != set(MEASURES)}
    weather_cols = [c for c in src.columns if '__' in c]
    other_cols = [c for c in src.columns if c not in weather_cols and not any(c in v.values() for v in crop_cols.values())]
    meta_cols = [c for c in other_cols if c not in ID_COLS]
    crops = list(complete)
    chk('Every crop has area, production and yield columns', len(partial) == 0, f'{len(crops)} crops x 3 = {3*len(crops)} columns; partial: {list(partial)}')
    chk('All columns classified (crop / weather / identifier-metadata)', len(crops) * 3 + len(weather_cols) + len(other_cols) == src.shape[1],
        f'{len(crops)*3} crop + {len(weather_cols)} weather + {len(other_cols)} id/meta = {src.shape[1]}')
    chk('Source key unique (district_code, year)', not src.duplicated(['district_code', 'year']).any(), f'{len(src)} rows')

    ref = pd.read_csv(a.reference) if a.reference else None
    ref_map = {}
    if ref is not None:  # map reference (original column names) to snake_case names
        for c in ref.columns:
            m = re.match(r'^(.*) (AREA|PRODUCTION|YIELD) \((.*)\)$', c)
            if m:
                unit = {'1000 ha': '1000_ha', '1000 tons': '1000_tons', 'Kg per ha': 'kg_per_ha'}[m[3]]
                ref_map[f"{m[1].lower().replace(' ', '_')}_{m[2].lower()}_{unit}"] = c

    # ---- 2. wide -> long
    frames = []
    for crop, cols in complete.items():
        t = src[ID_COLS].copy()
        t['crop'] = crop
        for meas, c in cols.items():
            t[MEASURES[meas]] = src[c].values
        if ref is not None:
            filled = pd.Series(False, index=src.index)
            for meas, c in cols.items():
                rc = ref_map.get(c)
                if rc is not None:
                    filled |= ref[rc].isna().values & src[c].notna().values
            t['flag_nan_filled_with_zero'] = filled.values
        frames.append(t)
    L = pd.concat(frames, ignore_index=True)
    L['crop_name'] = L.crop.str.replace('_', ' ').str.title()
    L['crop_group'] = L.crop.map(CROP_GROUP).fillna('unclassified')
    L['is_aggregate'] = L.crop.isin(AGGREGATES)
    L['is_seasonal_component'] = L.crop.isin(SEASONAL)
    L['season_component'] = L.crop.map(SEASONAL).fillna('')
    comp_of = {m: agg for agg, mem in AGGREGATES.items() for m in mem}
    L['component_of_aggregate'] = L.crop.map(comp_of).fillna('')
    L['is_group_category'] = L.crop.isin(GROUP_CATEGORY)
    A, P, Y = 'area_1000_ha', 'production_1000_tons', 'yield_kg_per_ha'

    # ---- 3. observation-level flags
    L['yield_check_kg_per_ha'] = np.where(L[A] > 0, L[P] / L[A] * 1000, np.nan)
    L['yield_rel_diff'] = (L[Y] - L.yield_check_kg_per_ha).abs() / L.yield_check_kg_per_ha.replace(0, np.nan)
    L['flag_any_missing'] = L[[A, P, Y]].isna().any(axis=1)
    L['flag_area_zero'] = L[A] == 0
    L['flag_production_zero'] = L[P] == 0
    L['flag_yield_zero'] = L[Y] == 0
    L['flag_all_zero'] = (L[A] == 0) & (L[P] == 0) & (L[Y] == 0)
    L['flag_zero_area_positive_production'] = (L[A] == 0) & (L[P] > 0)
    L['flag_positive_area_zero_production'] = (L[A] > 0) & (L[P] == 0)
    L['flag_positive_area_zero_yield'] = (L[A] > 0) & (L[Y] == 0)
    L['flag_negative_value'] = (L[[A, P, Y]] < 0).any(axis=1)
    L['flag_yield_mismatch'] = (L[A] > 0) & (L[P] > 0) & (L.yield_rel_diff > YIELD_TOL)
    L['flag_low_area_lt_1000ha'] = (L[A] > 0) & (L[A] < LOW_AREA)

    def jump(g):
        out = []
        for i, r in g.iterrows():
            o = g.drop(i); o = o[(o[A] >= LOW_AREA) & (o[Y] > 0)]
            out.append(r[Y] / o[Y].median() if len(o) >= 3 and r[A] >= LOW_AREA and r[Y] > 0 else np.nan)
        return pd.Series(out, index=g.index)
    L['yield_vs_other_years_median'] = L.groupby(['crop', 'district_code'], group_keys=False).apply(jump)
    L['flag_yield_jump'] = (L.yield_vs_other_years_median > 2) | (L.yield_vs_other_years_median < 0.5)

    # aggregates: compare with member sum
    L['aggregate_member_sum_area'] = np.nan; L['aggregate_member_sum_production'] = np.nan; L['flag_aggregate_mismatch'] = False
    for agg, mem in AGGREGATES.items():
        piv = {m: L[L.crop == m].set_index(['district_code', 'year']) for m in mem}
        ix = L.index[L.crop == agg]; keys = list(zip(L.loc[ix, 'district_code'], L.loc[ix, 'year']))
        sa = np.array([sum(piv[m].loc[k, A] for m in mem) for k in keys]); sp = np.array([sum(piv[m].loc[k, P] for m in mem) for k in keys])
        L.loc[ix, 'aggregate_member_sum_area'] = sa; L.loc[ix, 'aggregate_member_sum_production'] = sp
        da = np.abs(L.loc[ix, A].values - sa); dp = np.abs(L.loc[ix, P].values - sp)
        bad = (da > np.maximum(AGG_TOL_ABS, AGG_TOL_REL * np.abs(sa))) | (dp > np.maximum(AGG_TOL_ABS, AGG_TOL_REL * np.abs(sp)))
        L.loc[ix, 'flag_aggregate_mismatch'] = bad

    # provenance flags for known issues (documented earlier in the project)
    L['flag_sugarcane_unit_corrected'] = (L.crop == 'sugarcane') & (L.year <= 2017)   # production/yield x10 applied in v1c
    L['flag_cotton_2019_suspect'] = (L.crop == 'cotton') & (L.year == 2019)
    L['flag_minor_pulses_not_reported'] = (L.crop == 'minor_pulses') & (L.year >= 2018) & (L[A] < LOW_AREA)
    if 'flag_nan_filled_with_zero' not in L: L['flag_nan_filled_with_zero'] = False

    flag_cols = [c for c in L.columns if c.startswith('flag_')]
    L['n_flags'] = L[flag_cols].sum(axis=1)
    L['flag_summary'] = L[flag_cols].apply(lambda r: ';'.join(c.replace('flag_', '') for c in flag_cols if r[c]), axis=1)

    # ---- 4. attach district-year features (ids, metadata, weather) unchanged
    feat = src[ID_COLS + meta_cols + weather_cols].copy()
    L = L.merge(feat.drop(columns=['district_name']), on=['district_code', 'year'], how='left', validate='many_to_one')
    key = ['district_code', 'year', 'crop']
    first = ['district_code', 'district_name', 'year', 'crop', 'crop_name', 'crop_group', 'is_aggregate', 'is_seasonal_component', 'season_component',
             'component_of_aggregate', 'is_group_category', A, P, Y, 'yield_check_kg_per_ha', 'yield_rel_diff', 'yield_vs_other_years_median',
             'aggregate_member_sum_area', 'aggregate_member_sum_production'] + flag_cols + ['n_flags', 'flag_summary']
    L = L[first + [c for c in L.columns if c not in first]].sort_values(key).reset_index(drop=True)

    # ---- 5. acceptance checks
    chk('One row per district-year-crop', len(L) == len(src) * len(crops), f'{len(L)} rows = {len(src)} district-years x {len(crops)} crops')
    chk('No duplicate keys (district_code, year, crop)', not L.duplicated(key).any(), f'{int(L.duplicated(key).sum())} duplicates')
    rt_ok = True; rt_cells = 0
    for crop, cols in complete.items():
        s = L[L.crop == crop].set_index(['district_code', 'year'])
        for meas, c in cols.items():
            w = src.set_index(['district_code', 'year'])[c]
            same = np.array_equal(s[MEASURES[meas]].reindex(w.index).to_numpy(), w.to_numpy(), equal_nan=True); rt_ok &= same; rt_cells += len(w)
    chk('Round trip: long values identical to every source crop cell', rt_ok, f'{rt_cells} cells compared exactly')
    wk = True
    for c in weather_cols:
        wk &= np.array_equal(L.groupby(['district_code', 'year'])[c].first().to_numpy(), src.sort_values(['district_code', 'year'])[c].to_numpy(), equal_nan=True)
    chk('Weather features preserved unchanged (all rows)', wk, f'{len(weather_cols)} weather columns')
    chk('Weather identical across crops within a district-year', bool((L.groupby(['district_code', 'year'])[weather_cols].nunique(dropna=False) <= 1).all().all()), '')
    chk('Source file unchanged (SHA-256 before == after)', sha256(a.input) == h0, h0[:16] + '...')

    # ---- 6. summary tables
    chk('Negative values', not L.flag_negative_value.any(), f'{int(L.flag_negative_value.sum())} rows', finding=True)
    chk('Missing values in crop columns', not L.flag_any_missing.any(), f'{int(L[[A,P,Y]].isna().sum().sum())} NaN cells (v1d has none)', finding=True)
    for agg in AGGREGATES:
        s = L[L.crop == agg]
        chk(f'Aggregate "{agg}" equals sum of its members', not s.flag_aggregate_mismatch.any(), f'{int(s.flag_aggregate_mismatch.sum())} of {len(s)} district-years mismatch (tolerance {AGG_TOL_REL:.0%} or {AGG_TOL_ABS})', finding=True)
    chk('Yield reported = production/area x 1000 (area>0, prod>0)', L.flag_yield_mismatch.sum() == 0, f'{int(L.flag_yield_mismatch.sum())} rows differ by more than {YIELD_TOL:.0%}', finding=True)
    chk('Zero area with positive production', L.flag_zero_area_positive_production.sum() == 0, f'{int(L.flag_zero_area_positive_production.sum())} rows (rounding of tiny areas)', finding=True)
    chk('Positive area with zero production', L.flag_positive_area_zero_production.sum() == 0, f'{int(L.flag_positive_area_zero_production.sum())} rows', finding=True)
    pd.DataFrame(checks).to_csv(os.path.join(tdir, 'validation_checks.csv'), index=False)

    cs = L.groupby('crop').agg(crop_group=('crop_group', 'first'), is_aggregate=('is_aggregate', 'first'), is_seasonal_component=('is_seasonal_component', 'first'),
        rows=('year', 'size'), area_positive=(A, lambda s: int((s > 0).sum())), area_zero=(A, lambda s: int((s == 0).sum())), area_missing=(A, lambda s: int(s.isna().sum())),
        filled_from_nan=('flag_nan_filled_with_zero', 'sum'), mean_area=(A, 'mean'), mean_production=(P, 'mean'),
        mean_yield_positive=(Y, lambda s: s[s > 0].mean()), yield_mismatch=('flag_yield_mismatch', 'sum'), yield_jump=('flag_yield_jump', 'sum'),
        low_area=('flag_low_area_lt_1000ha', 'sum'), any_flag=('n_flags', lambda s: int((s > 0).sum()))).round(2).reset_index()
    cs.to_csv(os.path.join(tdir, 'crop_summary.csv'), index=False)
    L[flag_cols].sum().rename('rows_flagged').to_frame().to_csv(os.path.join(tdir, 'flag_counts.csv'))
    L.groupby('crop')[flag_cols].sum().to_csv(os.path.join(tdir, 'flag_counts_by_crop.csv'))
    L[L.n_flags > 0][['district_code', 'district_name', 'year', 'crop', A, P, Y, 'n_flags', 'flag_summary']].to_csv(os.path.join(tdir, 'flagged_observations.csv'), index=False)
    pd.DataFrame({'wide_column': [c for v in complete.values() for c in v.values()], 'crop': [k for k, v in complete.items() for _ in v], 'measure': [m for v in complete.values() for m in v]}).to_csv(os.path.join(tdir, 'column_inventory.csv'), index=False)
    pd.DataFrame([{'aggregate': k, 'members': ', '.join(v)} for k, v in AGGREGATES.items()] + [{'aggregate': 'minor_pulses', 'members': '(group category; members not in dataset)'}]).to_csv(os.path.join(tdir, 'aggregate_definitions.csv'), index=False)
    L.to_csv(os.path.join(a.outdir, 'crop_long_v1.csv'), index=False)
    print(pd.DataFrame(checks).to_string()); print('crops:', len(crops), '| long shape:', L.shape)
    assert all(c['result'] == 'PASS' for c in checks if c['type'] == 'structural'), 'acceptance check failed'


if __name__ == '__main__':
    main()
