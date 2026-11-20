"""
Builds the simulation inputs from public downloads (no made-up data).

Put these raw files in raw/ (download links are in config.json -> "_downloads"):
  raw/acs_s1810.json        Census API: ACS 5-year S1810, ambulatory-difficulty count by tract
     -- or instead --       data.census.gov download of table S1810 for the county's tracts
                            (the .zip, or the *S1810*-Data.csv inside it)
  raw/gaz_tracts_36.txt     Census Gazetteer: 2020 census tract centroids, New York State
  raw/facilities.csv        NYC Facilities Database (NYC Planning), CSV
  raw/registrants.csv       MTA NYCT Paratransit Registrants (data.ny.gov cxps-6wkq), CSV
  raw/tl_2020_36_tabblock20.zip   Census 2020 TIGER/Line blocks, New York State (keep zipped)

Writes:
  data/tracts.csv           geoid, lat, lon, weight (weight = ACS ambulatory-difficulty count)
  data/blocks.csv           block_geoid, tract_geoid, lat, lon, pop (2020 Census population > 0)
  data/destinations.csv     name, type, lat, lon, weight
  data/derived.json         borough share of AAR registrants (latest month), with its source
  data/facdb_types_in_borough.csv   every facility type in the borough, to tune destination_factype_weights
  data/dialysis_found.csv           the facilities identified as dialysis centers by name, to check

Usage: python prepare_inputs.py --config config.json
"""
import argparse, glob, json, os, sys
import pandas as pd

BOROUGHS = {  # county FIPS, FacDB borough name variants, registrant-file name variants
    "bronx": ("005", {"bronx", "bx", "2"}, {"bronx"}),
    "brooklyn": ("047", {"brooklyn", "bk", "3"}, {"brooklyn", "kings"}),
    "manhattan": ("061", {"manhattan", "mn", "1"}, {"manhattan", "new york"}),
    "queens": ("081", {"queens", "qn", "4"}, {"queens"}),
    "staten island": ("085", {"staten island", "si", "5"}, {"staten island", "richmond"}),
}


def find_col(df, *needles, required=True):
    for c in df.columns:
        lc = c.lower()
        if all(n in lc for n in needles):
            return c
    if required:
        sys.exit(f"Could not find a column containing {needles} in: {list(df.columns)}")
    return None


def read_acs(raw):
    """ACS S1810 by tract, from either source:
      A) raw/acs_s1810.json from the Census API (needs a free API key), or
      B) the data.census.gov CSV download of table S1810 (file name contains S1810 and
         ends in -Data.csv), unzipped in raw/ or still inside its .zip.
    Returns a table with state, county, tract and the S1810 columns as text."""
    jpath = os.path.join(raw, "acs_s1810.json")
    if os.path.exists(jpath) and os.path.getsize(jpath) > 0:
        try:
            rows = json.load(open(jpath))
        except json.JSONDecodeError:
            rows = None
        if isinstance(rows, list) and len(rows) > 1 and isinstance(rows[0], list):
            print("  ACS source: Census API file", jpath)
            return pd.DataFrame(rows[1:], columns=rows[0])
        print(f"  warning: {jpath} holds no data (often an API-key error); "
              "looking for a data.census.gov CSV instead")
    csvs = [f for f in glob.glob(os.path.join(raw, "**", "*S1810*Data*.csv"), recursive=True)]
    zips = glob.glob(os.path.join(raw, "*S1810*.zip")) + glob.glob(os.path.join(raw, "*s1810*.zip"))
    if csvs:
        path, df = csvs[0], pd.read_csv(csvs[0], dtype=str)
    elif zips:
        import zipfile
        z = zipfile.ZipFile(zips[0])
        member = next((m for m in z.namelist() if m.endswith("-Data.csv")), None)
        if member is None:
            sys.exit(f"{zips[0]} has no *-Data.csv inside")
        path, df = f"{zips[0]}:{member}", pd.read_csv(z.open(member), dtype=str)
    else:
        sys.exit("No ACS file found. Put either raw/acs_s1810.json (Census API) or the "
                 "data.census.gov S1810 download (zip or its -Data.csv) in raw/.")
    print("  ACS source: data.census.gov download", path)
    if "GEO_ID" not in df.columns:
        sys.exit(f"{path} has no GEO_ID column; it should be the S1810 '-Data.csv' file")
    df = df[df.GEO_ID.str.startswith("1400000US", na=False)].copy()   # tracts only; drops label row
    if df.empty:
        sys.exit(f"{path} has no census-tract rows; download it with geography = all tracts in the county")
    g = df.GEO_ID.str[-11:]
    df["state"], df["county"], df["tract"] = g.str[:2], g.str[2:5], g.str[5:]
    return df


def build_tracts(cfg, raw, county):
    acs = read_acs(raw)
    var, tot = cfg["acs_ambulatory_var"], cfg["acs_total_pop_var"]
    for v in (var, tot, "state", "county", "tract"):
        if v not in acs.columns:
            sys.exit(f"The ACS file has no column {v}; re-download it (see config.json)")
    acs = acs[acs.county == county].copy()
    acs["geoid"] = acs.state + acs.county + acs.tract
    acs["weight"] = pd.to_numeric(acs[var], errors="coerce")
    acs["total_pop"] = pd.to_numeric(acs[tot], errors="coerce")
    acs.loc[acs.weight < 0, "weight"] = pd.NA          # Census uses negative codes for N/A
    bad = acs[acs.weight > acs.total_pop]
    if len(bad):
        sys.exit(f"{len(bad)} tracts have ambulatory count > total population: "
                 f"check that {var} is the ambulatory-difficulty COUNT, not a percent.")

    path = os.path.join(raw, "gaz_tracts_36.txt")
    try:
        gaz = pd.read_csv(path, sep="\t", dtype={"GEOID": str})
    except UnicodeDecodeError:
        gaz = pd.read_csv(path, sep="\t", dtype={"GEOID": str}, encoding="latin-1")
    gaz.columns = [c.strip() for c in gaz.columns]
    gaz = gaz.rename(columns={"GEOID": "geoid", "INTPTLAT": "lat", "INTPTLONG": "lon"})
    t = acs.merge(gaz[["geoid", "lat", "lon"]], on="geoid", how="left")
    unmatched = t.lat.isna().sum()
    if unmatched:
        print(f"  warning: {unmatched} ACS tracts had no Gazetteer centroid and were dropped")
    t = t.dropna(subset=["lat", "lon", "weight"])
    t = t[t.weight > 0]
    print(f"  tracts: {len(t)} with ambulatory-difficulty population, "
          f"total {int(t.weight.sum()):,} people")
    return t[["geoid", "lat", "lon", "weight", "total_pop"]]


def build_blocks(raw, county, tract_ids):
    """Populated 2020 census blocks in the borough, with their internal points.

    Each simulated home is placed at one of these blocks, chosen within its tract in
    proportion to block population, so homes never land in water, parks or other
    blocks where nobody lives.
    """
    try:
        import shapefile                                      # pip install pyshp
    except ImportError:
        sys.exit("Missing library: run  pip3 install pyshp")
    hits = (glob.glob(os.path.join(raw, "*tabblock20*.zip"))
            + glob.glob(os.path.join(raw, "**", "*tabblock20*.dbf"), recursive=True))
    if not hits:
        sys.exit("No census block file in raw/ -- download tl_2020_36_tabblock20.zip (see config.json)")
    path = hits[0]
    rdr = shapefile.Reader(path) if path.endswith(".zip") else shapefile.Reader(dbf=open(path, "rb"))
    names = [f[0] for f in rdr.fields[1:]]
    need = ["GEOID20", "COUNTYFP20", "POP20", "INTPTLAT20", "INTPTLON20"]
    if any(n not in names for n in need):
        sys.exit(f"Block file is missing fields {set(need) - set(names)}; it should be the 2020 TABBLOCK20 file")
    rows = [r for r in rdr.iterRecords(fields=need) if r["COUNTYFP20"] == county and int(r["POP20"]) > 0]
    b = pd.DataFrame({
        "block_geoid": [r["GEOID20"] for r in rows],
        "tract_geoid": [r["GEOID20"][:11] for r in rows],
        "lat": [float(r["INTPTLAT20"]) for r in rows],
        "lon": [float(r["INTPTLON20"]) for r in rows],
        "pop": [int(r["POP20"]) for r in rows]})
    no_blocks = set(tract_ids) - set(b.tract_geoid)
    if no_blocks:
        print(f"  warning: {len(no_blocks)} ACS tracts have no populated 2020 block; "
              "their homes will use the tract's internal point")
    print(f"  blocks: {len(b):,} populated blocks, {b['pop'].sum():,} people (2020 Census)")
    return b


def build_destinations(cfg, raw, boro_names):
    f = pd.read_csv(os.path.join(raw, "facilities.csv"), low_memory=False)
    f.columns = [c.lower() for c in f.columns]
    name_c = find_col(f, "facname")
    lat_c, lon_c = find_col(f, "latitude"), find_col(f, "longitude")
    boro_c = next((c for c in ("boro", "borough", "borocode", "boro_code") if c in f.columns), None)
    if boro_c is None:
        boro_c = find_col(f, "boro")
    type_cols = [c for c in ("factype", "facsubgrp", "facgroup") if c in f.columns]
    if not type_cols:
        sys.exit("facilities.csv has none of factype / facsubgrp / facgroup")
    boro_vals = f[boro_c].astype(str).str.strip().str.lower().str.replace(r"\.0$", "", regex=True)
    f = f[boro_vals.isin(boro_names)]
    if f.empty:
        sys.exit(f"No facilities matched this borough in column '{boro_c}'. "
                 f"Values look like: {sorted(boro_vals.unique())[:10]}")
    f = f.dropna(subset=[lat_c, lon_c])
    os.makedirs("data", exist_ok=True)
    (f.groupby(type_cols).size().rename("count").reset_index()
       .sort_values("count", ascending=False)
       .to_csv("data/facdb_types_in_borough.csv", index=False))

    # Destinations are the exact FacDB facility types listed in destination_factype_weights
    # (case-insensitive); every other type is left out. Exact types, not keywords, so that
    # e.g. school-based or mobile clinics can't slip in through a shared word.
    weights = {k.strip().upper(): float(v) for k, v in cfg["destination_factype_weights"].items()
               if not k.startswith("_")}
    ftype = f["factype"].astype(str).str.strip().str.upper()
    f["weight"] = ftype.map(weights).fillna(0.0)
    f["matched"] = ftype.where(f.weight > 0, "")
    # Dialysis centers have no FacDB type of their own (NY licenses them as diagnostic &
    # treatment centers or hospital clinics), so they are found by name among the listed
    # health facility types and given dialysis_weight.
    dial = cfg.get("dialysis_name_keywords", [])
    if dial:
        names = f[name_c].astype(str).str.upper()
        is_dial = (f.weight > 0) & names.str.contains("|".join(k.upper() for k in dial), regex=True)
        f.loc[is_dial, "weight"] = float(cfg["dialysis_weight"])
        f.loc[is_dial, "matched"] = "DIALYSIS (by name)"
        print(f"  dialysis centers found by name: {int(is_dial.sum())}")
        f.loc[is_dial, [name_c, "factype"]].to_csv("data/dialysis_found.csv", index=False)
    missing = sorted(set(weights) - set(ftype))
    if missing:
        print(f"  note: {len(missing)} listed facility types have no facility in this borough: {missing}")
    d = f[f.weight > 0]
    print(f"  destinations: {len(d)} facilities matched; by facility type and weight:")
    print(d.groupby(["matched", "weight"]).size().rename("facilities").to_string())
    return pd.DataFrame({"name": d[name_c].values, "type": d["matched"].values,
                         "lat": d[lat_c].values, "lon": d[lon_c].values,
                         "weight": d.weight.values})


def borough_share(raw, reg_names):
    r = pd.read_csv(os.path.join(raw, "registrants.csv"))
    cols = {c.lower(): c for c in r.columns}
    # The borough column has been labeled Borough, County or Location in different releases
    boro_c = next((cols[c] for c in cols if any(w in c for w in ("borough", "boro", "county", "location"))), None)
    if boro_c is None:
        sys.exit(f"No borough/county/location column in registrants.csv: {list(r.columns)}")
    # Date: a single month/date column, or separate Year + Month columns
    year_c = next((cols[c] for c in cols if c == "year"), None)
    month_c = next((cols[c] for c in cols if "month" in c or "date" in c), None)
    if month_c is None:
        sys.exit(f"No month/date column in registrants.csv: {list(r.columns)}")
    if year_c and pd.to_numeric(r[month_c], errors="coerce").notna().all():
        r["_d"] = pd.to_datetime(dict(year=r[year_c], month=r[month_c], day=1), errors="coerce")
    else:
        r["_d"] = pd.to_datetime(r[month_c], errors="coerce")
    if r._d.isna().all():
        sys.exit(f"Could not read dates in column '{month_c}'")
    # Count: a numeric column, preferring one whose name mentions registrants
    skip = {boro_c, month_c, year_c}
    numeric = [c for c in r.columns if c not in skip and c != "_d"
               and pd.to_numeric(r[c].astype(str).str.replace(",", ""), errors="coerce").notna().mean() > 0.9]
    named = [c for c in numeric if "registrant" in c.lower() or "count" in c.lower()]
    pick = named if named else numeric
    if len(pick) != 1:
        sys.exit(f"Can't tell which registrants column holds the count (candidates {pick}); "
                 f"columns: {list(r.columns)}")
    num_c = pick[0]
    r[num_c] = pd.to_numeric(r[num_c].astype(str).str.replace(",", ""), errors="coerce")
    latest = r[r._d == r._d.max()].copy()
    latest["_n"] = latest[num_c]
    latest["_b"] = (latest[boro_c].astype(str).str.strip().str.lower()
                    .str.replace(r"^the ", "", regex=True).str.replace(r" county$", "", regex=True))
    # Denominator = ALL registrants, including those living outside the five boroughs
    # (e.g. Nassau, Westchester), because the citywide weekday trip total covers every
    # AAR registrant. Only summary rows (Total, All, ...) are left out, so nothing counts twice.
    is_total = latest._b.str.contains(r"\btotal\b|^all\b|systemwide|citywide|all locations", regex=True)
    citywide = latest[~is_total]._n.sum()
    mine = latest[latest._b.isin(reg_names)]._n.sum()
    if not citywide or not mine:
        sys.exit(f"Registrant borough names not recognised in column '{boro_c}': "
                 f"{sorted(latest._b.unique())}")
    share = mine / citywide
    five = {n for _, _, s in BOROUGHS.values() for n in s}
    outside = sorted(set(latest[~is_total]._b) - five)
    if outside:
        print(f"  included in the total, outside the five boroughs: {outside}")
    if is_total.any():
        print(f"  left out as summary rows: {sorted(set(latest[is_total]._b))}")
    print(f"  registrants ({latest._d.max():%Y-%m}, column '{boro_c}'): {int(mine):,} of {int(citywide):,} "
          f"all AAR registrants = {share:.1%} in this borough")
    return {"borough_registrant_share": share, "month": f"{latest._d.max():%Y-%m}",
            "borough_registrants": int(mine), "all_registrants": int(citywide),
            "denominator": "all AAR registrants, including those outside the five boroughs",
            "source": "MTA NYCT Paratransit Registrants: Beginning 2018 (data.ny.gov cxps-6wkq)"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--raw", default="raw")
    args = ap.parse_args()
    cfg = json.load(open(args.config))
    boro = cfg["borough"].strip().lower()
    if boro not in BOROUGHS:
        sys.exit(f"borough must be one of {list(BOROUGHS)}")
    county, facdb_names, reg_names = BOROUGHS[boro]
    os.makedirs("data", exist_ok=True)

    print("ACS + Gazetteer ->", cfg["tracts_csv"])
    tracts = build_tracts(cfg, args.raw, county)
    tracts.to_csv(cfg["tracts_csv"], index=False)
    print("2020 Census blocks ->", cfg["blocks_csv"])
    build_blocks(args.raw, county, tracts.geoid).to_csv(cfg["blocks_csv"], index=False)
    print("Facilities Database ->", cfg["dests_csv"])
    build_destinations(cfg, args.raw, facdb_names).to_csv(cfg["dests_csv"], index=False)
    print("Registrants -> data/derived.json")
    json.dump(borough_share(args.raw, reg_names), open("data/derived.json", "w"), indent=2)
    print("\nDone. Check data/facdb_types_in_borough.csv and adjust destination_factype_weights "
          "if important destination types are missing, then re-run.")


if __name__ == "__main__":
    main()
