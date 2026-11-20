"""
AAR group-pooling feasibility simulation -- Hypothesis #7 (Supply, risk factor 9)

Control: every synthetic trip served as its own vehicle run.
Test:    the SAME trips; feasible, non-overlapping groups of 2..max_group_size riders
         share one run; every other trip stays solo.

Usage
  python prepare_inputs.py --config config.json   # once: builds data/tracts.csv, data/destinations.csv
  python pooling_sim.py --config config.json

Inputs (built by prepare_inputs.py from public downloads)
  tracts_csv : geoid, lat, lon, weight   (weight = ACS ambulatory-difficulty count)
  blocks_csv : block_geoid, tract_geoid, lat, lon, pop  (2020 Census populated blocks)
  Homes: tract chosen by ACS ambulatory weight, then a populated block in that tract
  chosen by 2020 population; the home sits at the block's Census internal point.
  dests_csv  : name, type, lat, lon, weight

Outputs (in --out, default outputs/)
  <scenario>/trips_seed<k>.csv   trip-level log
  summary.csv                    one row per scenario x seed
  manifest.json                  config, code and data hashes (freeze record)

Method
  1. Prefilter: two trips are "neighbours" if their estimated pickup times are within
     prefilter_time_min and both their origins and their destinations are within the
     distance limits. (Appointment trips: estimated pickup = appointment - solo ride.)
  2. Pairs: every neighbour pair is tested by inserting the second trip's pickup and
     drop-off into every position of the first trip's solo run (exhaustive for pairs).
  3. Groups of 3..K: a feasible group is extended by any trip that is a neighbour of
     ALL its members, using the same insertion into the group's best schedule
     (a standard insertion heuristic; not exhaustive for 3+).
  4. A schedule is feasible only if: pickup-time trips are picked up within the promised
     window (0..+W after the requested time; -W..+W if allow_early_pickup), appointment
     trips arrive no more than appt_early_min early and no later than the appointment,
     every ride stays within max_ride_buffer x AAR's maximum ride time for its distance,
     leaving slack for traffic and delays (plus the optional
     max_added_min / max_ride_factor caps),
     seats and wheelchair spots are never exceeded on board, the vehicle never empties
     before the last stop, and the run saves more than min_saved_min vehicle-minutes.
  5. Assignment: an integer program picks non-overlapping groups that remove the most
     vehicle runs first, then save the most vehicle-minutes. solver_gap in summary.csv
     is the proven bound on distance from the optimum (0 = optimal).
"""
import argparse, copy, hashlib, json, math, os, sys
from datetime import datetime
import numpy as np
import pandas as pd
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import csr_matrix

R_MI = 3958.8


def hav_vec(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = (np.sin((p2 - p1) / 2) ** 2
         + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2)
    return 2 * R_MI * np.arcsin(np.sqrt(a))


def hav(lat1, lon1, lat2, lon2):
    p1, p2 = math.radians(lat1), math.radians(lat2)
    a = (math.sin((p2 - p1) / 2) ** 2
         + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lon2 - lon1) / 2) ** 2)
    return 2 * R_MI * math.asin(math.sqrt(a))


# ---------------------------------------------------------------- inputs
def load_inputs(cfg):
    """Real inputs only: build them with prepare_inputs.py from ACS / Gazetteer / FacDB downloads."""
    for key in ("tracts_csv", "dests_csv", "blocks_csv"):
        if not os.path.exists(cfg[key]):
            sys.exit(f"Missing {cfg[key]} -- run prepare_inputs.py first (see README in config.json).")
    if not os.path.exists(cfg["blocks_csv"]):
        sys.exit(f"Missing {cfg['blocks_csv']} -- run prepare_inputs.py first.")
    tracts = pd.read_csv(cfg["tracts_csv"], dtype={"geoid": str})
    dests = pd.read_csv(cfg["dests_csv"])
    for name, df, cols in [("tracts", tracts, ["geoid", "lat", "lon", "weight"]),
                           ("dests", dests, ["name", "lat", "lon", "weight"])]:
        missing = set(cols) - set(df.columns)
        if missing:
            sys.exit(f"{name} file is missing columns: {missing}")
    blocks = pd.read_csv(cfg["blocks_csv"], dtype={"block_geoid": str, "tract_geoid": str})
    tracts = tracts[tracts.weight > 0].reset_index(drop=True)
    # For each tract: its populated blocks' coordinates and population-based probabilities.
    # Tracts with no populated block fall back to the tract's internal point.
    by_tract = {g: d for g, d in blocks.groupby("tract_geoid")}
    tracts.attrs["blocks"] = [
        (by_tract[g].lat.to_numpy(), by_tract[g].lon.to_numpy(),
         (by_tract[g]["pop"] / by_tract[g]["pop"].sum()).to_numpy())
        if g in by_tract else
        (np.array([la]), np.array([lo]), np.array([1.0]))
        for g, la, lo in zip(tracts.geoid, tracts.lat, tracts.lon)]
    return tracts, dests[dests.weight > 0].reset_index(drop=True)


def trips_in_area_peak(cfg):
    """Explicit number if given; otherwise citywide weekday trips x borough share x peak share."""
    if cfg.get("trips_in_area_peak"):
        return int(cfg["trips_in_area_peak"])
    share = cfg.get("borough_registrant_share")
    if share is None:                                   # computed by prepare_inputs.py
        share = json.load(open("data/derived.json"))["borough_registrant_share"]
    return int(round(cfg["weekday_trips_citywide"] * share * cfg["peak_share_of_daily_trips"]))


# ---------------------------------------------------------------- trip generation
def generate_trips(cfg, tracts, dests, seed):
    rng = np.random.default_rng(seed)
    n = trips_in_area_peak(cfg)
    ti = rng.choice(len(tracts), n, p=tracts.weight / tracts.weight.sum())
    di = rng.choice(len(dests), n, p=dests.weight / dests.weight.sum())
    # Home = a populated 2020 census block inside the chosen tract, picked by block population
    blk = tracts.attrs["blocks"]
    home_lat, home_lon = np.empty(n), np.empty(n)
    for k, t in enumerate(ti):
        lat_a, lon_a, p = blk[t]
        b = rng.choice(len(p), p=p)
        home_lat[k], home_lon[k] = lat_a[b], lon_a[b]
    dest_lat, dest_lon = dests.lat.values[di], dests.lon.values[di]
    outbound = rng.random(n) < cfg["outbound_share"]          # home -> destination
    olat = np.where(outbound, home_lat, dest_lat); olon = np.where(outbound, home_lon, dest_lon)
    dlat = np.where(outbound, dest_lat, home_lat); dlon = np.where(outbound, dest_lon, home_lon)
    t_req = rng.uniform(cfg["peak_start_min"], cfg["peak_end_min"], n)
    wc = rng.random(n) < cfg["wheelchair_share"]
    dwell = np.where(wc, cfg["dwell_wc_min"], cfg["dwell_amb_min"]).astype(float)
    miles = hav_vec(olat, olon, dlat, dlon) * cfg["circuity"]
    solo_travel = miles / cfg["speed_mph"] * 60
    # Outbound (home -> destination) trips booked by appointment time: t_req is then the
    # appointment (arrival) time, not a pickup time.
    appt = outbound & (rng.random(n) < cfg["appointment_share_outbound"])
    return pd.DataFrame({
        "trip_id": np.arange(n), "origin_tract": tracts.geoid.values[ti],
        "dest_name": dests.name.values[di], "outbound": outbound,
        "olat": olat, "olon": olon, "dlat": dlat, "dlon": dlon,
        "t_req": t_req, "appointment": appt, "wheelchair": wc, "dwell": dwell,
        "trip_miles": miles, "max_ride": max_ride_minutes(cfg, miles),
        "solo_travel": solo_travel,
        "solo_ride": solo_travel + 2 * dwell})   # board + travel + alight


def max_ride_minutes(cfg, miles):
    """AAR maximum ride time by trip distance (config max_ride_table: [upper miles, minutes])."""
    out = np.full(len(miles), float(cfg["max_ride_table"][-1][1]))
    for upper, minutes in reversed(cfg["max_ride_table"]):
        out[miles < upper] = minutes
    return out


# ---------------------------------------------------------------- schedule simulation
class Sim:
    def __init__(self, T, cfg):
        self.cfg = cfg
        self.A = {c: T[c].to_numpy() for c in
                  ["olat", "olon", "dlat", "dlon", "t_req", "appointment", "wheelchair", "dwell",
                   "solo_ride", "max_ride"]}
        self.k_min_per_mi = cfg["circuity"] / cfg["speed_mph"] * 60
        self._legs = {}

    def leg(self, a, b):
        """Travel minutes between two stops, cached (insertion re-uses the same legs a lot)."""
        key = (a, b)
        v = self._legs.get(key)
        if v is None:
            p, q = self.loc(*a), self.loc(*b)
            v = hav(p[0], p[1], q[0], q[1]) * self.k_min_per_mi
            self._legs[key] = v
        return v

    def loc(self, k, kind):
        A = self.A
        return (A["olat"][k], A["olon"][k]) if kind == "O" else (A["dlat"][k], A["dlon"][k])

    def _run(self, stops, start):
        A, cfg, W = self.A, self.cfg, self.cfg["window_min"]
        early_appt, late_appt = cfg["appt_early_min"], cfg["appt_late_min"]
        time, prev = start, None
        pickup, drop = {}, {}
        wc_on = amb_on = 0
        last = len(stops) - 1
        for idx, (k, kind) in enumerate(stops):
            here = (k, kind)
            if prev is not None:
                time += self.leg(prev, here)
            if kind == "O":
                if A["appointment"][k]:
                    # No pickup window: the drop-off window and max ride time bound the pickup.
                    earliest = A["t_req"][k] - early_appt - A["max_ride"][k]
                    time = max(time, earliest)
                else:
                    earliest = A["t_req"][k] - (W if cfg["allow_early_pickup"] else 0)
                    time = max(time, earliest)                # vehicle waits if early
                    if time > A["t_req"][k] + W:               # outside promised pickup window
                        return None
                pickup[k] = time
                time += A["dwell"][k]                         # boarding
                if A["wheelchair"][k]:
                    wc_on += 1
                else:
                    amb_on += 1
                if wc_on > cfg["wc_spots"] or amb_on > cfg["amb_seats"]:
                    return None
            else:
                if A["appointment"][k]:
                    # Don't drop off more than appt_early_min before the appointment
                    time = max(time, A["t_req"][k] - early_appt - A["dwell"][k])
                time += A["dwell"][k]                         # alighting
                drop[k] = time
                if A["appointment"][k] and time > A["t_req"][k] + late_appt:   # late
                    return None
                if A["wheelchair"][k]:
                    wc_on -= 1
                else:
                    amb_on -= 1
                if wc_on + amb_on == 0 and idx < last:        # vehicle empty mid-run
                    return None
            prev = here
        added = {}
        for k in pickup:
            ride = drop[k] - pickup[k]
            solo = A["solo_ride"][k]
            if ride > cfg["max_ride_buffer"] * A["max_ride"][k]:   # AAR max ride, with a buffer
                return None
            if cfg.get("max_added_min") is not None and ride - solo > cfg["max_added_min"]:
                return None
            if cfg.get("max_ride_factor") is not None and ride > cfg["max_ride_factor"] * solo:
                return None
            added[k] = ride - solo
        duration = time - pickup[stops[0][0]]
        return {"stops": stops, "duration": duration, "added": added,
                "pickup_delay": {k: pickup[k] - A["t_req"][k] for k in pickup
                                 if not A["appointment"][k]},
                "drop_vs_appt": {k: drop[k] - A["t_req"][k] for k in drop
                                 if A["appointment"][k]}}

    def best_schedule(self, stops):
        """Try several start times for the first pickup; keep the shortest run."""
        cfg, A, first = self.cfg, self.A, stops[0][0]
        t = A["t_req"][first]
        if A["appointment"][first]:
            lo = t - cfg["appt_early_min"] - A["max_ride"][first]
            hi = max(lo, t + cfg["appt_late_min"] - A["solo_ride"][first])
        else:
            lo = t - (cfg["window_min"] if cfg["allow_early_pickup"] else 0)
            hi = t + cfg["window_min"]
        best = None
        for start in np.linspace(lo, hi, cfg["start_grid"]):
            r = self._run(stops, start)
            if r and (best is None or r["duration"] < best["duration"]):
                best = r
        return best

    def insert(self, base_stops, t):
        """Insert trip t's pickup and drop-off at every position of base_stops."""
        best, L = None, len(base_stops)
        for i in range(L + 1):
            for j in range(i + 1, L + 2):
                s = list(base_stops)
                s.insert(i, (t, "O"))
                s.insert(j, (t, "D"))
                r = self.best_schedule(s)
                if r and (best is None or r["duration"] < best["duration"]):
                    best = r
        if best is None:
            return None
        members = {k for k, _ in best["stops"]}
        best["saved"] = sum(self.A["solo_ride"][k] for k in members) - best["duration"]
        return best if best["saved"] > self.cfg["min_saved_min"] else None


# ---------------------------------------------------------------- group enumeration
def neighbours(T, cfg):
    # Estimated pickup time: the request for pickup-time trips, appointment minus solo ride
    # for appointment trips. Pairs further apart than prefilter_time_min can't share a run.
    t = np.where(T.appointment.values, T.t_req.values - T.solo_ride.values, T.t_req.values)
    olat, olon, dlat, dlon = T.olat.values, T.olon.values, T.dlat.values, T.dlon.values
    adj = [set() for _ in range(len(T))]
    for i in range(len(T) - 1):
        j = np.arange(i + 1, len(T))
        m = ((np.abs(t[j] - t[i]) <= cfg["prefilter_time_min"])
             & (hav_vec(olat[i], olon[i], olat[j], olon[j]) <= cfg["max_origin_mi"])
             & (hav_vec(dlat[i], dlon[i], dlat[j], dlon[j]) <= cfg["max_dest_mi"]))
        for k in j[m]:
            adj[i].add(int(k)); adj[int(k)].add(i)
    # Keep only each trip's closest neighbours (origin + destination distance) so the
    # group search stays tractable; a pair is kept if either trip lists the other.
    cap = cfg.get("max_neighbours")
    if cap:
        keep = [set() for _ in range(len(T))]
        for i, nb in enumerate(adj):
            if len(nb) > cap:
                nbl = np.array(sorted(nb))
                score = (hav_vec(olat[i], olon[i], olat[nbl], olon[nbl])
                         + hav_vec(dlat[i], dlon[i], dlat[nbl], dlon[nbl]))
                nb = set(int(x) for x in nbl[np.argsort(score, kind="stable")[:cap]])
            for k in nb:
                keep[i].add(k); keep[k].add(i)
        adj = keep
    return adj


def enumerate_groups(T, cfg, sim):
    adj = neighbours(T, cfg)
    level = {}
    for i in range(len(T)):                                      # pairs (exhaustive)
        for j in adj[i]:
            if j > i:
                r = sim.insert([(i, "O"), (i, "D")], j)
                if r:
                    level[frozenset((i, j))] = r
    groups = dict(level)
    for size in range(3, cfg["max_group_size"] + 1):            # extend to 3..K
        nxt = {}
        for g, r in level.items():
            common = set.intersection(*(adj[m] for m in g)) - g
            for t in common:
                key = g | {t}
                cand = sim.insert(r["stops"], t)
                if cand and (key not in nxt or cand["duration"] < nxt[key]["duration"]):
                    nxt[key] = cand
            if len(nxt) > cfg["max_groups_per_size"]:
                sys.exit(f"More than {cfg['max_groups_per_size']} groups of size {size}; "
                         "tighten the distance/time limits or raise max_groups_per_size.")
        groups.update(nxt)
        level = nxt
    return groups


def assign(groups, N, T, cfg):
    """Exact set packing: remove the most runs, then save the most vehicle-minutes."""
    if not groups:
        return [], 0, 0.0
    items = sorted(groups.items(), key=lambda kv: sorted(kv[0]))   # fixed order
    M = float(T.solo_ride.sum()) + 1.0          # savings term sums to < 1, so one extra
    c = -np.array([(len(g) - 1) + r["saved"] / M for g, r in items])   # run removed always wins
    rows, cols = [], []
    for gi, (g, _) in enumerate(items):
        for k in g:
            rows.append(k); cols.append(gi)
    A = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(N, len(items)))
    res = milp(c, constraints=LinearConstraint(A, 0, 1), integrality=np.ones(len(items)),
               bounds=Bounds(0, 1),
               options={"mip_rel_gap": cfg["solver_gap"], "time_limit": cfg["solver_time_limit_s"]})
    if res.x is None:
        sys.exit(f"Assignment solver found no solution: {res.message}")
    # Bound on how far from optimal the chosen groups can be (0 = proven optimal).
    gap = getattr(res, "mip_gap", np.nan)
    return [items[i] for i in np.where(res.x > 0.5)[0]], res.status, gap


# ---------------------------------------------------------------- one run + metrics
def run_one(cfg, T):
    sim = Sim(T, cfg)
    groups = enumerate_groups(T, cfg, sim)
    chosen, solver_status, solver_gap = assign(groups, len(T), T, cfg)

    log = T[["trip_id", "origin_tract", "dest_name", "outbound", "appointment", "t_req",
             "wheelchair", "trip_miles", "solo_ride", "max_ride"]].copy()
    log["status"], log["group_id"], log["group_size"], log["stop_order"] = "solo", -1, 1, ""
    log["pickup_delay_min"] = np.where(T.appointment, np.nan, 0.0)
    log["drop_vs_appt_min"] = np.where(T.appointment, 0.0, np.nan)
    log["added_ride_min"] = 0.0
    pooled_duration = 0.0
    status, gid_a, size_a, order_a = (log.status.to_numpy(object, copy=True), log.group_id.to_numpy(copy=True),
                                      log.group_size.to_numpy(copy=True), log.stop_order.to_numpy(object, copy=True))
    delay_a, added_a = log.pickup_delay_min.to_numpy(copy=True), log.added_ride_min.to_numpy(copy=True)
    appt_a = log.drop_vs_appt_min.to_numpy(copy=True)
    for gid, (g, r) in enumerate(chosen):
        pooled_duration += r["duration"]
        order = " ".join(f"{kind}{k}" for k, kind in r["stops"])
        for k in g:
            status[k], gid_a[k], size_a[k], order_a[k] = "pooled", gid, len(g), order
            added_a[k] = r["added"][k]
            if k in r["pickup_delay"]:
                delay_a[k] = r["pickup_delay"][k]
            if k in r["drop_vs_appt"]:
                appt_a[k] = r["drop_vs_appt"][k]
    log["status"], log["group_id"], log["group_size"], log["stop_order"] = status, gid_a, size_a, order_a
    log["pickup_delay_min"], log["added_ride_min"], log["drop_vs_appt_min"] = delay_a, added_a, appt_a

    N, G = len(T), len(chosen)
    U = int((log.status == "solo").sum())
    pooled = log[log.status == "pooled"]
    veh_control = T.solo_ride.sum()
    veh_test = pooled_duration + T.solo_ride[log.status == "solo"].sum()
    sizes = pd.Series([len(g) for g, _ in chosen]).value_counts()
    summary = {
        "N": N, "groups_G": G, "unmatched_U": U,
        "p_share_pooled": (N - U) / N,
        "riders_per_run": N / (G + U),
        "runs_control": N, "runs_test": G + U,
        **{f"groups_size_{s}": int(sizes.get(s, 0)) for s in range(2, cfg["max_group_size"] + 1)},
        "veh_min_control": veh_control, "veh_min_test": veh_test,
        "veh_min_saved_pct": 1 - veh_test / veh_control,
        "feasible_groups_found": len(groups),
        "solver_status": solver_status, "solver_gap": solver_gap,
        "mean_solo_ride_min": T.solo_ride.mean(),
        "mean_added_ride_min_pooled": pooled.added_ride_min.mean() if len(pooled) else np.nan,
        "max_added_ride_min_pooled": pooled.added_ride_min.max() if len(pooled) else np.nan,
        "share_trips_appointment": T.appointment.mean(),
        "mean_pickup_wait_min_pooled": pooled.pickup_delay_min.mean() if len(pooled) else np.nan,
        "share_pooled_picked_up_early": (pooled.pickup_delay_min < -0.5).mean() if len(pooled) else np.nan,
        "mean_min_before_appt_pooled": -pooled.drop_vs_appt_min.mean() if len(pooled) else np.nan,
        "max_ride_used_share_pooled": ((pooled.solo_ride + pooled.added_ride_min) / pooled.max_ride).max()
                                      if len(pooled) else np.nan,
    }
    return log, summary


def sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


CODE_PATH = os.path.abspath(__file__)
CONFIG_PATH = None


def main():
    global CONFIG_PATH
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default="outputs")
    args = ap.parse_args()
    CONFIG_PATH = os.path.abspath(args.config)
    args.out = os.path.abspath(args.out)

    base = json.load(open(args.config))
    os.chdir(os.path.dirname(os.path.abspath(args.config)))   # data/ paths are relative to config
    tracts, dests = load_inputs(base)
    print(f"Trips per run: {trips_in_area_peak(base)}  |  tracts: {len(tracts)}  |  destinations: {len(dests)}")
    os.makedirs(args.out, exist_ok=True)

    scenarios = [("base", {})] + [(s["name"], {k: v for k, v in s.items() if k != "name"})
                                  for s in base.get("sensitivity", [])]
    rows = []
    for name, override in scenarios:
        cfg = copy.deepcopy(base); cfg.update(override)
        sdir = os.path.join(args.out, name); os.makedirs(sdir, exist_ok=True)
        seeds = cfg["seeds"] if name == "base" else cfg.get("sensitivity_seeds", cfg["seeds"])
        for seed in seeds:
            T = generate_trips(cfg, tracts, dests, seed)   # same seed -> same trips everywhere
            log, s = run_one(cfg, T)
            log.to_csv(os.path.join(sdir, f"trips_seed{seed}.csv"), index=False)
            rows.append({"scenario": name, "changed": json.dumps(override), "seed": seed, **s})
            print(f"{name:>14} seed {seed:>3}: mean solo ride={T.solo_ride.mean():.0f} min  "
                  f"p={s['p_share_pooled']:.1%}  "
                  f"riders/run={s['riders_per_run']:.2f}  "
                  f"veh-min saved={s['veh_min_saved_pct']:.1%}", flush=True)

    summ = pd.DataFrame(rows)
    summ.to_csv(os.path.join(args.out, "summary.csv"), index=False)
    spread = summ.groupby("scenario", sort=False)[
        ["p_share_pooled", "riders_per_run", "veh_min_saved_pct"]].agg(["mean", "min", "max"])
    print("\nAcross seeds:\n", spread.round(3).to_string())

    import platform, scipy
    manifest = {"run_at": datetime.now().isoformat(timespec="seconds"),
                "versions": {"python": platform.python_version(), "numpy": np.__version__,
                             "pandas": pd.__version__, "scipy": scipy.__version__},
                "config": base, "trips_per_run": trips_in_area_peak(base),
                "config_sha256": sha256(CONFIG_PATH), "code_sha256": sha256(CODE_PATH),
                "tracts_sha256": sha256(base["tracts_csv"]),
                "blocks_sha256": sha256(base["blocks_csv"]),
                "dests_sha256": sha256(base["dests_csv"])}
    if os.path.exists("data/derived.json"):
        manifest["derived"] = json.load(open("data/derived.json"))
        manifest["derived_sha256"] = sha256("data/derived.json")
    json.dump(manifest, open(os.path.join(args.out, "manifest.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
