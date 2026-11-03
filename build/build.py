"""Build compact routing graphs for the NL road / cycle-network poster.

Usage: python build.py <scratch-data-dir> <out-dir>

Inputs (from Overpass + cartomap):
  roads.json   motorway..tertiary (+links) and route=ferry ways
  minor.json   highway=unclassified ways or minor/*.json tiles (optional; skip with --no-minor)
  bike.json    ways that are members of rcn/ncn bicycle route relations (+ rcn_ref nodes)
  places.json  place=city|town|village|hamlet nodes
  gem.geojson  municipal boundaries (land outline)

Outputs:
  car.bin, bike.bin   gzip'd: [u32 headerLen][JSON header][pad8][typed arrays]
  nl.json.gz          land outline + places (shared)
"""
import glob, gzip, json, math, os, re, struct, sys
from collections import defaultdict

import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from shapely.geometry import shape, LineString, Point
from shapely.ops import unary_union
from shapely.prepared import prep

SRC, OUT = sys.argv[1], sys.argv[2]
MINOR = "--no-minor" not in sys.argv  # --no-minor rebuilds the main-roads-only car graph
sys.argv = [a for a in sys.argv if a != "--no-minor"]
BBOX = (3.2, 50.70, 7.30, 53.65)  # European NL only (drops Caribbean NL)
LON0, LAT0 = 5.3, 52.15
Q = 5.0  # quantisation of stored coordinates, metres


def proj(lon, lat):
    # sinusoidal around 5.3E; fine for an area the size of NL
    return ((lon - LON0) * 111320.0 * math.cos(math.radians(lat)), (lat - LAT0) * 110950.0)


def in_bbox(lon, lat):
    return BBOX[0] <= lon <= BBOX[2] and BBOX[1] <= lat <= BBOX[3]


# ---------------------------------------------------------------- land
gem = json.load(open(f"{SRC}/gem.geojson"))
land_ll = unary_union([shape(f["geometry"]).buffer(0) for f in gem["features"]])
land_ll_buf = prep(land_ll.buffer(0.01))  # ~1 km, keeps border-crossing stubs short


def _projxy(x, y):
    x = np.asarray(x); y = np.asarray(y)
    return (x - LON0) * 111320.0 * np.cos(np.radians(y)), (y - LAT0) * 110950.0


from shapely.ops import transform as _tf
land = _tf(_projxy, land_ll)
minx, miny, maxx, maxy = land.bounds
PAD = 3000
X0, Y0 = minx - PAD, miny - PAD
W_M, H_M = maxx - minx + 2 * PAD, maxy - miny + 2 * PAD
print("extent km", W_M / 1000, H_M / 1000)


def qx(x): return int(round((x - X0) / Q))
def qy(y): return int(round((y - Y0) / Q))


# ---------------------------------------------------------------- places
places = []
for e in json.load(open(f"{SRC}/places.json"))["elements"]:
    t = e["tags"]; pl = t.get("place")
    if pl not in ("city", "town", "village", "hamlet") or not in_bbox(e["lon"], e["lat"]):
        continue
    if not land_ll_buf.contains(Point(e["lon"], e["lat"])):
        continue
    name = t.get("name")
    if not name:
        continue
    try:
        pop = int(re.sub(r"[^0-9]", "", t.get("population", "")) or 0)
    except ValueError:
        pop = 0
    if pop <= 0:
        pop = {"city": 50000, "town": 10000, "village": 1000, "hamlet": 120}[pl]
    x, y = proj(e["lon"], e["lat"])
    places.append(dict(name=name, type=pl, pop=pop, x=x, y=y, lon=e["lon"], lat=e["lat"]))
print("places", len(places), "pop sum", sum(p["pop"] for p in places))

# ---------------------------------------------------------------- grid over land for weights
CELL = 250.0
gx = np.arange(X0 + CELL / 2, X0 + W_M, CELL)
gy = np.arange(Y0 + CELL / 2, Y0 + H_M, CELL)
GX, GY = np.meshgrid(gx, gy)
from shapely import contains_xy
inland = contains_xy(land, GX.ravel(), GY.ravel())
cells = np.column_stack([GX.ravel()[inland], GY.ravel()[inland]])
cell_area = np.full(len(cells), CELL * CELL / 1e6)  # km2
print("land cells", len(cells), "area km2", cell_area.sum())

# population spread with a gaussian kernel per place
cell_pop = np.zeros(len(cells))
ctree = cKDTree(cells)
for p in places:
    sig = min(4000.0, max(150.0, 150.0 * math.sqrt(p["pop"] / 1000.0)))
    idx = ctree.query_ball_point([p["x"], p["y"]], 2.5 * sig)
    if not idx:
        _, i = ctree.query([p["x"], p["y"]]); idx = [i]
    d2 = ((cells[idx] - [p["x"], p["y"]]) ** 2).sum(1)
    w = np.exp(-d2 / (2 * sig * sig)); w /= w.sum()
    cell_pop[idx] += p["pop"] * w


# ---------------------------------------------------------------- helpers
def parse_duration(s):
    if not s:
        return None
    s = s.strip()
    m = re.fullmatch(r"PT(?:(\d+)H)?(?:(\d+)M)?", s)
    if m:
        return (int(m.group(1) or 0) * 60 + int(m.group(2) or 0)) * 60
    m = re.fullmatch(r"(\d+):(\d+)(?::(\d+))?", s)
    if m:
        return int(m.group(1)) * 3600 + int(m.group(2)) * 60 + int(m.group(3) or 0)
    if s.isdigit():
        return int(s) * 60
    return None


def seg_len(coords):
    return sum(math.hypot(coords[i + 1][0] - coords[i][0], coords[i + 1][1] - coords[i][1]) for i in range(len(coords) - 1))


CAR_SPEED = dict(motorway=105, motorway_link=60, trunk=85, trunk_link=50, primary=65, primary_link=45,
                 secondary=55, secondary_link=40, tertiary=45, tertiary_link=35, unclassified=40)


def car_allowed(t):
    """Minor roads are often closed to cars (farm tracks, bus gates, private estates)."""
    if t.get("highway") != "unclassified": return True
    if t.get("area") == "yes": return False
    for k in ("motor_vehicle", "motorcar", "vehicle", "access"):
        v = t.get(k)
        if v is not None:
            return v not in ("no", "private", "agricultural", "forestry", "delivery", "destination;agricultural")
    return True


def car_speed(t):
    hw = t["highway"]
    base = CAR_SPEED[hw]
    ms = re.match(r"\d+", t.get("maxspeed", "") or "")
    if ms and not hw.endswith("_link"):
        v = int(ms.group())
        f = 0.92 if v >= 100 else 0.85 if v >= 70 else 0.75 if v >= 50 else 0.7
        return max(15, min(125, v * f))
    return base


def car_oneway(t):
    ow = t.get("oneway")
    if ow in ("yes", "1", "true"): return 1
    if ow == "-1": return -1
    if ow == "no": return 0
    if t["highway"] in ("motorway", "motorway_link") or t.get("junction") in ("roundabout", "circular"): return 1
    return 0


def ferry_ok(t, mode):
    if t.get("access") in ("private", "no") and t.get(mode == "car" and "motor_vehicle" or "bicycle") != "yes":
        return False
    if mode == "car":
        return t.get("motorcar") == "yes" or (t.get("motor_vehicle") == "yes" and t.get("motorcar") != "no")
    b = t.get("bicycle")
    return b in ("yes", "designated") or (b is None and t.get("motor_vehicle") == "yes")


# ---------------------------------------------------------------- build one mode
def build(mode):
    print("=====", mode)
    roads_raw = json.load(open(f"{SRC}/roads.json"))["elements"]
    nodes = {e["id"]: (e["lon"], e["lat"]) for e in roads_raw if e["type"] == "node"}
    road_ways, ferry_ways = [], []
    for e in roads_raw:
        if e["type"] != "way": continue
        t = e.get("tags", {})
        if t.get("route") == "ferry":
            if ferry_ok(t, mode): ferry_ways.append(e)
        elif mode == "car" and t.get("highway") in CAR_SPEED:
            road_ways.append(e)
    # highway=unclassified: minor.json, or tiles in minor/*.json (ways on tile edges appear twice)
    mfiles = [f"{SRC}/minor.json"] if os.path.exists(f"{SRC}/minor.json") else sorted(glob.glob(f"{SRC}/minor/*.json"))
    if mode == "car" and MINOR and mfiles:
        seen_m, n0 = {e["id"] for e in road_ways}, len(road_ways)
        for fn in mfiles:
            mraw = json.load(open(fn))["elements"]
            for e in mraw:
                if e["type"] == "node": nodes[e["id"]] = (e["lon"], e["lat"])
            for e in mraw:
                if e["type"] == "way" and e["id"] not in seen_m and car_allowed(e.get("tags", {})):
                    seen_m.add(e["id"]); road_ways.append(e)
        print("  minor roads added:", len(road_ways) - n0, "from", len(mfiles), "file(s)")
    if mode == "bike":
        braw = json.load(open(f"{SRC}/bike.json"))["elements"]
        for e in braw:
            if e["type"] == "node": nodes[e["id"]] = (e["lon"], e["lat"])
        seen = set()
        for e in braw:
            if e["type"] != "way" or e["id"] in seen: continue
            seen.add(e["id"])
            t = e.get("tags", {})
            if t.get("route") == "ferry":
                if e["id"] not in {f["id"] for f in ferry_ways}: ferry_ways.append(e)
            elif t.get("highway") and t["highway"] not in ("proposed", "construction", "steps", "platform"):
                road_ways.append(e)
        if os.path.exists(f"{SRC}/islands.json"):  # Wadden islands lack knooppunten in the data
            iraw = json.load(open(f"{SRC}/islands.json"))["elements"]
            for e in iraw:
                if e["type"] == "node": nodes[e["id"]] = (e["lon"], e["lat"])
            for e in iraw:
                if e["type"] == "way" and e["id"] not in seen:
                    seen.add(e["id"]); road_ways.append(e)
        rcn = {}
        for e in braw:
            if e["type"] == "node" and "rcn_ref" in e.get("tags", {}):
                rcn[e["id"]] = e["tags"]["rcn_ref"]
    del roads_raw

    def way_ok(w):
        return all(n in nodes for n in w["nodes"]) and any(in_bbox(*nodes[n]) for n in w["nodes"])

    road_ways = [w for w in road_ways if way_ok(w) and all(in_bbox(*nodes[n]) for n in w["nodes"][:1])]
    ferry_ways = [w for w in ferry_ways if way_ok(w) and all(in_bbox(*nodes[n]) for n in w["nodes"])]
    ferry_ways = [w for w in ferry_ways if (parse_duration(w["tags"].get("duration")) or 0) < 3 * 3600]
    print("road ways", len(road_ways), "ferries", len(ferry_ways))

    # unified way records: (node list, fwd allowed, bwd allowed, speed km/h or fixed seconds, name, kind)
    recs = []
    for w in road_ways:
        t = w["tags"]
        if mode == "car":
            ow = car_oneway(t); sp = car_speed(t)
            nl = w["nodes"][::-1] if ow == -1 else w["nodes"]
            nm = (t.get("ref") or t.get("name") or "").split(";")[0]
            recs.append(dict(n=nl, ow=ow != 0, sp=sp, name=nm, ferry=0))
        else:
            hw = t["highway"]
            sp = 16.0
            if hw in ("track", "path", "bridleway") and t.get("surface") not in ("asphalt", "paved", "concrete", "paving_stones", "concrete:plates"):
                sp = 13.0
            if hw in ("pedestrian", "footway", "living_street"):
                sp = 10.0
            recs.append(dict(n=w["nodes"], ow=False, sp=sp, name=t.get("name", ""), ferry=0))

    # snap ferry ends onto the network
    road_node_ids = list({n for r in recs for n in r["n"]})
    rxy = np.array([proj(*nodes[n]) for n in road_node_ids])
    rtree = cKDTree(rxy)
    road_node_set = set(road_node_ids)
    for w in ferry_ways:
        t = w["tags"]
        nl = w["nodes"]
        coords = [proj(*nodes[n]) for n in nl]
        L = seg_len(coords)
        dur = parse_duration(t.get("duration"))
        sec = (dur if dur else L / (18 / 3.6)) + 600  # + 10 min boarding/wait
        recs.append(dict(n=nl, ow=False, fixed=sec, name=t.get("name", "Veer"), ferry=1))
        # two-way connectors from each terminal to a few nearby network nodes, so one-way
        # terminal lanes cannot strand an island outside the strongly connected network
        for end in (nl[0], nl[-1]):
            ds, ii = rtree.query(proj(*nodes[end]), k=6)
            lim = 2000 if mode == "bike" else 800
            for d, i in zip(ds, ii):
                if d < lim and road_node_ids[i] != end:
                    recs.append(dict(n=[end, road_node_ids[i]], ow=False, sp=15.0, name="", ferry=0))

    # junctions: way endpoints + nodes shared by several ways
    use = defaultdict(int)
    for r in recs:
        for n in r["n"]: use[n] += 1
        use[r["n"][0]] += 1; use[r["n"][-1]] += 1
    if mode == "bike":
        for n in rcn:
            if n in use: use[n] += 1  # knooppunten are always junctions

    links = []  # dict(u, v, pts(list of node ids), ow, sec, len, name, ferry)
    for r in recs:
        nl = r["n"]; start = 0
        for i in range(1, len(nl)):
            if use[nl[i]] >= 2 or i == len(nl) - 1:
                seg = nl[start:i + 1]
                if len(seg) >= 2 and not (seg[0] == seg[-1] and len(seg) <= 2):
                    coords = [proj(*nodes[n]) for n in seg]
                    L = seg_len(coords)
                    if "fixed" in r:
                        Ltot = max(1.0, seg_len([proj(*nodes[n]) for n in nl]))
                        sec = r["fixed"] * L / Ltot
                    else:
                        sec = L / (r["sp"] / 3.6)
                    links.append(dict(u=seg[0], v=seg[-1], pts=seg, ow=r["ow"], sec=sec, len=L, name=r["name"], ferry=r["ferry"]))
                start = i
    print("raw links", len(links))

    # drop links clearly outside NL
    def inside(n): return land_ll_buf.contains(Point(*nodes[n]))
    cache = {}
    def ins(n):
        if n not in cache: cache[n] = inside(n)
        return cache[n]
    links = [l for l in links if ins(l["u"]) or ins(l["v"])]

    # dedupe parallel identical links (keep fastest)
    best = {}
    for l in links:
        k = (l["u"], l["v"], l["ow"]) if l["ow"] else (min(l["u"], l["v"]), max(l["u"], l["v"]), False)
        if k not in best or l["sec"] < best[k]["sec"]: best[k] = l
    links = [l for l in best.values() if l["u"] != l["v"]]

    # largest (strongly) connected component
    ids = {}
    for l in links:
        for n in (l["u"], l["v"]):
            if n not in ids: ids[n] = len(ids)
    def scc():
        rows, cols = [], []
        for l in links:
            a, b = ids[l["u"]], ids[l["v"]]
            rows.append(a); cols.append(b)
            if not l["ow"]: rows.append(b); cols.append(a)
        G = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(ids), len(ids))).tocsr()
        nc, lab = connected_components(G, directed=True, connection="strong")
        return nc, lab, np.bincount(lab).argmax()
    nc, lab, big = scc()
    # ferry terminals stranded outside the main component (one-way terminal lanes):
    # tie them to the nearest main-network nodes with short two-way connectors
    id_list = list(ids.keys())
    main = [n for n in id_list if lab[ids[n]] == big]
    mtree = cKDTree(np.array([proj(*nodes[n]) for n in main]))
    added = 0
    for l in [l for l in links if l["ferry"]]:
        for end in (l["u"], l["v"]):
            if lab[ids[end]] == big: continue
            ds, ii = mtree.query(proj(*nodes[end]), k=3)
            for d, i in zip(ds, ii):
                if d < 1500:
                    L = max(d, 1.0)
                    links.append(dict(u=end, v=main[i], pts=[end, main[i]], ow=False, sec=L / (15 / 3.6), len=L, name="", ferry=0)); added += 1
    if added:
        nc, lab, big = scc()
    print("ferry tie-ins", added)
    keep = {n for n, i in ids.items() if lab[i] == big}
    links = [l for l in links if l["u"] in keep and l["v"] in keep]
    print("components", nc, "kept nodes", len(keep), "links", len(links))

    # contract degree-2 chains with compatible direction
    inc = defaultdict(list)
    for i, l in enumerate(links):
        inc[l["u"]].append(i); inc[l["v"]].append(i)
    alive = [True] * len(links)
    protect = set(rcn.keys()) if mode == "bike" else set()
    changed = True
    while changed:
        changed = False
        for n in list(inc.keys()):
            li = [i for i in inc[n] if alive[i]]
            inc[n] = li
            if len(li) != 2 or n in protect: continue
            a, b = links[li[0]], links[li[1]]
            if a is b or a["ferry"] != b["ferry"]: continue
            if a["ow"] != b["ow"]: continue
            # orient a: x -> n, b: n -> y
            if a["ow"]:
                if a["v"] == n and b["u"] == n: pass
                elif b["v"] == n and a["u"] == n: a, b = b, a
                else: continue
                pa, pb = a["pts"], b["pts"]
            else:
                pa = a["pts"] if a["v"] == n else a["pts"][::-1]
                pb = b["pts"] if b["u"] == n else b["pts"][::-1]
            if pa[0] == pb[-1]: continue  # would make a loop
            new = dict(u=pa[0], v=pb[-1], pts=pa + pb[1:], ow=a["ow"], sec=a["sec"] + b["sec"], len=a["len"] + b["len"],
                       name=a["name"] if a["len"] >= b["len"] else b["name"], ferry=a["ferry"])
            alive[li[0]] = alive[li[1]] = False
            links.append(new); alive.append(True)
            k = len(links) - 1
            inc[new["u"]].append(k); inc[new["v"]].append(k)
            inc[n] = []
            changed = True
    links = [l for i, l in enumerate(links) if alive[i]]
    jids = {}
    for l in links:
        for n in (l["u"], l["v"]):
            if n not in jids: jids[n] = len(jids)
    NJ, NL = len(jids), len(links)
    print("contracted: junctions", NJ, "links", NL)

    jxy = np.zeros((NJ, 2))
    for n, i in jids.items(): jxy[i] = proj(*nodes[n])

    # geometry: simplify intermediate points
    names = {"": 0}; name_list = [""]
    lu = np.zeros(NL, np.uint32); lv = np.zeros(NL, np.uint32)
    lt = np.zeros(NL, np.float32); lk = np.zeros(NL, np.float32)
    lf = np.zeros(NL, np.uint8); ln = np.zeros(NL, np.uint32)
    gptr = np.zeros(NL + 1, np.uint32); gpts = []
    samples, s_link, s_t = [], [], []
    TOL = 6.0 if mode == "car" else 5.0
    for i, l in enumerate(links):
        lu[i] = jids[l["u"]]; lv[i] = jids[l["v"]]
        lt[i] = l["sec"]; lk[i] = l["len"]
        lf[i] = (1 if l["ow"] else 0) | (2 if l["ferry"] else 0)
        nm = l["name"] or ""
        if nm not in names: names[nm] = len(name_list); name_list.append(nm)
        ln[i] = names[nm]
        coords = [proj(*nodes[n]) for n in l["pts"]]
        if len(coords) > 2:
            sim = list(LineString(coords).simplify(TOL, preserve_topology=False).coords)
            for x, y in sim[1:-1]: gpts.append((qx(x), qy(y)))
        gptr[i + 1] = len(gpts)
        # weight samples every ~150 m along the link
        L = max(l["len"], 1.0); ns = max(2, int(L / 150) + 1)
        ls = LineString(coords)
        for k in range(ns):
            t = k / (ns - 1)
            p = ls.interpolate(t, normalized=True)
            samples.append((p.x, p.y)); s_link.append(i); s_t.append(t)
    samples = np.array(samples); s_link = np.array(s_link); s_t = np.array(s_t)

    # assign land/pop cells to nearest sample, split between link ends
    stree = cKDTree(samples)
    _, si = stree.query(cells)
    jarea = np.zeros(NJ); jpop = np.zeros(NJ)
    li_ = s_link[si]; t_ = s_t[si]
    np.add.at(jarea, lu[li_], cell_area * (1 - t_)); np.add.at(jarea, lv[li_], cell_area * t_)
    np.add.at(jpop, lu[li_], cell_pop * (1 - t_)); np.add.at(jpop, lv[li_], cell_pop * t_)
    print("area", jarea.sum(), "pop", jpop.sum())

    jx = np.array([qx(x) for x in jxy[:, 0]], np.uint16)
    jy = np.array([qy(y) for y in jxy[:, 1]], np.uint16)
    gp = np.array(gpts, np.uint16).reshape(-1, 2) if gpts else np.zeros((0, 2), np.uint16)

    arrays = dict(jx=jx, jy=jy, jarea=jarea.astype(np.float32), jpop=jpop.astype(np.float32),
                  lu=lu, lv=lv, lt=lt, lk=lk, lf=lf, ln=ln, gptr=gptr,
                  gx=np.ascontiguousarray(gp[:, 0]), gy=np.ascontiguousarray(gp[:, 1]))
    if mode == "bike":
        jref = np.zeros(NJ, np.uint16)
        for n, r in rcn.items():
            if n in jids and r.isdigit(): jref[jids[n]] = min(int(r), 65535)
        arrays["jref"] = jref

    # nearest junction for each place
    jtree = cKDTree(jxy)
    _, pj = jtree.query(np.array([[p["x"], p["y"]] for p in places]))

    header = dict(mode=mode, NJ=NJ, NL=NL, Q=Q, names=name_list, placeJ=pj.tolist(), arrays=[])
    blobs = []; off = 0
    for k, a in arrays.items():
        a = np.ascontiguousarray(a)
        header["arrays"].append(dict(name=k, type=str(a.dtype), length=int(a.size), offset=off))
        b = a.tobytes(); pad = (-len(b)) % 8
        blobs.append(b + b"\0" * pad); off += len(b) + pad
    hb = json.dumps(header, ensure_ascii=False, separators=(",", ":")).encode()
    hpad = (-(4 + len(hb))) % 8
    raw = struct.pack("<I", len(hb)) + hb + b" " * hpad + b"".join(blobs)
    with gzip.open(f"{OUT}/{mode}.bin", "wb", compresslevel=9) as f: f.write(raw)
    print("wrote", mode, len(raw) / 1e6, "MB raw")
    write_b64(f"{OUT}/{mode}.bin", f"{OUT}/{mode}.txt")


# ---------------------------------------------------------------- shared json
def ring_q(r):
    out = []
    for x, y in r.coords:
        p = (qx(x), qy(y))
        if not out or out[-1] != p: out.append(p)
    return [c for p in out for c in p]


land_s = land.simplify(40)
polys = list(land_s.geoms) if land_s.geom_type == "MultiPolygon" else [land_s]
land_rings = [[ring_q(p.exterior)] + [ring_q(h) for h in p.interiors] for p in polys if p.area > 2e5]
prov = json.load(open(f"{SRC}/prov.geojson"))
prov_lines = []
pb = unary_union([_tf(_projxy, shape(f["geometry"])).boundary for f in prov["features"]])
pb = pb.intersection(land.buffer(-500)).simplify(80)
for g in getattr(pb, "geoms", [pb]):
    if g.geom_type == "LineString" and g.length > 2000: prov_lines.append(ring_q(g))
shared = dict(X0=X0, Y0=Y0, W=W_M, H=H_M, Q=Q, LON0=LON0, LAT0=LAT0, land=land_rings, prov=prov_lines,
              places=[dict(n=p["name"], t=p["type"][0], p=p["pop"], x=qx(p["x"]), y=qy(p["y"]),
                           lo=round(p["lon"], 5), la=round(p["lat"], 5)) for p in places])
with gzip.open(f"{OUT}/nl.json.gz", "wt", compresslevel=9, encoding="utf8") as f:
    json.dump(shared, f, ensure_ascii=False, separators=(",", ":"))

import base64


def write_b64(src, dst):
    with open(src, "rb") as f, open(dst, "w") as g:
        g.write(base64.b64encode(f.read()).decode())


write_b64(f"{OUT}/nl.json.gz", f"{OUT}/nl.txt")
for m in sys.argv[3:] or ["car", "bike"]:
    build(m)
