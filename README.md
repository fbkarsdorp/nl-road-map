# Alle wegen leiden naar…

A poster-style map of the Netherlands where every junction routes to one place. Line width shows how much land (or how many people) flows down each road, and the biggest branches get their own colour. It's a Dutch take on [All Roads Lead To](https://all-roads-lead-to.netlify.app/).

Two networks:

- **Car roads**: OSM motorway → tertiary (plus links) and car ferries. Speeds come from road class and `maxspeed`, and one-way rules are respected.
- **Fietsroutes**: the national *fietsknooppunten* network (OSM `rcn`/`ncn` route relations), plus bike ferries and the Wadden-island cycle paths. It uses a steady 16 km/h, and hovering a junction lists the knooppunt numbers along the route.

Three map types: **one place**, **rival cities** (each junction drains to the nearest of N cities), and **meet halfway** (the fairest or least-total meeting spot for up to 12 travellers).

The page is in English or Dutch (the EN/NL switch in the panel). It follows the browser language at first, remembers the choice, and `?lang=nl` or `?lang=en` in a link forces one.

**Live:** https://karsdorp.io/nl-road-map/ (try `#bike-groningen` or `#car-maastricht`)

## Run locally

```sh
python3 -m http.server 8000
# open http://localhost:8000/index.html  (e.g. #bike-groningen or #car-maastricht)
```

## Rebuild the data

`build/build.py` turns Overpass extracts into compact gzip'd graphs in `data/` (`car.bin`, `bike.bin`, `nl.json.gz`) plus base64 copies (`*.txt`). The page loads the `.txt` copies, which also works on hosts that only serve text, so only those are committed.

| input | source |
|---|---|
| `roads.json` | Overpass: `highway~motorway…tertiary_link` + `route=ferry` in NL |
| `bike.json` | Overpass: ways in `route=bicycle` relations with `network~rcn|ncn` in NL, plus `rcn_ref` nodes |
| `islands.json` | Overpass: cycleways/minor roads on Vlieland, Terschelling, Ameland, Schiermonnikoog |
| `minor.json` *(optional)* | Overpass: `way[highway=unclassified](area.nl)`, adds minor rural roads to the car graph. One file, or tiles in `minor/*.json`. Leave out, or pass `--no-minor`, for main roads only |
| `places.json` | Overpass: `place=city|town|village|hamlet` nodes (population for line widths) |
| `gem.geojson`, `prov.geojson` | cartomap.github.io/nl (CBS municipal/provincial outlines, 2023) |

```sh
pip install numpy scipy shapely
python build/build.py <dir-with-inputs> data          # both modes
python build/build.py <dir-with-inputs> data bike     # one mode
```

The build splits ways at junctions, contracts degree-2 chains, and keeps the largest strongly connected component. Stranded ferry terminals get two-way tie-in links. Land area (a 250 m grid) and population (places spread with a Gaussian kernel) are then assigned to the nearest point on the network.

Data © OpenStreetMap contributors (ODbL).
