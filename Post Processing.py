#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import geopandas as gpd
import re
import panel as pn
import folium

from IPython.display import display


# In[2]:


# Load files
df_all = pd.read_csv("QUERY_all_schools_de.csv", dtype={"schulnummer": "string"})
df_named = pd.read_csv("QUERY_named_after_de.csv")
gdf_nuts1 = gpd.read_file("NUTS5000_N1.shp")

# Extract QIDs vectorized
qid_pat = r"(Q\d+)$"
df_all["QID"] = df_all["schule"].astype(str).str.extract(qid_pat)
df_named["QID"] = df_named["QID"].astype(str).str.extract(qid_pat)

# Exclude non-BBiG training occupations
mask_has_bbig = df_all["ausbildungsberufe"].astype(str).str.contains(r"\[[^\]]*\d[^\]]*\]", regex=True, na=False)
df_excluded = df_all.loc[~mask_has_bbig].copy()
df_all = df_all.loc[mask_has_bbig].copy()

keep_cols = ["schule","schuleLabel","koord","adresse","gründung","jedeschuleID",
             "ausbildungsberufe","teil_von","besteht_aus","website","QID"]
df_all_clean = df_all[keep_cols]
df_named_clean = df_named[["QID","benanntNach"]]
df = df_all_clean.merge(df_named_clean, on="QID", how="left")

lonlat = df["koord"].astype(str).str.extract(r"Point\(([-\d\.]+)\s+([-\d\.]+)\)").astype(float)
df[["lon","lat"]] = lonlat
df = df.dropna(subset=["lon","lat"])

gdf_schulen = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df["lon"], df["lat"]), crs="EPSG:4326")
if gdf_nuts1.crs != gdf_schulen.crs:
    gdf_nuts1 = gdf_nuts1.to_crs(gdf_schulen.crs)
gdf_joined = gpd.sjoin(gdf_schulen, gdf_nuts1[["NUTS_NAME","geometry"]], how="left", predicate="within")

# State column
df["Bundesland"] = gdf_joined["NUTS_NAME"].values
df.loc[df["QID"] == "Q134174080", "Bundesland"] = "Brandenburg"  # manual fix

# Save 
print(df.head(15))
df = df.drop(columns=["lon", "lat"], errors="ignore")
df.to_csv("Public_vocational_schools_Germany_de.csv", index=False, encoding="utf-8-sig")


# In[3]:


# Satellite Campuses
aussenstellen_ids = df["besteht_aus"].dropna().astype(str).str.split(r"[;,]\s*").explode().str.strip()
hauptstandorte = df[df["teil_von"].isna()]
anzahl_hauptstandorte = hauptstandorte["schule"].nunique()
anzahl_aussenstellen = aussenstellen_ids.nunique()

print("\n=== Vocational schools ===")
print(f"Number of vocational schools (only main campus): {anzahl_hauptstandorte}")
print(f"Number of satellite campuses: {anzahl_aussenstellen}")

# Table: Group schools by state
df_haupt = df[df["teil_von"].isna()].copy()
status_haupt = df_haupt.groupby("Bundesland")["schule"].nunique().to_frame(name="nur_hauptstandorte")
status_gesamt = df.groupby("Bundesland")["schule"].nunique().to_frame(name="inkl_aussenstellen")
status_counts = status_haupt.join(status_gesamt, how="outer").fillna(0).astype(int)
status_counts.loc["Gesamt"] = status_counts.sum()

print("\n=== Number of vocational schools by federal state ===")
print(status_counts)


# ### Quality control

# In[4]:


sample_schools = df.sample(n=150, random_state=70)[["schuleLabel", "QID"]]

print("\n=== 150 random schools ===")
print(sample_schools.to_string(index=False))


# ## English Dataset

# In[5]:


df_all = pd.read_csv("QUERY_all_schools_en.csv")
df_named = pd.read_csv("QUERY_named_after_en.csv")

print(df_all.columns.tolist())
print(df_named.columns.tolist())

def extract_qid(uri):
    if isinstance(uri, str) and "Q" in uri:
        return uri.split("/")[-1]
    return None

df_all["QID"] = df_all["QID"].apply(extract_qid)
df_named["QID"] = df_named["QID"].apply(extract_qid)

def only_dashes(berufe_str):
    if pd.isna(berufe_str):
        return True
    berufe = [b.strip() for b in str(berufe_str).split(";")]
    for b in berufe:
        match = re.search(r"\[(.*?)\]", b)
        if match and re.search(r"\d", match.group(1)):
            return False
    return True

df_excluded = df_all[df_all["training_occupations"].apply(only_dashes)].copy()
display(df_excluded)
df_all = df_all[~df_all["training_occupations"].apply(only_dashes)]


# In[6]:


# === Select relevant columns (from df_all & df_named) ===
df_all_clean = df_all[[
    "QID", "schoolLabel", "coordinates", "address", "foundation", "jedeschuleID",
    "training_occupations", "part_of", "consists_of", "website"
]].copy()

df_named_clean = df_named[[
    "QID", "namedAfter"
]].copy()

# === Mergen auf QID ===
df = df_all_clean.merge(df_named_clean, on="QID", how="left")

df.to_csv("Public_vocational_schools_Germany_en.csv", index=False, encoding="utf-8-sig")
display(df)


# In[7]:


# === Reload ===
df = pd.read_csv("Public_vocational_schools_Germany_en.csv", encoding="utf-8-sig")

df = df.dropna(subset=["coordinates"]).copy()
lonlat = (
    df["coordinates"].astype(str)
      .str.extract(r"Point\(\s*([-\d\.]+)[,\s]+([-\d\.]+)\s*\)")
      .astype(float)
)
df["lon"], df["lat"] = lonlat[0], lonlat[1]
df = df.dropna(subset=["lon", "lat"]).copy()

gdf_schools = gpd.GeoDataFrame(
    df,
    geometry=gpd.points_from_xy(df["lon"], df["lat"]),
    crs="EPSG:4326"
)

# === Load NUTS1 geometry ===
gdf_nuts1 = gpd.read_file("NUTS5000_N1.shp")

# === Align CRS ===
if gdf_nuts1.crs != gdf_schools.crs:
    gdf_nuts1 = gdf_nuts1.to_crs(gdf_schools.crs)

name_col = None
for cand in ["NUTS_NAME", "NAME_1"]:
    if cand in gdf_nuts1.columns:
        name_col = cand
        break
if name_col is None:
    raise KeyError("Keine Namensspalte im NUTS1-Shapefile gefunden.")

gdf_nuts1_subset = gdf_nuts1[[name_col, "geometry"]].copy()

# === Spatial join (school within NUTS1 region) -> only 'state' retained ===
gdf_joined = gpd.sjoin(gdf_schools, gdf_nuts1_subset, how="left", predicate="within")
gdf_joined = gdf_joined.rename(columns={name_col: "state"})
df = gdf_joined.drop(columns=["geometry", "index_right"], errors="ignore").copy()

# === One manual fix  ===
df.loc[df.get("QID", pd.Series(dtype=str)) == "Q134174080", "state"] = "Brandenburg"

# === German -> English state mapping ===
state_map = {
    "Baden-Württemberg": "Baden-Wuerttemberg",
    "Bayern": "Bavaria",
    "Berlin": "Berlin",
    "Brandenburg": "Brandenburg",
    "Bremen": "Bremen",
    "Hamburg": "Hamburg",
    "Hessen": "Hesse",
    "Mecklenburg-Vorpommern": "Mecklenburg-Western Pomerania",
    "Niedersachsen": "Lower Saxony",
    "Nordrhein-Westfalen": "North Rhine-Westphalia",
    "Rheinland-Pfalz": "Rhineland-Palatinate",
    "Saarland": "Saarland",
    "Sachsen": "Saxony",
    "Sachsen-Anhalt": "Saxony-Anhalt",
    "Schleswig-Holstein": "Schleswig-Holstein",
    "Thüringen": "Thuringia",
}

df["state"] = df["state"].map(state_map)

# === Save ===
df = df.drop(columns=["lon", "lat"], errors="ignore")
df.to_csv("Public_vocational_schools_Germany_en.csv", index=False, encoding="utf-8-sig")
display(df)

# === Extract branch IDs from 'consists_of' ===
branch_ids = (
    df["consists_of"].dropna().astype(str)
      .str.split(r"[;,]\s*").explode().str.strip()
)

# === Main sites = rows that are not 'part_of' anything ===
main_sites = df[df["part_of"].isna()].copy()

# === Branches overview ===
num_main_sites = main_sites["QID"].nunique()
num_branches = branch_ids.nunique()

print("\n=== Vocational schools (by main campuses) ===")
print(f"Count of schools (main campuses only): {num_main_sites}")
print(f"Count of recorded satellite campuses:     {num_branches}")


# === Aggregation by state ===
df_main = main_sites
counts_main = df_main.groupby("state")["QID"].nunique().to_frame(name="main_sites_only")
counts_total = df.groupby("state")["QID"].nunique().to_frame(name="incl_branches")

status_counts = counts_main.join(counts_total, how="outer").fillna(0).astype(int)
status_counts.loc["Total"] = status_counts.sum()

print("\n=== Number of vocational schools by state ===")
print(status_counts)


# # Interactive Map

# In[ ]:


pn.extension('folium')

df = pd.read_csv("Public_vocational_schools_Germany_de.csv")
START_POS  = (51.0, 10.0)
START_ZOOM = 6

df = df.dropna(subset=["koord"]).copy()
df[["lon", "lat"]] = (
    df["koord"]
      .str.extract(r"Point\(([-\d\.]+) ([-\d\.]+)\)")
      .astype(float)
)

def clean_list(raw_list):
    cleaned = []
    for raw in raw_list:
        if not raw or not raw.strip():
            continue
        if re.search(r"\[\s*[-–—]*\s*\]", raw):
            continue
        if not re.search(r"\[\s*\d+\s*\]", raw):
            continue
        item = re.sub(r"\[[^\]]*\]", "", raw)
        item = re.sub(r"^\s*[-–—]\s*|\s*[-–—]\s*$", "", item).strip()
        if item:
            cleaned.append(item)
    return cleaned

df["ausbildungsberufe"] = (
    df["ausbildungsberufe"]
      .fillna("")
      .astype(str)
      .str.split(";")
      .apply(clean_list)
)

# Dropdown menu
alle_berufe = sorted({beruf for lst in df["ausbildungsberufe"] for beruf in lst})

def generate_map(beruf: str | None = None):
    m = folium.Map(location=START_POS, zoom_start=START_ZOOM)
    data = df if not beruf else df[df["ausbildungsberufe"].apply(lambda lst: beruf in lst)]
    for _, row in data.iterrows():
        popup = (
            f"<b>{row['schuleLabel']}</b><br>"
            f"{', '.join(row['ausbildungsberufe'])}"
        )
        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=4,
            weight=0.5,     
            color='#3388ff',
            fill=True,
            fill_opacity=0.7,
            popup=popup,
            tooltip=row["schuleLabel"]
        ).add_to(m)
    return pn.pane.plot.Folium(m, width=800, height=600)

# Dashboard
dropdown = pn.widgets.Select(name="Beruf", options=[""] + alle_berufe)
dashboard = pn.Column(
    "## Berufsschulen nach Ausbildungsberuf (BBiG/HwO)",
    dropdown,
    pn.bind(generate_map, beruf=dropdown)
)

# Save
HTML_PATH = "public_schools_map.html"
dashboard.save(HTML_PATH, embed=True, resources="cdn")
print(f"✅  Interaktive Karte gespeichert unter:  {HTML_PATH}")


# ## Items created by user Weinessig

# In[ ]:


df_all = pd.read_csv("QUERY_all_schools_de.csv")
df_user = pd.read_excel("User_Contributions.xlsx")

# QID-Extractor (robust über Spalten)
qid_pat = re.compile(r"(Q\d+)")
def extract_qids(df, prefer=("QID","title","schule","item","item_id")):
    cols = [c for c in prefer if c in df.columns] or list(df.columns)
    for c in cols:
        s = df[c].astype(str).str.extract(qid_pat)[0]
        if s.notna().any():
            return s.dropna()
    # Fallback: suche über alle Spalten
    s = df.apply(lambda col: col.astype(str).str.extract(qid_pat)[0]).bfill(axis=1).iloc[:,0]
    return s.dropna()

qids_all  = extract_qids(df_all).drop_duplicates()
qids_user = extract_qids(df_user).drop_duplicates()

# Overlap
overlap = sorted(set(qids_all) & set(qids_user))

# Speichern + Report
print(f"QUERY_all_schools_de: {len(qids_all)}")
print(f"User_Contributions:   {len(qids_user)}")
print(f"Overlap:               {len(overlap)}")

