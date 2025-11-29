#!/usr/bin/env python
# coding: utf-8

# In[1]:


import re
import panel as pn
import folium
import pandas as pd
import geopandas as gpd
import sys

from SPARQLWrapper import SPARQLWrapper, JSON
from IPython.display import display


# In[2]:


endpoint_url = "https://query.wikidata.org/sparql"

# 1) SPARQL Query

query_main = """
SELECT
  ?QID ?Label_de ?Label_en ?coordinates ?address ?foundation ?jedeschuleID
  (GROUP_CONCAT(DISTINCT CONCAT(?ausbildungsberufeLabel, " [", COALESCE(?kldbLabel, "–"), "]"); separator="; ") AS ?occupations_de)
  (GROUP_CONCAT(DISTINCT CONCAT(?ausbildungsberufeLabel_en, " [", COALESCE(?kldbLabel, "–"), "]"); separator="; ") AS ?occupations_en)
  (GROUP_CONCAT(DISTINCT ?teil_vonLabel; separator="; ") AS ?part_of)
  (GROUP_CONCAT(DISTINCT ?bestehtAusLabel; separator="; ") AS ?consists_of)
  (GROUP_CONCAT(DISTINCT STR(?website); separator="; ") AS ?website)
  (GROUP_CONCAT(DISTINCT ?benanntNachLabel; separator="; ") AS ?namedAfter)
WHERE {
  VALUES ?typ_schule { wd:Q322563 wd:Q828973 wd:Q1650577 wd:Q829000 wd:Q23005337 }

  ?QID wdt:P31 ?typ_schule ;
       wdt:P17 wd:Q183 ;
       wdt:P101 ?occupations_de .

  OPTIONAL {
    ?QID p:P625 ?coordStmt .
    ?coordStmt ps:P625 ?coordinates .
    FILTER NOT EXISTS { ?coordStmt pq:P582 ?coordEnd. }
  }
  OPTIONAL {
    ?QID p:P6375 ?addrStmt .
    ?addrStmt ps:P6375 ?address .
    FILTER NOT EXISTS { ?addrStmt pq:P582 ?addrEnd. }
  }

  OPTIONAL { ?occupations_de wdt:P1021 ?kldb. }
  OPTIONAL { ?QID wdt:P361  ?part_of. }
  OPTIONAL { ?QID wdt:P527  ?consists_of. }
  OPTIONAL { ?QID wdt:P856  ?website. }
  OPTIONAL { ?QID wdt:P571  ?foundation. }
  OPTIONAL { ?QID wdt:P9224 ?jedeschuleID. }

  OPTIONAL {
    ?QID p:P138 ?benennungStatement .
    ?benennungStatement ps:P138 ?namedAfter .
    FILTER NOT EXISTS { ?benennungStatement pq:P582 ?enddatum. }

    ?namedAfter wdt:P31 ?typBenennung .
    FILTER (?typBenennung IN (wd:Q5, wd:Q8436, wd:Q13417114))
  }

  FILTER NOT EXISTS { ?QID wdt:P576  ?aufloesung. }
  FILTER NOT EXISTS { ?QID wdt:P1366 ?ersetztDurch. }
  FILTER NOT EXISTS { ?QID wdt:P3999 ?schließung. }
  FILTER NOT EXISTS { ?QID wdt:P31   wd:Q423208. }       # Privatschule
  FILTER NOT EXISTS { ?QID wdt:P31   wd:Q23002042. }     # Privatschule

  SERVICE wikibase:label {
    bd:serviceParam wikibase:language "de" .
    ?QID rdfs:label ?Label_de .
    ?occupations_de rdfs:label ?ausbildungsberufeLabel .
    ?kldb rdfs:label ?kldbLabel .
    ?part_of rdfs:label ?teil_vonLabel .
    ?consists_of rdfs:label ?bestehtAusLabel .
    ?namedAfter rdfs:label ?benanntNachLabel .
  }

  OPTIONAL {
    ?QID rdfs:label ?schuleLabel_en_raw .
    FILTER(LANG(?schuleLabel_en_raw) = "en")
  }
  BIND(?schuleLabel_en_raw AS ?Label_en)

  OPTIONAL {
    ?occupations_de rdfs:label ?ausbildungsberufeLabel_en_raw .
    FILTER(LANG(?ausbildungsberufeLabel_en_raw) = "en")
  }
  BIND(?ausbildungsberufeLabel_en_raw AS ?ausbildungsberufeLabel_en)

}
GROUP BY
  ?QID ?Label_de ?Label_en ?coordinates ?address ?foundation ?jedeschuleID
"""

# 2) English labels (part_of / consists_of)

query_en_parts = """
SELECT
  ?QID
  (GROUP_CONCAT(DISTINCT ?teil_vonLabel_en; separator="; ") AS ?part_of_en)
  (GROUP_CONCAT(DISTINCT ?bestehtAusLabel_en; separator="; ") AS ?consists_of_en)
WHERE {
  VALUES ?typ_schule { wd:Q322563 wd:Q828973 wd:Q1650577 wd:Q829000 wd:Q23005337 }

  ?QID wdt:P31 ?typ_schule ;
       wdt:P17 wd:Q183 .

  OPTIONAL {
    ?QID wdt:P361 ?part_of .
    ?part_of rdfs:label ?teil_vonLabel_en .
    FILTER(LANG(?teil_vonLabel_en) = "en")
  }

  OPTIONAL {
    ?QID wdt:P527 ?consists_of .
    ?consists_of rdfs:label ?bestehtAusLabel_en .
    FILTER(LANG(?bestehtAusLabel_en) = "en")
  }

  FILTER NOT EXISTS { ?QID wdt:P576  ?aufloesung. }
  FILTER NOT EXISTS { ?QID wdt:P1366 ?ersetztDurch. }
  FILTER NOT EXISTS { ?QID wdt:P3999 ?schließung. }
  FILTER NOT EXISTS { ?QID wdt:P31   wd:Q423208. }       # Privatschule
  FILTER NOT EXISTS { ?QID wdt:P31   wd:Q23002042. }     # Privatschule
}
GROUP BY ?QID
"""

# 3) SPARQL

def get_results(endpoint_url: str, query: str):
    user_agent = "WDQS-example Python/%s.%s" % (sys.version_info[0], sys.version_info[1])
    sparql = SPARQLWrapper(endpoint_url, agent=user_agent)
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    return sparql.query().convert()

def results_to_df(results_json) -> pd.DataFrame:
    rows = []
    vars_ = results_json["head"]["vars"]
    for b in results_json["results"]["bindings"]:
        row = {}
        for v in vars_:
            row[v] = b.get(v, {}).get("value")
        rows.append(row)
    return pd.DataFrame(rows)

results_main = get_results(endpoint_url, query_main)
df_main = results_to_df(results_main)
print("Main DF:", df_main.shape)

results_en = get_results(endpoint_url, query_en_parts)
df_en = results_to_df(results_en)
print("EN parts DF:", df_en.shape)

# 4) Merge and save

df_merged = df_main.merge(df_en, on="QID", how="left")
df_merged.to_csv("vocational_schools.csv", index=False)

print("Merged DF:", df_merged.shape)
print(df_merged.head())


# In[3]:


# Load files
df_all = pd.read_csv("vocational_schools.csv")
gdf_nuts1 = gpd.read_file("NUTS5000_N1.shp")


# Extract QIDs vectorized
qid_pat = r"(Q\d+)$"
df_all["QID"] = df_all["QID"].astype(str).str.extract(qid_pat)

# Exclude non-BBiG training occupations
mask_has_bbig = df_all["occupations_de"].astype(str).str.contains(
    r"\[[^\]]*\d[^\]]*\]", regex=True, na=False
)
df_excluded = df_all.loc[~mask_has_bbig].copy()
df_all = df_all.loc[mask_has_bbig].copy()

print("\n=== Excluded schools (non-BBiG training occupations) ===")
print(f"Number of excluded rows: {len(df_excluded)}")

keep_cols = [
    "QID", "Label_en", "Label_de", "coordinates", "address", "foundation", "jedeschuleID", "namedAfter", 
    "occupations_de", "occupations_en", "part_of", "part_of_en", "consists_of", "consists_of_en", "website",
]

df = df_all[keep_cols].copy()

lonlat = (
    df["coordinates"]
    .astype(str)
    .str.extract(r"Point\(([-\d\.]+)\s+([-\d\.]+)\)")
    .astype(float)
)
df[["lon", "lat"]] = lonlat
df = df.dropna(subset=["lon", "lat"])

gdf_schulen = gpd.GeoDataFrame(
    df,
    geometry=gpd.points_from_xy(df["lon"], df["lat"]),
    crs="EPSG:4326",
)
if gdf_nuts1.crs != gdf_schulen.crs:
    gdf_nuts1 = gdf_nuts1.to_crs(gdf_schulen.crs)

gdf_joined = gpd.sjoin(
    gdf_schulen,
    gdf_nuts1[["NUTS_NAME", "geometry"]],
    how="left",
    predicate="within",
)

# Add federal state
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

df["federal_state_de"] = gdf_joined["NUTS_NAME"].values
df.loc[df["QID"] == "Q134174080", "federal_state_de"] = "Brandenburg"  # manual fix
df["federal_state_en"] = df["federal_state_de"].map(state_map)
df["federal_state_en"] = df["federal_state_en"].fillna(df["federal_state_de"])

# Save
print(df.head(15))
df = df.drop(columns=["lon", "lat"], errors="ignore")
df.to_csv("Public_vocational_schools_Germany_de.csv", index=False, encoding="utf-8-sig")


# ## Delete schools in consists_of/part_of that are not part of the final dataset

# In[4]:


valid_labels = (
    set(df["Label_de"].dropna().astype(str))
    | set(df["Label_en"].dropna().astype(str))
)

def filter_label_list(value):
    if pd.isna(value):
        return pd.NA
    text = str(value).strip()
    if not text:
        return pd.NA
    
    # Annahme: Einträge sind per ';' getrennt
    parts = [p.strip() for p in text.split(";")]
    kept = [p for p in parts if p in valid_labels]
    
    if not kept:
        return pd.NA
    return "; ".join(kept)

# ===== 2–4. Cleaning + Beispiele, ohne *_orig im df =====

def clean_and_show_examples(col, n=10):
    if col not in df.columns:
        print(f"Spalte {col} nicht vorhanden, übersprungen.")
        return
    
    # Originalwerte separat sichern
    orig = df[col].copy()

    # Cleaning anwenden
    df[col] = df[col].apply(filter_label_list)

    # Änderungen finden (orig vs. neu)
    changed_mask = orig.astype(str) != df[col].astype(str)
    changed = df.loc[changed_mask, ["QID", "Label_de", "Label_en"]].copy()
    changed[f"{col}_before"] = orig[changed_mask].astype(str).values
    changed[f"{col}_after"] = df.loc[changed_mask, col].astype(str).values

    print(f"\n=== Beispiele für Änderungen in '{col}' (max. {n}) ===")
    print(changed.head(n))

for col in ["consists_of", "part_of", "consists_of_en", "part_of_en"]:
    clean_and_show_examples(col)


# ## Descriptive statistics

# In[5]:


# Satellite Campuses
aussenstellen_ids = (
    df["consists_of"].dropna().astype(str).str.split(r"[;,]\s*").explode().str.strip()
)
main_campus = df[df["part_of"].isna()]
number_main_campus = main_campus["QID"].nunique()
number_satellite_campus = df[df["part_of"].notna()]["QID"].nunique()

print("\n=== Vocational schools ===")
print(f"Number of vocational schools (only main campus): {number_main_campus}")
print(f"Number of satellite campuses: {number_satellite_campus}")

# Table: Group schools by state
df_main = df[df["part_of"].isna()].copy()
df_sat   = df[df["part_of"].notna()].copy()

status_main  = df_main.groupby("federal_state_de")["QID"].nunique().to_frame(name="main_campuses")
status_sat    = df_sat.groupby("federal_state_de")["QID"].nunique().to_frame(name="satellite_campuses")
status_total = df.groupby("federal_state_de")["QID"].nunique().to_frame(name="total")

status_counts = (
    status_main.join(status_sat, how="outer")
                .join(status_total, how="outer")
                .fillna(0).astype(int)
)
status_counts.loc["Total"] = status_counts.sum()

print("\n=== Number of vocational schools by federal state ===")
print(status_counts)


# ### Missing values

# In[6]:


print("\n=== Missing values per column ===")
missing_counts = df.isna().sum().to_frame("n_missing")
missing_counts["percent_missing"] = (
    missing_counts["n_missing"] / len(df) * 100
).round(2)
missing_counts = missing_counts.sort_values("n_missing", ascending=False)
print(missing_counts)

total_rows = len(df)
complete_rows = df.dropna().shape[0]

print("\n=== Completeness summary ===")
print(f"Total rows: {total_rows:,}")
print(f"Rows without any missing value: {complete_rows:,} ({complete_rows/total_rows:.1%})")
print(f"Rows with at least one missing value: {total_rows - complete_rows:,} ({1 - complete_rows/total_rows:.1%})")


# # Interactive Map

# In[ ]:


pn.extension('folium')

df = pd.read_csv("Public_vocational_schools_Germany_de.csv")
START_POS  = (51.0, 10.0)
START_ZOOM = 6

df = df.dropna(subset=["coordinates"]).copy()
df[["lon", "lat"]] = (
    df["coordinates"]
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

df["occupations_de"] = (
    df["occupations_de"]
      .fillna("")
      .astype(str)
      .str.split(";")
      .apply(clean_list)
)

# Dropdown menu
alle_berufe = sorted({beruf for lst in df["occupations_de"] for beruf in lst})

def generate_map(beruf: str | None = None):
    m = folium.Map(location=START_POS, zoom_start=START_ZOOM)
    data = df if not beruf else df[df["occupations_de"].apply(lambda lst: beruf in lst)]
    for _, row in data.iterrows():
        popup = (
            f"<b>{row['Label_de']}</b><br>"
            f"{', '.join(row['occupations_de'])}"
        )
        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=4,
            weight=0.5,     
            color='#3388ff',
            fill=True,
            fill_opacity=0.7,
            popup=popup,
            tooltip=row["Label_de"]
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


df_all = pd.read_csv("vocational_schools.csv")
df_user = pd.read_excel("User_Contributions.xlsx")

qid_pat = re.compile(r"(Q\d+)")
def extract_qids(df, prefer=("QID","title","schule","item","item_id")):
    cols = [c for c in prefer if c in df.columns] or list(df.columns)
    for c in cols:
        s = df[c].astype(str).str.extract(qid_pat)[0]
        if s.notna().any():
            return s.dropna()
    s = df.apply(lambda col: col.astype(str).str.extract(qid_pat)[0]).bfill(axis=1).iloc[:,0]
    return s.dropna()

qids_all  = extract_qids(df_all).drop_duplicates()
qids_user = extract_qids(df_user).drop_duplicates()

# Overlap
overlap = sorted(set(qids_all) & set(qids_user))

# Save + Report
print(f"QUERY_all_schools_de: {len(qids_all)}")
print(f"User_Contributions:   {len(qids_user)}")
print(f"Overlap:               {len(overlap)}")

