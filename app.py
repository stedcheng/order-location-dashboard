import streamlit as st
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import folium
from folium import Choropleth
import time
import re
import datetime
st.set_page_config(layout = "wide")

# CONSTANTS
LOCATION = [13.5, 122.5] # lat, long
ZOOM_START = 6

##### A. DATA PREPARATION
@st.cache_resource()
def prepare_data():
    ##### 1. IMPORTING
    gdf1_proj = gpd.read_file("ph_datasets/gdf1_simplified.gpkg")
    gdf2_proj = gpd.read_file("ph_datasets/gdf2_simplified.gpkg")
    
    # Local
    # gdf3_proj = gpd.read_file("ph_datasets/PH_Adm3_MuniCities.shp/PH_Adm3_MuniCities.shp.shp")
    # gdf4_proj = gpd.read_file("ph_datasets/PH_Adm4_BgySubMuns.shp/PH_Adm4_BgySubMuns.shp.shp")
    
    # Online
    gdf3_proj_url = r"https://github.com/altcoder/philippines-psgc-shapefiles/raw/refs/heads/main/dist/PH_Adm3_MuniCities.shp.zip"
    gdf3_proj = gpd.read_file(f"zip+{gdf3_proj_url}", layer = "PH_Adm3_MuniCities.shp")
    gdf4_proj_url = r"https://github.com/altcoder/philippines-psgc-shapefiles/raw/refs/heads/main/dist/PH_Adm4_BgySubMuns.shp.zip"
    gdf4_proj = gpd.read_file(f"zip+{gdf4_proj_url}", layer = "PH_Adm4_BgySubMuns.shp")
    
    gdf2_proj = gdf2_proj[~(gdf2_proj["adm2_psgc"] == 1909900000)] 
    ph_admin_div_names = pd.read_csv("output/ph_admin_div_names.csv")
    df_plot = pd.read_csv("output/df_plot.csv")

    ##### 2. DATATYPES AND FORMATTING
    ph_admin_div_names = ph_admin_div_names.astype(str)
    for col in df_plot.columns[df_plot.columns.str.contains("PSGC")]:
        df_plot[col] = df_plot[col].astype(str)
    
    for col in df_plot.columns[df_plot.columns.str.contains("_date")]:
        df_plot[col] = pd.to_datetime(df_plot[col])

    gdfs_proj = [gdf1_proj, gdf2_proj, gdf3_proj, gdf4_proj]
    for gdf_proj in gdfs_proj:
        
        # Convert all PSGC columns to strings
        for psgc_col in gdf_proj.columns[gdf_proj.columns.str.contains("psgc")]:
            gdf_proj[psgc_col] = gdf_proj[psgc_col].astype(str)

        # Capitalize all names
        for english_col in gdf_proj.columns[gdf_proj.columns.str.contains("_en")]:
            gdf_proj[english_col] = gdf_proj[english_col].str.upper()

    ##### 3. AREA DICTIONARIES
    # Region Number -> Area dictionary
    region_area_dict = {
        "NORTH LUZON" : [1, 2, 3, 14], 
        "SOUTH LUZON" : [4, 5, 17],
        "VISAYAS" : [6, 7, 8],
        "MINDANAO" : [9, 10, 11, 12, 16, 19],
        "METRO MANILA" : [13]
    }
    region_area_dict = {elt : key for key, value in region_area_dict.items() for elt in value}
    
    # Province -> Area dictionary from list of provinces (adm2_en) and region numbers (adm1_psgc)
    province_area_dict = gdf2_proj[["adm1_psgc", "adm2_en"]].copy()
    province_area_dict["adm1_psgc"] = (province_area_dict["adm1_psgc"].astype(int)) / 100000000
    province_area_dict["Area"] = province_area_dict["adm1_psgc"].map(region_area_dict)
    province_area_dict.drop(columns = ["adm1_psgc"], inplace = True)
    province_area_dict.rename(columns = {"adm2_en" : "Province"}, inplace = True)
    province_area_dict = dict(zip(province_area_dict["Province"], province_area_dict["Area"]))

    # Manually add for some provinces
    province_area_dict["METRO MANILA"] = "METRO MANILA" 
    province_area_dict["SPECIAL GEOGRAPHIC AREA"] = "MINDANAO"
    province_area_dict["MAGUINDANAO"] = "MINDANAO"
    province_area_dict["SAMAR"] = "VISAYAS"
    province_area_dict["COTABATO"] = "MINDANAO"
    
    ##### 4. PROCESSING
    provdist_to_provdist_dict = {"CITY OF ISABELA (NOT A PROVINCE)" : "BASILAN"}
    municity_to_provdist = pd.read_csv("ph_datasets/municity_to_provdist.csv")
    new_municity_to_provdist_dict = dict(zip(municity_to_provdist["new_municity"], municity_to_provdist["provdist"]))
    municity_to_municity = pd.read_csv("ph_datasets/municity_to_municity.csv")
    municity_to_municity_dict = dict(zip(municity_to_municity["old"], municity_to_municity["new"]))

    # Roman numeral mapping (expand if needed)
    roman_to_arabic = {
        "I": 1, "II": 2, "III": 3, "IV": 4, "V": 5,
        "VI": 6, "VII": 7, "VIII": 8, "IX": 9, "X": 10,
        "XI": 11, "XII": 12, "XIII": 13, "XIV": 14, "XV": 15,
        "XVI": 16, "XVII": 17, "XVIII": 18, "XIX": 19, "XX": 20,
        "XXI": 21, "XXII": 22, "XXIII": 23, "XXIV": 24, "XXV": 25,
        "XXVI": 26, "XXVII": 27, "XXVIII": 28, "XXIX": 29, "XXX": 30
    }

    def convert_roman_suffix(name):
        if not isinstance(name, str):
            return name
        name = name.strip()
        
        # Pattern 1: "Barangay IV" or "Poblacion XII"
        match1 = re.search(r"\b([IVXLCDM]+)$", name, re.IGNORECASE)
        if match1:
            roman = match1.group(1).upper()
            if roman in roman_to_arabic:
                return re.sub(r"\b" + roman + r"$", str(roman_to_arabic[roman]), name, flags =re.IGNORECASE)

        # Pattern 2: "Barangay IV A" or "Zone X B"
        match2 = re.search(r"\b([IVXLCDM]+) ([A-Z])$", name, re.IGNORECASE)
        if match2:
            roman = match2.group(1).upper()
            suffix = match2.group(2).upper()
            if roman in roman_to_arabic:
                return re.sub(r"\b" + roman + r"-" + suffix + r"$", f"{roman_to_arabic[roman]}-{suffix}", name, flags =re.IGNORECASE)

        return name

    def clean_admin_name(s):
        return (
            s.str.strip()
            .str.replace("Ñ", "N", regex = False)
            .str.replace("STA.", "SANTA", regex = False)
            .str.replace("STO.", "SANTO", regex = False)
            .str.replace(r"[-,]", " ", regex = True)
            .str.replace(r"[\'\.]", "", regex = True)
            .str.replace(r"\s+", " ", regex = True)
        )
    
    def replace_adm2_psgc(municity):
        provdist = new_municity_to_provdist_dict[municity]
        adm2_psgc = gdf2_proj[gdf2_proj["adm2_en"] == provdist]["adm2_psgc"].values[0]
        return adm2_psgc
    
    # GDF 1 (Area)
    # Make a new Area column from region_area_dict and dissolve geometry by Area
    gdf1_proj["Area"] = (gdf1_proj["adm1_psgc"].astype(int) / 100000000).map(region_area_dict)
    gdf1_proj = gdf1_proj.dissolve(by = "Area")
    gdf1_proj = gdf1_proj.reset_index()

    # GDF 2 (Province)
    # Rename all the NCRs to Metro Manila
    dissolve_mask = (gdf2_proj["adm2_en"].str.contains("NCR"))
    gdf2_proj_ncr = gdf2_proj[dissolve_mask].copy()
    gdf2_proj_non_ncr = gdf2_proj[~dissolve_mask].copy() 
    gdf2_proj_ncr_dissolved = gdf2_proj_ncr.dissolve()
    gdf2_proj = pd.concat([gdf2_proj_ncr_dissolved, gdf2_proj_non_ncr], ignore_index = True)
    gdf2_proj.loc[gdf2_proj["adm2_en"].str.contains("NCR"), "adm2_en"] = "METRO MANILA"
    gdf2_proj.loc[gdf2_proj["adm2_en"] == "METRO MANILA", "adm2_psgc"] = "1300000000"
    gdf2_proj["Area"] = (gdf2_proj["adm1_psgc"].astype(int) / 100000000).map(region_area_dict)

    # Clean the English name and apply the province / district map
    gdf2_proj["adm2_en"] = clean_admin_name(gdf2_proj["adm2_en"])
    gdf2_proj["adm2_en"] = gdf2_proj["adm2_en"].map(provdist_to_provdist_dict).fillna(gdf2_proj["adm2_en"])
    # gdf2_proj

    # GDF 3 (City)
    # Set PSGC code (province) of Metro Manila cities to 1300000000
    ncr_cities = list(set(gdf3_proj[gdf3_proj["adm1_psgc"] == "1300000000"]["adm3_en"]))
    gdf3_proj.loc[gdf3_proj["adm3_en"].isin(ncr_cities), "adm2_psgc"] = "1300000000"

    # Clean the English name and apply the municipality / city map
    gdf3_proj["adm3_en"] = clean_admin_name(gdf3_proj["adm3_en"])
    gdf3_proj["adm3_en"] = gdf3_proj["adm3_en"].map(municity_to_municity_dict).fillna(gdf3_proj["adm3_en"])

    # Fix Manila Districts
    manila_dists = ["TONDO", "TONDO I/II", "TONDO I / II", "TONDO I/2", "TONDO NORTH", "BINONDO", "QUIAPO", "SAN NICOLAS", 
                    "SANTA CRUZ", "SANTA CRUZ NORTH", "SANTA CRUZ SOUTH", "STA CRUZ NORTH", "STA CRUZ SOUTH", "SAMPALOC", "SANTA MESA",
                    "SAN MIGUEL", "ERMITA", "INTRAMUROS", "MALATE", "PACO", "PANDACAN", "PORT AREA", "SANTA ANA"]
    manila_rows = (gdf3_proj["adm3_psgc"].str.startswith("13806")) & (~gdf3_proj["adm3_en"].isin(manila_dists))
    gdf3_proj.loc[manila_rows, "adm3_en"] = "MANILA"
    # gdf3_proj[gdf3_proj["adm2_psgc"] == "1300000000"]

    # Fix highly urbanized cities
    adm2_psgc_list = gdf2_proj["adm2_psgc"].unique()
    highly_urbanized_cities_df = gdf3_proj[~gdf3_proj["adm2_psgc"].isin(adm2_psgc_list) & ~gdf3_proj["adm3_en"].str.contains("CLUSTER")]
    mask = gdf3_proj["adm3_en"].isin(highly_urbanized_cities_df["adm3_en"])
    gdf3_proj.loc[mask, "adm2_psgc"] = gdf3_proj.loc[mask, "adm3_en"].map(replace_adm2_psgc)

    # GDF 4 (Barangay)
    # Set PSGC code (city) of Manila barangays to 1380600000
    gdf4_proj.loc[gdf4_proj["adm4_psgc"].str[:5] == "13806", "adm3_psgc"] = "1380600000"
    # gdf4_proj[gdf4_proj["adm4_psgc"].str[:5] == "13806"]

    # Remove "POB.", "POB", "(POB.)", "(POB)", or "PROPER" at the end of BgySubmun
    gdf4_proj["adm4_en"] = gdf4_proj["adm4_en"].str.replace(r"\s*(?:\(POB\)\.?|\bPOB\.?\b|\bPROPER\b)\s*$", "", regex = True)

    # Replace all Roman suffixes (BARANGAY IV or BARANGAY IV-A) with Hindu-Arabic numerals (BARANGAY 4 or BARANGAY 4-A)
    gdf4_proj["adm4_en"] = gdf4_proj["adm4_en"].apply(convert_roman_suffix)    

    return ph_admin_div_names, gdf1_proj, gdf2_proj, gdf3_proj, gdf4_proj, df_plot

start = time.perf_counter()
ph_admin_div_names, gdf1_proj, gdf2_proj, gdf3_proj, gdf4_proj, df_plot = prepare_data()
end = time.perf_counter()
print(f"Elapsed time for Data Preparation: {end - start:.4f} seconds")

##### B. MAP FUNCTIONS
def make_map(gdf_geo, internal_area_name, display_area_name, internal_plot_var, display_plot_var, ascending_bool, location = LOCATION, zoom_start = ZOOM_START):
    """Creates a map in Folium given a geodataframe with geographic coordinates, the area's internal and display names, and the 
    the plot variable's internal and display name"""
    
    start = time.perf_counter()
    m = folium.Map(location = location, zoom_start = zoom_start)
    fill_color = "RdYlGn_r" if ascending_bool else "RdYlGn"
    Choropleth(
        geo_data = gdf_geo,
        data = gdf_geo,
        columns = [internal_area_name, internal_plot_var],
        key_on = "feature.properties." + internal_area_name,
        fill_color = fill_color,
        legend_name = display_plot_var
    ).add_to(m)
    end = time.perf_counter()
    print(f"Elapsed time for Map: {end - start:.4f} seconds")

    start = time.perf_counter()
    feature_group = folium.FeatureGroup(name = "Areas")
    folium.GeoJson(
        gdf_geo,
        style_function = lambda x: {
            "color": "black",
            "weight": 1
        },
        popup = folium.GeoJsonPopup(
            fields = [internal_area_name, internal_plot_var],
            aliases = [display_area_name, display_plot_var],
            localize = True,
            labels = True,
            style = "font-size: 12px;"
        )
    ).add_to(feature_group)
    m.add_child(feature_group)
    end = time.perf_counter()
    print(f"Elapsed time for Label: {end - start:.4f} seconds")
    
    return m

def calculate_geo_centroid(gdf_proj, col_name, col_value):
    """Calculates the geographical centroid given the geodataframe wih projected coordinates"""
    start = time.perf_counter()
    
    # Calculate centroid (projected) and x and y coordinates
    centroid_proj = gdf_proj[gdf_proj[col_name] == col_value]["geometry"].centroid
    centroid_proj_x = centroid_proj.x.mean()
    centroid_proj_y = centroid_proj.y.mean()

    # Create GeoSeries in projected coordinate system, then convert to geographic coordinate system
    centroid_gs_proj = gpd.GeoSeries([Point(centroid_proj_x, centroid_proj_y)], crs = gdf_proj.crs)
    centroid_gs_geo = centroid_gs_proj.to_crs(epsg = 4326)

    # Extract longitude and latitude
    centroid_lon = centroid_gs_geo.x[0]
    centroid_lat = centroid_gs_geo.y[0]
    centroid_geo = [centroid_lat, centroid_lon]

    end = time.perf_counter()
    print(f"Elapsed time for Calculating Geographic Centroid: {end - start:.4f} seconds")
    
    return centroid_geo

def make_map_country(gdf1_proj, internal_plot_var, display_plot_var, ascending_bool):
    """Plots a variable per area"""
    
    # Convert projected to geographic
    gdf1_geo = gdf1_proj.to_crs(epsg = 4326)

    m = make_map(gdf1_geo, "Area", "Area", internal_plot_var, display_plot_var, ascending_bool)
    return m

def make_map_area(area, gdf1_proj, gdf2_proj, internal_plot_var, display_plot_var, ascending_bool):
    """Zooms in to an area and plots a variable per province"""
    start = time.perf_counter()
    
    # Filter the provinces table to only include the area
    gdf2_proj_filtered = gdf2_proj[gdf2_proj["Area"] == area].copy()

    # Convert projected to geographic
    gdf2_geo_filtered = gdf2_proj_filtered.to_crs(epsg = 4326)

    # Create centroid
    centroid = calculate_geo_centroid(gdf1_proj, "Area", area)

    end = time.perf_counter()
    print(f"Elapsed time for Filtering: {end - start:.4f} seconds")

    m = make_map(gdf2_geo_filtered, "adm2_en", "Province", internal_plot_var, display_plot_var, ascending_bool, centroid, 9)
    return m

def make_map_province(provdist, gdf2_proj, gdf3_proj, internal_plot_var, display_plot_var, ascending_bool):
    """Zooms in to a province and plots a variable per municipality / city"""
    start = time.perf_counter()

    # Get the PSGC code for the province
    province_psgc = str(gdf2_proj[gdf2_proj["adm2_en"] == provdist]["adm2_psgc"].values[0])

    # Filter the municipalities / cities table to only include the province
    gdf3_proj_filtered = gdf3_proj[gdf3_proj["adm2_psgc"] == province_psgc].copy()

    # Convert projected to geographic 
    gdf3_geo_filtered = gdf3_proj_filtered.to_crs(epsg = 4326)

    # Create centroid
    centroid = calculate_geo_centroid(gdf2_proj, "adm2_en", provdist)
    
    end = time.perf_counter()
    print(f"Elapsed time for Filtering: {end - start:.4f} seconds")

    m = make_map(gdf3_geo_filtered, "adm3_en", "City", internal_plot_var, display_plot_var, ascending_bool, centroid, 11)
    return m

def make_map_city(provdist, municity, gdf2_proj, gdf3_proj, gdf4_proj, internal_plot_var, display_plot_var, ascending_bool):
    """Zooms in to a municipality / city and plots a variable per barangay / submunicipality"""
    start = time.perf_counter()

    # Get the PSGC code for the province and city
    province_psgc = str(gdf2_proj[gdf2_proj["adm2_en"] == provdist]["adm2_psgc"].values[0])
    city_psgc = str(gdf3_proj[(gdf3_proj["adm3_en"] == municity) & (gdf3_proj["adm2_psgc"] == province_psgc)]["adm3_psgc"].values[0])

    # Filter the barangays / submunicipalities table to only include the municipality / city
    gdf4_proj_filtered = gdf4_proj[gdf4_proj["adm3_psgc"] == city_psgc].copy()

    # Convert projected to geographic 
    gdf4_geo_filtered = gdf4_proj_filtered.to_crs(epsg = 4326)

    # Create centroid
    centroid = calculate_geo_centroid(gdf3_proj, "adm3_en", municity)
    
    end = time.perf_counter()
    print(f"Elapsed time for Filtering: {end - start:.4f} seconds")

    m = make_map(gdf4_geo_filtered, "adm4_en", "Barangay", internal_plot_var, display_plot_var, ascending_bool, centroid, 13)
    return m

def user_input(df_plot, gdf1_proj, gdf2_proj, gdf3_proj, gdf4_proj):
    st.title("Location Data")
    with st.sidebar:

        ##### 1. DATE & CATEGORY
        date_bool = True
        start_date = st.date_input("Start Date", value = (datetime.date.today() - pd.DateOffset(months = 1)).date(),
                                   min_value = pd.to_datetime("2025-07-01"))
        end_date = st.date_input("End Date", value = datetime.date.today(), max_value = datetime.date.today())
        if start_date > end_date:
            date_bool = False
        category_options = ["ED", "HL", "PE", "WL", "uncategorized"]
        categories = st.multiselect("Category", category_options, default = category_options)

        df_plot = df_plot[
            (df_plot["ordered_date"] >= pd.to_datetime(start_date)) & 
            (df_plot["ordered_date"] <= pd.to_datetime(end_date)) & 
            (df_plot["category_slug"].isin(categories))
        ]

        ##### 2. PLOT VARIABLE
        original_plot_vars = ["id", "ordered_to_processed", "processed_to_delivered_returned"]
        internal_plot_vars = ["order_count", "ordered_to_processed_mean", "processed_to_delivered_returned_mean"]
        display_plot_vars = ["Order Count", "Average Number of Hours from Ordered to Processed Status", 
                             "Average Number of Hours from Processed to Delievered Status"]
        short_display_plot_vars = ["Order Count", "Ordered to Processed (Avg Hrs)", "Processed to Delievered (Avg Hrs)"]
        agg_funcs = ["count", "mean", "mean"]
        ascending_bools = [False, True, True]

        short_display_plot_var = st.selectbox("Metric", short_display_plot_vars)
        idx = short_display_plot_vars.index(short_display_plot_var)
        original_plot_var = original_plot_vars[idx]
        internal_plot_var = internal_plot_vars[idx]
        display_plot_var = display_plot_vars[idx]
        ascending_bool = ascending_bools[idx]
        agg_func = agg_funcs[idx]
        agg_dict = {original_plot_var : agg_func}
        agg_col = [internal_plot_var]

        ##### 3. LOCATION
        input_checker = {"area" : 0, "provdist" : 0, "municity" : 0}

        area_list = ph_admin_div_names["Area"].unique()
        area = st.selectbox("Area", area_list, index = None, help = "Leave this blank to generate a dashboard for the Philippines.")
        provdist = None
        municity = None

        if area is not None:
            input_checker["area"] = 1
            provdist_list = ph_admin_div_names[ph_admin_div_names["Area"] == area]["ProvDist"].unique()
            if area == "METRO MANILA":
                provdist = st.selectbox("Province", provdist_list, index = 0)
            else: 
                provdist = st.selectbox("Province", provdist_list, index = None, help = f"Leave this blank to generate a dashboard for {area}.")

            if provdist is not None:
                input_checker["provdist"] = 1
                municity_list = ph_admin_div_names[ph_admin_div_names["ProvDist"] == provdist]["MuniCity"].unique()
                municity = st.selectbox("City", municity_list, index = None, help = f"Leave this blank to generate a dashboard for {provdist}.")

                if municity is not None:
                    input_checker["municity"] = 1

        # Session states
        if "generated" not in st.session_state: st.session_state.generated = False
        if "last_filters" not in st.session_state: st.session_state.last_filters = {}
        current_filters = {
            "start": start_date,
            "end": end_date,
            "category" : categories,
            "metric": short_display_plot_var,
            "area": area,
            "provdist": provdist,
            "municity": municity,
        }
        if current_filters != st.session_state.last_filters:
            st.session_state.generated = False
        if st.button("Generate"):
            st.session_state.generated = True
            st.session_state.last_filters = current_filters

    # Map text   
    if input_checker["municity"] == 1: 
        location_text = f"for {municity}, {provdist}"
    elif input_checker["provdist"] == 1: 
        location_text = f"for {provdist}"
    elif input_checker["area"] == 1:
        location_text = f"for {area}"
    else: 
        location_text = f"for the Philippines"
    map_text = f"Map of {display_plot_var} {location_text} ({start_date} - {end_date})" # insert categories

    def generate_heatmap(df):

        heatmap_data = df.groupby(["ordered_dow", "ordered_hour"]).size().unstack(fill_value = 0)
        heatmap_data.index += 1 # day of week: 0-6 -> 1-7
        heatmap_long = heatmap_data.reset_index().melt(id_vars = "ordered_dow", value_name = "count")
        
        # st.write(heatmap_data)
        # st.write(heatmap_long)
        # plt.figure(figsize = (6, 4))
        # sns.heatmap(heatmap_data, annot = True, cmap="YlOrBr", cbar=True)
        # plt.title("Order Heatmap by Hour and Day")
        # plt.xlabel("Hour of Day")
        # plt.ylabel("Day of Week")
        # st.pyplot(plt)
        
        # fig = px.density_heatmap(heatmap_long, x = "ordered_hour", y = "ordered_dow", z = "count", 
        #                          color_continuous_scale = "YlOrBr", nbinsx = 24, nbinsy = 7)

        blue_vibrant = [
            [0.0, "rgb(230, 245, 255)"],
            [0.3, "rgb(90, 170, 255)"],
            [0.6, "rgb(0, 90, 200)"],
            [1.0, "rgb(0, 0, 90)"]
        ]
        fig = go.Figure(
            data = go.Heatmap(
                x = heatmap_long["ordered_hour"],
                y = heatmap_long["ordered_dow"],
                z = heatmap_long["count"],
                colorscale = blue_vibrant,
                hovertemplate = "Hour: %{x}<br>Day: %{y}<br>Count: %{z}",
                xgap = 1,   # horizontal border
                ygap = 1,    # vertical border
                name = ""
            )
        )

        # fig.update_traces(hovertemplate = "Hour: %{x}<br>" + "Day: %{y}<br>" + "Count: %{z}")
        fig.update_layout(
            title = "Order Heatmap by Hour and Day",
            xaxis_title = "Hour of Day",
            yaxis_title = "Day of Week",
            xaxis = {
                "tickmode" : "array", "tickvals" : list(range(0, 24, 4)), "ticktext" : list(range(0, 24, 4))
            },
            yaxis = {
                "tickmode" : "array", "tickvals" : list(range(1, 8)), 
                "ticktext" : ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"], "autorange" : "reversed"
            }
        )
        st.plotly_chart(fig, use_container_width = True)

    # Display
    status_placeholder = st.empty()
    if date_bool and st.session_state.generated:
        status_placeholder.write("Generating...")
        if input_checker["municity"] == 1: 
            # Aggregate
            gdf4_agg = df_plot.groupby("BgySubmunPSGC").agg(agg_dict)
            gdf4_agg.columns = agg_col
            gdf4_proj = gdf4_proj.merge(gdf4_agg, left_on = "adm4_psgc", right_index = True, how = "left")
            
            # Make map
            m = make_map_city(provdist, municity, gdf2_proj, gdf3_proj, gdf4_proj, internal_plot_var, display_plot_var, ascending_bool)

            # Filter display table
            province_psgc = str(gdf2_proj[gdf2_proj["adm2_en"] == provdist]["adm2_psgc"].values[0])
            city_psgc = str(gdf3_proj[(gdf3_proj["adm3_en"] == municity) & (gdf3_proj["adm2_psgc"] == province_psgc)]["adm3_psgc"].values[0])
            display_table = gdf4_proj[gdf4_proj["adm3_psgc"] == city_psgc].copy()
            display_table = display_table[["adm4_en"] + agg_col]
            display_table.columns = ["Barangay", short_display_plot_var]
            display_table = display_table.sort_values(by = short_display_plot_var, ascending = ascending_bool)

            # Heatmap
            df_filtered = df_plot[(df_plot["MuniCity"] == municity) & (df_plot["ProvDist"] == provdist) & (df_plot["Area"] == area)]

        elif input_checker["provdist"] == 1: 
            # Aggregate
            gdf3_agg = df_plot.groupby("MuniCityPSGC").agg(agg_dict)
            gdf3_agg.columns = agg_col
            gdf3_proj = gdf3_proj.merge(gdf3_agg, left_on = "adm3_psgc", right_index = True, how = "left")
            
            # Make map
            m = make_map_province(provdist, gdf2_proj, gdf3_proj, internal_plot_var, display_plot_var, ascending_bool)

            # Filter display table
            province_psgc = str(gdf2_proj[gdf2_proj["adm2_en"] == provdist]["adm2_psgc"].values[0])
            display_table = gdf3_proj[gdf3_proj["adm2_psgc"] == province_psgc].copy()
            display_table = display_table[["adm3_en"] + agg_col]
            display_table.columns = ["City", short_display_plot_var]
            display_table = display_table.sort_values(by = short_display_plot_var, ascending = ascending_bool)

            # Heatmap
            df_filtered = df_plot[(df_plot["ProvDist"] == provdist) & (df_plot["Area"] == area)]

        elif input_checker["area"] == 1:
            # Aggregate
            gdf2_agg = df_plot.groupby("ProvDistPSGC").agg(agg_dict)
            gdf2_agg.columns = agg_col
            gdf2_proj = gdf2_proj.merge(gdf2_agg, left_on = "adm2_psgc", right_index = True, how = "left")
            
            # Make map
            m = make_map_area(area, gdf1_proj, gdf2_proj, internal_plot_var, display_plot_var, ascending_bool)

            # Filter display table
            display_table = gdf2_proj[gdf2_proj["Area"] == area].copy()
            display_table = display_table[["adm2_en"] + agg_col]
            display_table.columns = ["Province", short_display_plot_var]
            display_table = display_table.sort_values(by = short_display_plot_var, ascending = ascending_bool)

            # Heatmap
            df_filtered = df_plot[df_plot["Area"] == area]

        else: 
            # Aggregate
            gdf1_agg = df_plot.groupby("Area").agg(agg_dict)
            gdf1_agg.columns = agg_col
            gdf1_proj = gdf1_proj.merge(gdf1_agg, left_on = "Area", right_index = True, how = "left")

            # Make map
            m = make_map_country(gdf1_proj, internal_plot_var, display_plot_var, ascending_bool)

            # Filter display table
            display_table = gdf1_proj[["Area"] + agg_col]
            display_table.columns = ["Area", short_display_plot_var]
            display_table = display_table.sort_values(by = short_display_plot_var, ascending = ascending_bool)

            # Heatmap
            df_filtered = df_plot

        # After filtering for location, remove duplicate ID rows
        df_filtered_no_duplicates = df_filtered.drop_duplicates(keep = "first", subset = ["id"])
        counts = df_filtered["id"].value_counts()
        multiple = counts[counts >= 2].index
        df_filtered_duplicate_ids = df_filtered[df_filtered["id"].isin(multiple)]
        display_columns_old = ["id", "Area", "ProvDist", "MuniCity", "BgySubmun", "logistics_name", "ordered_date", "processed_date", "delivered_returned_date",
                                "ordered_to_processed", "processed_to_delivered_returned", "category_slug"]
        display_columns_new = ["ID", "Area", "Province", "City", "Barangay", "Logistics", "Ordered Date", "Processed Date", "Delivered/Returned Date",
                                "Ordered-to-Processed", "Processed-to-Delivered/Returned", "Category"]
        df_filtered_no_duplicates_display = df_filtered_no_duplicates[display_columns_old].rename(columns = dict(zip(display_columns_old, display_columns_new)))
        df_filtered_duplicate_ids_display = df_filtered_duplicate_ids[display_columns_old].rename(columns = dict(zip(display_columns_old, display_columns_new)))
        
        # Visuals
        st.header("Visuals")
        col1, col2 = st.columns([0.3, 0.7])
        with col1: st.subheader("Metrics Table")
        with col2: st.subheader(map_text)
        col1, col2 = st.columns([0.3, 0.7])
        with col1: 
            st.dataframe(display_table, hide_index = True, use_container_width = True)
            st.write(f"Number of Multiple-Item Orders: {len(multiple)}")
            st.write(f"Number of Rows Removed due to Multiple-Item Orders: {len(df_filtered) - len(df_filtered_no_duplicates)}")
        with col2: 
            st.components.v1.html(m._repr_html_(), height = 600)
        generate_heatmap(df_filtered_no_duplicates)

        # Tables
        st.header("Tables")
        columns = st.multiselect("Columns to Display", display_columns_new, default = display_columns_new)
        if len(columns) > 0:
            st.subheader("Unique Order IDs")
            st.dataframe(df_filtered_no_duplicates_display[columns], hide_index = True)
            st.subheader("Duplicate Order IDs")
            st.dataframe(df_filtered_duplicate_ids_display[columns], hide_index = True)
        else:
            st.write("No columns have been selected.")
        
        status_placeholder.empty()

    elif not date_bool:
        st.warning("Start date must be earlier than end date. Please check your input.")
    else:
        st.info("Please generate the dashboard using the settings in the sidebar.")
    
user_input(df_plot, gdf1_proj, gdf2_proj, gdf3_proj, gdf4_proj)