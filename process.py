import numpy as np
import pandas as pd
import re
from read import *

def process_data():

    ##### A. Loading and preprocessing the Sheets and Supabase data

    df_sh, datetime_sh = read_sheets()
    df_sb, datetime_sb = read_supabase()

    ########## A1. Initial preprocessing

    # Remove invalid rows
    df_sh = df_sh[(df_sh["Order Date"].notna()) & (df_sh["Order Date"] != "Order Date")]

    # Parse dates
    date_cols_sheets = ["Order Date", "Created Date", "Packing SLA", "Packed Date", "Shipped Date", "Delivered Date", "Returned Date"]
    for col in date_cols_sheets:
        df_sh[col] = pd.to_datetime(df_sh[col], format = "%m-%d-%Y %H:%M:%S")
    date_cols_sb = ["ordered_date", "processed_date", "delivered_time"]
    for col in date_cols_sb:
        df_sb[col] = pd.to_datetime(df_sb[col], format = "ISO8601").dt.tz_convert("Asia/Manila").dt.tz_localize(None)

    # Rename columns
    for col in ["barangay", "city_municipality", "province"]:
        df_sb[col] = df_sb[col].str.replace("-", " ").str.upper()
    df_sb.rename(columns = {"barangay" : "BgySubmun", "city_municipality" : "MuniCity", "province" : "ProvDist"}, inplace = True)

    ########## A2. Check for duplicates in Sheets

    # Frequency table of Partner OR Code (order ID)
    sh_value_counts = df_sh["Partner OR Code"].value_counts()

    # Check for duplicate order IDs
    sh_order_ids_multiple = sh_value_counts[sh_value_counts >= 2].index

    # Filter Sheets table for rows with duplicate order IDs
    df_sh_dupl = df_sh[df_sh["Partner OR Code"].isin(sh_order_ids_multiple)].sort_values("Partner OR Code")
    df_sh_dupl.drop(columns = df_sh_dupl.columns[0], inplace = True)

    # For each ID, check if all the entries are equal
    bools = []
    for id in sh_order_ids_multiple:
        df_temp = df_sh_dupl[df_sh_dupl["Partner OR Code"] == id]
        same = df_temp.apply(lambda row: row.equals(df_temp.iloc[0]), axis = 1).all()
        bools.append(same)

    # if min(bools) is True, then everything is true
    # if bools == []:
    #     print("Empty")
    # else:
    #     print(min(bools))

    df_sh.drop_duplicates(keep = "first", subset = df_sh.columns[1:], inplace = True)
    df_sh.reset_index(drop = True, inplace = True)

    ########## A3. Fix inconsistent IDs between Supabase and Sheets

    # Supabase has two SO-ROCK-29019 and SO-ROCK-29020
    # Sheets has SO-ROCK-29019, SO-ROCK-29019-1, SO-ROCK-29020, SO-ROCK-29020-1
    df_sb.loc[(df_sb["gr_order_id"] == "SO-ROCK-29019") & (df_sb["full_name"].str[0] == "R"), "gr_order_id"] = "SO-ROCK-29019-1"
    df_sb.loc[(df_sb["gr_order_id"] == "SO-ROCK-29020") & (df_sb["full_name"].str[0] == "M"), "gr_order_id"] = "SO-ROCK-29020-1"

    ########## A4. Parse addresses in Sheets, and assign logistics as J&T Express

    # Take the last three entries of Full Address column and assign that to BgySubmun, MuniCity, and ProvDist, respectively
    df_address = pd.DataFrame(df_sh["Full Address"]) 
    df_address["Full Address Reversed List"] = df_address["Full Address"].str.split(",") \
        .apply(lambda row : row[::-1]) \
        .apply(lambda row : [str(elt).strip() for elt in row]) \
        .apply(lambda row : [x for x in row if x != "need prescription"])
    df_expanded = pd.DataFrame(df_address["Full Address Reversed List"].tolist())
    df_sh["ProvDist"] = df_expanded[1].str.replace("-", " ")
    df_sh["MuniCity"] = df_expanded[2].str.replace("-", " ")
    df_sh["BgySubmun"] = df_expanded[3].str.replace("-", " ")

    # Manually assign J&T express to the logistics_name column
    df_sh.loc[:, "logistics_name"] = "J&T Express"

    ########## A5. Merge sheets and Supabase tables

    # Merge the two tables on the ID column
    df_merged = pd.merge(df_sb[["gr_order_id", "ordered_date", "processed_date", "ProvDist", "MuniCity", "BgySubmun", "logistics_name", "category_slug", "sku", "delivered_time"]], 
                        df_sh[["Partner OR Code", "Delivered Date", "Returned Date", "ProvDist", "MuniCity", "BgySubmun", "logistics_name"]], 
                        left_on = "gr_order_id", right_on = "Partner OR Code", how = "outer", indicator = True,
                        suffixes = ["_sb", "_sh"])

    # Rename columns
    df_merged.rename(columns = {"_merge" : "source", "Delivered Date" : "delivered_date", "Returned Date" : "returned_date",
                                "gr_order_id" : "sb_id", "Partner OR Code" : "sh_id"}, inplace = True)

    # Remove rows where both IDs are NAN
    # df_merged = df_merged[~(df_merged["supabase_id"].isna() & df_merged["sh_id"].isna())]

    # Convert source from category to string and replace contents
    df_merged["source"] = df_merged["source"].astype(str)
    df_merged["source"] = df_merged["source"].replace({"left_only" : "Supabase", "right_only" : "Sheets"})

    # Combine ID columns into one, do the same for created date columns (taking average if necessary), remove unnecessary columns
    df_merged["id"] = df_merged["sb_id"].combine_first(df_merged["sh_id"])
    df_merged.drop(columns = ["sb_id", "sh_id"], inplace = True)

    df_merged["delivered_date"] = pd.to_datetime(df_merged["delivered_date"].fillna(df_merged["delivered_time"]))
    df_merged.drop(columns = ["delivered_time"], inplace = True)

    ##### B. Loading and preprocessing Philippine PSGC dataa

    ########## B1. Keeping relevant columns

    ph_provdists = pd.read_csv("ph_datasets/PH_Adm2_ProvDists.csv")
    ph_municities = pd.read_csv("ph_datasets/PH_Adm3_MuniCities.csv")
    ph_bgysubmuns = pd.read_csv("ph_datasets/PH_Adm4_BgySubMuns.csv")

    ph_provdists = ph_provdists.astype(str)
    ph_provdists["adm2_en"] = ph_provdists["adm2_en"].str.upper().str.replace("-", " ")
    ph_provdists = ph_provdists[["adm1_psgc", "adm2_psgc", "adm2_en"]]
    # ph_provdists.info()

    ph_municities = ph_municities.astype(str)
    ph_municities["adm3_en"] = ph_municities["adm3_en"].str.upper()
    ph_municities = ph_municities[["adm1_psgc", "adm2_psgc", "adm3_psgc", "adm3_en"]]
    # ph_municities.info()

    ph_bgysubmuns["adm4_en"] = ph_bgysubmuns["adm4_en"].str.upper()
    ph_bgysubmuns = ph_bgysubmuns[["adm1_psgc", "adm2_psgc", "adm3_psgc", "adm4_psgc", "adm4_en"]]
    ph_bgysubmuns = ph_bgysubmuns.astype(str)
    # ph_bgysubmuns.info()

    ########## B2. Creating dictionaries for region->area and province->area

    # Region Number -> Area dictionary
    region_area_dict = {
        "NORTH LUZON" : [1, 2, 3, 14], 
        "SOUTH LUZON" : [4, 5, 17],
        "VISAYAS" : [6, 7, 8],
        "MINDANAO" : [9, 10, 11, 12, 16, 19],
        "METRO MANILA" : [13]
    }
    region_area_dict = {elt : key for key, value in region_area_dict.items() for elt in value}
    region_area_dict

    # create dictionary (Province -> Area) from list of provinces (adm2_en) and region numbers (adm1_psgc)
    province_area_dict = ph_provdists[["adm1_psgc", "adm2_en"]].copy()
    province_area_dict["adm1_psgc"] = (province_area_dict["adm1_psgc"].astype(int)) / 100000000
    province_area_dict["Area"] = province_area_dict["adm1_psgc"].map(region_area_dict)
    province_area_dict.drop(columns = ["adm1_psgc"], inplace = True)
    province_area_dict.rename(columns = {"adm2_en" : "Province"}, inplace = True)
    province_area_dict = dict(zip(province_area_dict["Province"], province_area_dict["Area"]))

    # manually add for some provinces
    province_area_dict["METRO MANILA"] = "METRO MANILA" 
    province_area_dict["SPECIAL GEOGRAPHIC AREA"] = "MINDANAO"
    province_area_dict["MAGUINDANAO"] = "MINDANAO"
    province_area_dict["SAMAR"] = "VISAYAS"
    province_area_dict["COTABATO"] = "MINDANAO"

    ########## B3. Merge datasets in province, city, and barangay level

    ph_admin_div = pd.merge(pd.merge(ph_provdists, ph_municities, how = "outer"), ph_bgysubmuns, how = "outer")
    ph_admin_div_names = ph_admin_div[["adm2_en", "adm3_en", "adm4_en", "adm2_psgc", "adm3_psgc", "adm4_psgc"]].copy()

    ########## B4. Replace names of various provinces and cities

    provdist_to_provdist_dict = {"CITY OF ISABELA (NOT A PROVINCE)" : "BASILAN"}

    municity_to_provdist = pd.read_csv("ph_datasets/municity_to_provdist.csv")
    old_municity_to_provdist_dict = dict(zip(municity_to_provdist["old_municity"], municity_to_provdist["provdist"]))
    new_municity_to_provdist_dict = dict(zip(municity_to_provdist["new_municity"], municity_to_provdist["provdist"]))

    municity_to_municity = pd.read_csv("ph_datasets/municity_to_municity.csv")
    municity_to_municity_dict = dict(zip(municity_to_municity["old"], municity_to_municity["new"]))

    bgysubmun_to_bgysubmun = pd.read_csv("ph_datasets/bgysubmun_to_bgysubmun.csv")
    bgysubmun_to_bgysubmun_dict = {(provdist, municity, old) : new for provdist, municity, old, new in 
                                zip(bgysubmun_to_bgysubmun["ProvDist"], 
                                    bgysubmun_to_bgysubmun["MuniCity"], 
                                    bgysubmun_to_bgysubmun["old"],
                                    bgysubmun_to_bgysubmun["new"])}

    ##### C. Uniformizing location names

    ########## C1. Helper functions

    srcs = ["_sb", "_sh"]
    adm_names = ["ProvDist", "MuniCity", "BgySubmun"]
    adm_psgcs = [adm_name + "PSGC" for adm_name in adm_names]
    adm_names_sb = [adm_name + "_sb" for adm_name in adm_names]
    adm_names_sh = [adm_name + "_sh" for adm_name in adm_names]

    def check(provdist, municity, bgysubmun):
        return ph_admin_div_names[
            (ph_admin_div_names["ProvDist"].str.contains(provdist)) &
            (ph_admin_div_names["MuniCity"].str.contains(municity)) &
            (ph_admin_div_names["BgySubmun"].str.contains(bgysubmun))
        ]

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

    def replace_bgy(row, src):
        provdist = row["ProvDist" + src]
        municity = row["MuniCity" + src]
        bgysubmun = row["BgySubmun" + src]

        # Only replace if ProvDist + MuniCity + BgySubmun matches mapping
        key = (provdist, municity, bgysubmun)
        if key in bgysubmun_to_bgysubmun_dict:
            return bgysubmun_to_bgysubmun_dict[key]
        return bgysubmun

    ########## C2. Editing ph_admin_div_names

    # Rename columns
    ph_admin_div_names.rename(columns = {"adm2_en" : "ProvDist", "adm3_en" : "MuniCity", "adm4_en" : "BgySubmun",
                                        "adm2_psgc" : "ProvDistPSGC", "adm3_psgc" : "MuniCityPSGC", "adm4_psgc" : "BgySubmunPSGC",}, inplace = True)

    # Remove NCR districts that are considered ProvDists, and fill missing values 
    drop_prov_dists = [
        "NCR, CITY OF MANILA, FIRST DISTRICT (NOT A PROVINCE)",
        "NCR, SECOND DISTRICT (NOT A PROVINCE)",
        "NCR, THIRD DISTRICT (NOT A PROVINCE)",
        "NCR, FOURTH DISTRICT (NOT A PROVINCE)"
    ]
    for adm_psgc in adm_psgcs:
        ph_admin_div_names.replace({adm_psgc : {"0" : np.nan}}, inplace = True)

    ph_admin_div_names = ph_admin_div_names[~ph_admin_div_names["ProvDist"].isin(drop_prov_dists)]
    ph_admin_div_names = ph_admin_div_names[ph_admin_div_names["BgySubmun"].notna()]
    ph_admin_div_names.fillna("", inplace = True)

    # # Remove ñ's, dashes, periods, commas, apostrophes, extra spaces, and replace STA. and STO. with full spellings 
    def clean_admin_name(s):
        return (
            s.str.strip()
            .str.replace("Ñ", "N", regex=False)
            .str.replace("STA.", "SANTA", regex=False)
            .str.replace("STO.", "SANTO", regex=False)
            .str.replace(r"[-,]", " ", regex=True)
            .str.replace(r"[\'\.]", "", regex=True)
            .str.replace(r"\s+", " ", regex=True)
        )

    for adm_name in adm_names:
        ph_admin_div_names[adm_name] = clean_admin_name(ph_admin_div_names[adm_name])

    # Standardize ProvDists
    ph_admin_div_names["ProvDist"] = ph_admin_div_names["ProvDist"].map(provdist_to_provdist_dict).fillna(ph_admin_div_names["ProvDist"])

    # Assign the municities in the PSGC dataset that do not have a ProvDist to an appropriate one
    # Note: Mostly highly urbanized cities
    ph_admin_div_names["ProvDist"] = ph_admin_div_names["MuniCity"].map(old_municity_to_provdist_dict).fillna(ph_admin_div_names["ProvDist"])

    # Standardize MuniCities
    ph_admin_div_names["MuniCity"] = ph_admin_div_names["MuniCity"].map(municity_to_municity_dict).fillna(ph_admin_div_names["MuniCity"])

    # Fix Manila Districts
    manila_dists = ["TONDO", "TONDO I/II", "TONDO I / II", "TONDO I/2", "TONDO NORTH", "BINONDO", "QUIAPO", "SAN NICOLAS", 
                    "SANTA CRUZ", "SANTA CRUZ NORTH", "SANTA CRUZ SOUTH", "STA CRUZ NORTH", "STA CRUZ SOUTH", "SAMPALOC", "SANTA MESA",
                    "SAN MIGUEL", "ERMITA", "INTRAMUROS", "MALATE", "PACO", "PANDACAN", "PORT AREA", "SANTA ANA"]
    manila_rows = (ph_admin_div_names["BgySubmunPSGC"].str.startswith("13806")) & (~ph_admin_div_names["BgySubmun"].isin(manila_dists))
    ph_admin_div_names.loc[manila_rows, "ProvDist"] = "METRO MANILA"
    ph_admin_div_names.loc[manila_rows, "MuniCity"] = "MANILA"

    # Add AREA (N Luzon, S Luzon, Visayas, Mindanao, Metro Manila)
    ph_admin_div_names["Area"] = ph_admin_div_names["ProvDist"].map(province_area_dict)

    # Remove "POB.", "POB", "(POB.)", "(POB)", or "PROPER" at the end of BgySubmun
    ph_admin_div_names["BgySubmun"] = ph_admin_div_names["BgySubmun"].str.replace(r"\s*(?:\(POB\)\.?|\bPOB\.?\b|\bPROPER\b)\s*$", "", regex = True)

    # Replace all Roman suffixes (BARANGAY IV or BARANGAY IV-A) with Hindu-Arabic numerals (BARANGAY 4 or BARANGAY 4-A)
    ph_admin_div_names["BgySubmun"] = ph_admin_div_names["BgySubmun"].apply(convert_roman_suffix)

    # Replace provdist PSGC of the municities
    def replace_provdist_psgc(municity):
        provdist = new_municity_to_provdist_dict.get(municity)
        if provdist is None:
            return None
        return provdist_to_psgc.get(provdist)
    # Get unique provdist PSGCs
    provdist_psgc_list = ph_provdists["adm2_psgc"].unique()
    # HUCs = have different provdist PSGCs
    highly_urbanized_cities_df = ph_admin_div_names[~ph_admin_div_names["ProvDistPSGC"].isin(provdist_psgc_list) & ~ph_admin_div_names["MuniCity"].str.contains("CLUSTER")]
    # Top 1 provdist PSGC per province
    provdist_to_psgc = ph_admin_div_names.groupby("ProvDist")["ProvDistPSGC"].agg(lambda x: x.value_counts().index[0]).to_dict()
    # Replace provdists of HUCs (except Metro Manila) to top 1 provdist PSGC for that province
    mask = (ph_admin_div_names["MuniCity"].isin(highly_urbanized_cities_df["MuniCity"])) & (ph_admin_div_names["ProvDist"] != "METRO MANILA")
    ph_admin_div_names.loc[mask, "ProvDistPSGC"] = ph_admin_div_names.loc[mask, "MuniCity"].map(replace_provdist_psgc)
    # prov_list = highly_urbanized_cities_df["ProvDist"].unique()
    # for i in range(len(prov_list)):
    #     print(prov_list[i], ph_admin_div_names[ph_admin_div_names["ProvDist"] == prov_list[i]]["ProvDistPSGC"].value_counts())

    # Remove empty if any
    ph_admin_div_names = ph_admin_div_names[ph_admin_div_names["ProvDist"] != ""]

    ########## C3. Editing df_merged

    # Replace (PROVDIST, PROVDIST-MUNICITY) to (PROVDIST, MUNICITY) 
    # except if MUNICITY has the string CITY (to avoid transforming CEBU, CEBU CITY -> CEBU, CITY for example)
    # or if PROVDIST and MUNICITY are the same (e.g. SIQUIJOR, SIQUIJOR)
    def clean_municity(row, src):
        provdist = row["ProvDist" + src]
        municity = row["MuniCity" + src]

        if "CITY" in str(municity).upper() or provdist == municity:
            return municity

        # Remove province name from start, if present
        if str(municity).startswith(provdist):
            municity = municity.replace(provdist, "", 1).strip()
        
        return municity

    provdist_to_provdist_dict = {
        # sheets/supabase name : GitHub name
        "WESTERN SAMAR" : "SAMAR",
        "NORTH COTABATO" : "COTABATO"
    }

    for src in srcs:

        # Fill missing values
        for adm_name in adm_names:
            df_merged.fillna({adm_name + src : ""}, inplace = True)

        # Replace NCR with Metro Manila
        df_merged["ProvDist" + src] = df_merged["ProvDist" + src].replace({"NCR" : "METRO MANILA"})

        # Standardize ProvDist, MuniCity and BgySubmun
        df_merged["MuniCity" + src] = df_merged.apply(clean_municity, axis = 1, args = (src, ))
        df_merged["ProvDist" + src] = df_merged["ProvDist" + src].map(provdist_to_provdist_dict).fillna(df_merged["ProvDist" + src])
        df_merged["MuniCity" + src] = df_merged["MuniCity" + src].map(municity_to_municity_dict).fillna(df_merged["MuniCity" + src])
        df_merged["BgySubmun" + src] = df_merged.apply(replace_bgy, axis = 1, args = (src, ))

        # Add AREA (N Luzon, S Luzon, Visayas, Mindanao, Metro Manila)
        df_merged["Area" + src] = df_merged["ProvDist" + src].map(province_area_dict)

        # Replace Metro Manila districts (METRO MANILA, QUIAPO, BRGY XXX -> METRO MANILA, MANILA, BRGY XXX)
        df_merged.loc[(df_merged["ProvDist" + src] == "METRO MANILA") & (df_merged["MuniCity" + src].isin(manila_dists)), "MuniCity" + src] = "MANILA"

        # Move the EMBO barangays from Makati to Taguig
        df_merged.loc[(df_merged["ProvDist" + src] == "METRO MANILA") & df_merged["BgySubmun" + src].str.contains("EMBO"), "MuniCity" + src] = "TAGUIG"

        # Remove the last set of parentheses (and anything inside them)
        bgysubmuns_with_parentheses = ph_admin_div_names[ph_admin_div_names["BgySubmun"].str.contains(r"\(", regex = True)]["BgySubmun"].values.tolist()
        df_merged["BgySubmun" + src] = df_merged["BgySubmun" + src].apply(
            lambda x: re.sub(r"\s*(?:\([^()]*\))\s*$", "", x) \
                if x not in bgysubmuns_with_parentheses else x
        )

        # Replace all roman suffixes (BARANGAY IV or BARANGAY IV-A) with Hindu-Arabic numerals (BARANGAY 4 or BARANGAY 4-A)
        df_merged["BgySubmun" + src] = df_merged["BgySubmun" + src].apply(convert_roman_suffix)

        # Remove Ñ's, multiple spaces, and ~'s
        for adm_name in adm_names:
            df_merged[adm_name + src] = df_merged[adm_name + src].str.replace("Ñ", "N", regex = False).str.replace(r"\s+", " ", regex = True)
            df_merged[adm_name + src] = df_merged[adm_name + src].str.replace(r"[-,~]", " ", regex = True).str.replace(r"['\.]", "", regex = True).str.upper()

        # Remove "POB.", "POB", "(POB.)", "(POB)", or "PROPER" at the end of BgySubmun
        df_merged["BgySubmun" + src] = df_merged["BgySubmun" + src].str.replace(r"\s*(?:\(POB\)\.?|\bPOB\.?\b|\bPROPER\b)\s*$", "", regex = True)
        
        # # NEGROS OCCIDENTAL, NEGROS OCCIDENTAL SAN CARLOS -> NEGROS OCCIDENTAL, SAN CARLOS CITY
        # df_merged.loc[(df_merged["ProvDist" + src] == "NEGROS OCCIDENTAL") & (df_merged["MuniCity" + src] == "SAN CARLOS"), "MuniCity" + src] = "SAN CARLOS CITY"

        # # PANGASINAN, PANGASINAN SAN CARLOS CITY -> PANGASINAN, SAN CARLOS CITY
        # df_merged.loc[(df_merged["ProvDist" + src] == "PANGASINAN") & (df_merged["MuniCity" + src] == "SAN CARLOS"), "MuniCity" + src] = "SAN CARLOS CITY"

    ########## C4. Applying address logic

    # Create sets for fast membership checks
    provdist_municity_bgysubmun_set = set(tuple(row) for row in ph_admin_div_names[["ProvDist", "MuniCity", "BgySubmun"]].dropna().values)
    municity_bgysubmun_set = set(tuple(row) for row in ph_admin_div_names[["MuniCity", "BgySubmun"]].dropna().values)
    bgysubmun_set = set(ph_admin_div_names["BgySubmun"].dropna().unique())

    def address_logic(row, src):
        provdist_municity_bgysubmun_key = (row["ProvDist" + src], row["MuniCity" + src], row["BgySubmun" + src])
        municity_bgysubmun_key = (row["MuniCity" + src], row["BgySubmun" + src])
        bgysubmun_key = row["BgySubmun" + src]

        empty_row = (row["ProvDist" + src] == "" and row["MuniCity" + src] == "" and row["BgySubmun" + src] == "")

        if empty_row:
            return -1
        elif (provdist_municity_bgysubmun_key in provdist_municity_bgysubmun_set):
            return 3
        elif municity_bgysubmun_key in municity_bgysubmun_set:
            return 2
        elif bgysubmun_key in bgysubmun_set:
            return 1
        else:
            return 0

    # print(address_logic(df_merged.iloc[0], "_sb"))
    # print(address_logic(df_merged.iloc[0], "_sh"))

    df_merged["address_logic_sb"] = df_merged.apply(address_logic, axis = 1, args = ("_sb", ))
    df_merged["address_logic_sh"] = df_merged.apply(address_logic, axis = 1, args = ("_sh", ))

    address_logic_counts_sb = df_merged["address_logic_sb"].value_counts().sort_index()
    address_logic_counts_sh = df_merged["address_logic_sh"].value_counts().sort_index()

    # print(address_logic_counts_sb)
    # print(address_logic_counts_sh)

    percent_ok_sb = address_logic_counts_sb.loc[3] / address_logic_counts_sb.loc[0:].sum()
    percent_ok_sh = address_logic_counts_sh.loc[3] / address_logic_counts_sh.loc[0:].sum()

    # print(f"Supabase Address Logic: {percent_ok_sb*100:.2f}%")
    # print(f"Sheets Address Logic  : {percent_ok_sh*100:.2f}%")

    ########## C5. Reconcile source differences

    # If one source has a better address logic than the other, copy its values for the provdist, municity, and bgysubmun
    for adm_name in adm_names:
        df_merged[adm_name] = np.where(
            df_merged["address_logic_sb"] >= df_merged["address_logic_sh"],
            df_merged[adm_name + "_sb"],
            df_merged[adm_name + "_sh"]
        )
    df_merged["address_logic_max"] = df_merged[["address_logic_sb", "address_logic_sh"]].max(axis = 1)
    df_merged["Area"] = np.where(df_merged["address_logic_sb"] >= df_merged["address_logic_sh"], df_merged["Area_sb"], df_merged["Area_sh"])

    # If both logistics are the same, return either; if both are different, return "SOURCE CONFLICT", if one is missing, return the other; and if both are missing, return np.nan
    conditions = [
        df_merged["logistics_name_sb"].notna() & df_merged["logistics_name_sh"].notna() & (df_merged["logistics_name_sb"] == df_merged["logistics_name_sh"]),
        df_merged["logistics_name_sb"].notna() & df_merged["logistics_name_sh"].notna() & (df_merged["logistics_name_sb"] != df_merged["logistics_name_sh"]),
        df_merged["logistics_name_sb"].notna() & df_merged["logistics_name_sh"].isna(),
        df_merged["logistics_name_sh"].notna() & df_merged["logistics_name_sb"].isna(),
        df_merged["logistics_name_sb"].isna() & df_merged["logistics_name_sh"].isna()
    ]
    choices = [df_merged["logistics_name_sb"], "SOURCE CONFLICT", df_merged["logistics_name_sb"], df_merged["logistics_name_sh"], np.nan]
    df_merged["logistics_name"] = np.select(conditions, choices, default = np.nan)

    # Metrics
    df_merged["delivered_returned_date"] = np.where(df_merged["delivered_date"].isna(), df_merged["returned_date"], df_merged["delivered_date"])
    df_merged["ordered_to_processed"] = (df_merged["processed_date"] - df_merged["ordered_date"]).dt.total_seconds() / 3600
    df_merged["processed_to_delivered_returned"] = (df_merged["delivered_returned_date"] - df_merged["processed_date"]).dt.total_seconds() / 3600
    df_merged["ordered_to_delivered_returned"] = df_merged["ordered_to_processed"] + df_merged["processed_to_delivered_returned"]
    conditions = [
        df_merged["Area"] == "METRO MANILA",
        ((df_merged["Area"] == "SOUTH LUZON") | (df_merged["Area"] == "NORTH LUZON")) & (df_merged["ProvDist"] != "PALAWAN"),
        (df_merged["ProvDist"] == "PALAWAN") | (df_merged["Area"] == "VISAYAS"),
        df_merged["Area"] == "MINDANAO"
    ]
    choices = [48, 72, 120, 168]
    df_merged["ordered_to_delivered_returned_standard"] = np.select(conditions, choices, default = np.nan)
    df_merged["fast_order"] = np.where(df_merged["ordered_to_delivered_returned"] < df_merged["ordered_to_delivered_returned_standard"], 1, 0)
    df_merged["ordered_hour"] = df_merged["ordered_date"].dt.hour
    df_merged["ordered_dow"] = df_merged["ordered_date"].dt.dayofweek
    df_merged["processed_dow"] = df_merged["processed_date"].dt.dayofweek
    df_merged["delivered_returned_dow"] = df_merged["delivered_returned_date"].dt.dayofweek
    # df_merged["ordered_month"] = df_merged["ordered_date"].dt.month
    df_merged["ordered_month"] = df_merged["ordered_date"].dt.strftime("%Y-%m")

    # Rearrange columns for convenience
    sb_cols = ["Area_sb"] + adm_names_sb + ["logistics_name_sb"]
    sh_cols = ["Area_sh"] + adm_names_sh + ["logistics_name_sh"]
    consolidated_cols = ["Area"] + adm_names + ["logistics_name"]
    address_logics = [col for col in df_merged.columns if "address" in col]
    date_cols = [col for col in df_merged.columns if "date" in col]
    dow_cols = [col for col in df_merged.columns if "dow" in col or "month" in col]
    metric_cols = ["ordered_to_processed", "processed_to_delivered_returned", "ordered_to_delivered_returned", "fast_order", "ordered_hour"]
    misc_cols = ["category_slug", "sku", "source"]
    df_merged_columns = ["id"] + sb_cols + sh_cols + consolidated_cols + address_logics + date_cols + dow_cols + metric_cols + misc_cols
    df_merged = df_merged[df_merged_columns]

    ##### D. Preparation for plotting

    # Masks
    # At least one of the addresses has to completely match
    address_mask = (df_merged["address_logic_max"] == 3)
    id_mask = (df_merged["id"].notna())
    # Ordered, processed, and one of delivered or returned, should exist
    # date_mask = (df_merged["ordered_date"].notna()) & (df_merged["processed_date"].notna()) & ((df_merged["delivered_date"].notna()) | (df_merged["returned_date"].notna()))

    # Dataframe for future plotting
    # df_plot = df_merged[address_mask & date_mask][["id"] + consolidated_cols + date_cols + dow_cols + metric_cols + ["category_slug"]]
    df_plot = df_merged[address_mask & id_mask][["id"] + consolidated_cols + date_cols + dow_cols + metric_cols + ["category_slug", "sku"]]
    df_plot = pd.merge(df_plot, ph_admin_div_names, on = adm_names, how = "left")
    df_plot.drop(columns = ["Area_y"], inplace = True)
    df_plot.rename(columns = {"Area_x" : "Area"}, inplace = True)

    # Rename some PSGC IDs
    df_plot.loc[df_plot["ProvDist"] == "METRO MANILA", "ProvDistPSGC"] = "1300000000" 
    df_plot.loc[df_plot["MuniCity"] == "MANILA", "MuniCityPSGC"] = "1380600000"

    return ph_admin_div_names, df_plot, df_merged, datetime_sh, datetime_sb


