from googleapiclient.discovery import build
from google.oauth2 import service_account
import pandas as pd
import streamlit as st
import os
from supabase import create_client, Client
from dotenv import load_dotenv
import datetime
from zoneinfo import ZoneInfo

SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]
WARMUP_DONE = False

def get_sheets_credentials():
    try:
        creds_dict = {
            "type": st.secrets["ULTRA_GOOGLE_TYPE"],
            "project_id": st.secrets["ULTRA_GOOGLE_PROJECT_ID"],
            "private_key_id": st.secrets["ULTRA_GOOGLE_PRIVATE_KEY_ID"],
            "private_key": st.secrets["ULTRA_GOOGLE_PRIVATE_KEY"],
            "client_email": st.secrets["ULTRA_GOOGLE_CLIENT_EMAIL"],
            "client_id": st.secrets["ULTRA_GOOGLE_CLIENT_ID"],
            "auth_uri": st.secrets["ULTRA_GOOGLE_AUTH_URI"],
            "token_uri": st.secrets["ULTRA_GOOGLE_TOKEN_URI"],
            "auth_provider_x509_cert_url": st.secrets["ULTRA_GOOGLE_AUTH_PROVIDER_CERT_URL"],
            "client_x509_cert_url": st.secrets["ULTRA_GOOGLE_CLIENT_CERT_URL"],
            "universe_domain": st.secrets["ULTRA_GOOGLE_UNIVERSE_DOMAIN"],
        }
        spreadsheet_id = st.secrets["SPREADSHEET_ID"]
        sheet_name = st.secrets["SHEET_NAME"]
        return creds_dict, spreadsheet_id, sheet_name
    except Exception:
        load_dotenv()
        creds_dict = {
            "type": os.getenv("ULTRA_GOOGLE_TYPE"),
            "project_id": os.getenv("ULTRA_GOOGLE_PROJECT_ID"),
            "private_key_id": os.getenv("ULTRA_GOOGLE_PRIVATE_KEY_ID"),
            "private_key": os.getenv("ULTRA_GOOGLE_PRIVATE_KEY"),
            "client_email": os.getenv("ULTRA_GOOGLE_CLIENT_EMAIL"),
            "client_id": os.getenv("ULTRA_GOOGLE_CLIENT_ID"),
            "auth_uri": os.getenv("ULTRA_GOOGLE_AUTH_URI"),
            "token_uri": os.getenv("ULTRA_GOOGLE_TOKEN_URI"),
            "auth_provider_x509_cert_url": os.getenv("ULTRA_GOOGLE_AUTH_PROVIDER_CERT_URL"),
            "client_x509_cert_url": os.getenv("ULTRA_GOOGLE_CLIENT_CERT_URL"),
            "universe_domain": os.getenv("ULTRA_GOOGLE_UNIVERSE_DOMAIN"),
        }
        spreadsheet_id = os.getenv("SPREADSHEET_ID")
        sheet_name = os.getenv("SHEET_NAME")
        return creds_dict, spreadsheet_id, sheet_name

creds_dict, spreadsheet_id, sheet_name = get_sheets_credentials()
creds = service_account.Credentials.from_service_account_info(creds_dict, scopes = SCOPES)
sheets_service = build("sheets", "v4", credentials = creds)

def read_sheets(spreadsheet_id = spreadsheet_id, sheet_name = sheet_name, header = True):
    """Reads a Google Sheets file. 

    Parameters:
        spreadsheet_id (str): The spreadsheet ID of the spreadsheet to be read (default constant SPREADSHEET_ID)
        sheet_name (str): The sheet name of the spreadsheet to be read (default constant SHEET_NAME)
        id_col (str): The identifier column of the spreadsheet that contains unique values (default constant ID)
        last_modified_col (str): The column containing most recent date and time that the row was modified
        day_lag (float): The number of days to offset when comparing the current version with a previous version
        header (bool): Whether the file has a header or not (default True)
        track_time (bool): Whether to track execution time (default True)

    Returns:
        df (pd.DataFrame): Dataframe of sales data
    """

    # Warm up if this function is run for the first time
    global WARMUP_DONE
    if not WARMUP_DONE:
        sheets_service.spreadsheets().values().get(
            spreadsheetId = spreadsheet_id,
            range = sheet_name + "!A1:A1"
        ).execute()
        WARMUP_DONE = True

    # Read data
    result = sheets_service.spreadsheets().values().get(
        spreadsheetId = spreadsheet_id,
        range = sheet_name + "!A:ZZZ",
        valueRenderOption = "UNFORMATTED_VALUE",
        dateTimeRenderOption = "FORMATTED_STRING"
    ).execute()

    values = result.get("values", [])
    # Put appropriate header if there is
    if header:
        df_sh = pd.DataFrame(values[1:], columns = values[0])
    else:
        df_sh = pd.DataFrame(values)
    df_sh = df_sh.fillna("").astype(str)
    return df_sh, datetime.datetime.now(ZoneInfo("Asia/Manila"))

def get_supabase_credentials():
    try:
        url = st.secrets["supabase"]["SUPABASE_URL"]
        key = st.secrets["supabase"]["SUPABASE_SERVICE_KEY"]
    except:
        load_dotenv()
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_SERVICE_KEY")
    return url, key

def read_supabase():
    url, key = get_supabase_credentials()
    supabase: Client = create_client(url, key)
    batch_size = 5000
    offset = 0
    all_rows = []
    while True:
        response = supabase.rpc("get_order_location_data", {"p_limit": batch_size, "p_offset": offset}).execute()
        rows = response.data
        if not rows:
            break
        all_rows.extend(rows)
        offset += batch_size
    df_sb = pd.DataFrame(all_rows)
    return df_sb, datetime.datetime.now(ZoneInfo("Asia/Manila"))


