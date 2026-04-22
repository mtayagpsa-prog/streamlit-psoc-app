import streamlit as st
import pandas as pd
from rapidfuzz import process, fuzz
import os
from datetime import datetime
import matplotlib.pyplot as plt

# -------------------------------
# CONFIG
# -------------------------------
MAX_RESULTS = 100
LOG_FILE = "coding_output.csv"

st.set_page_config(page_title="Textual Classification", layout="wide")

# -------------------------------
# for styling of the user interface UI
# -------------------------------
st.markdown("""
<style>
[data-testid="stSidebar"] {
    background-color: #2E3B4E;
}
[data-testid="stSidebar"] * {
    color: white !important;
}
[data-testid="stSidebar"] input,
[data-testid="stSidebar"] textarea,
[data-testid="stSidebar"] div {
    color: black !important;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------
# NAVIGATION
# -------------------------------
page = st.sidebar.selectbox("Navigation", ["Encoding", "Dashboard"])

# -------------------------------
# LOAD DATA
# -------------------------------
@st.cache_data
def load_psoc():
    df = pd.read_csv("listcode_2.csv", dtype=str)
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    df["occupation_title"] = df["occupation_title"].fillna("")
    df["occupation_title_clean"] = df["occupation_title"].str.lower()
    return df


@st.cache_data
def load_mapping():
    map_df = pd.read_csv("psocpsiccow with description_group.csv", dtype=str, low_memory=False)
    map_df.columns = map_df.columns.str.strip().str.lower().str.replace(" ", "_")
    return map_df


df = load_psoc()
map_df = load_mapping()

# =====================================================
# ==================== ENCODING ========================
# =====================================================
if page == "Encoding":

    st.title("Classifying Textual Responses to PSOC-PSIC-COW")
    st.caption("Encode occupation → map to PSOC → PSIC → COW")

    # -------------------------------
    # SIDEBAR INPUT
    # -------------------------------
    st.sidebar.header("Respondent Information")

    respondent_id = st.sidebar.text_input("Respondent ID")
    age = st.sidebar.number_input("Age", min_value=0, max_value=80)

    region = st.sidebar.text_input("Region")
    province = st.sidebar.text_input("Province")
    municipality = st.sidebar.text_input("Municipality")
    barangay = st.sidebar.text_input("Barangay")
    husn = st.sidebar.text_input("HUSN")

    raw_text = st.sidebar.text_area("Occupation (Write-in Response)")

    # -------------------------------
    # SEARCH
    # -------------------------------
    user_input = raw_text.strip()
    filtered_df = df

    if user_input:
        matches = process.extract(
            user_input,
            df["occupation_title_clean"],
            scorer=fuzz.WRatio,
            limit=MAX_RESULTS
        )

        indices = [m[2] for m in matches]
        scores = [m[1] for m in matches]

        filtered_df = df.iloc[indices].assign(score=scores)
        filtered_df = filtered_df.sort_values(by="score", ascending=False)

    filtered_df = filtered_df.head(MAX_RESULTS)

    # -------------------------------
    # PSOC SELECT
    # -------------------------------
    options = filtered_df.apply(
        lambda r: f"{r['occupation_title']} ({r['psoc_code']})",
        axis=1
    ).tolist()

    if not options:
        st.warning("No matches found.")
        st.stop()

    st.subheader("Selection of PSOC")
    selected = st.selectbox("Select PSOC", options)

    selected_row = filtered_df.iloc[options.index(selected)]
    selected_psoc = selected_row["psoc_code"]
    psoc_desc = selected_row["occupation_title"]

    # -------------------------------
    # FILTER MAPPING
    # -------------------------------
    valid_map = map_df[map_df["psoc_code"] == selected_psoc]

    if valid_map.empty:
        st.warning("No mapping found.")
        st.stop()

    # -------------------------------
    # PSIC GROUP
    # -------------------------------
    group_df = valid_map[["psic_group_code", "psic_group_desc"]].drop_duplicates()

    group_options = {
        f"{r.psic_group_code} - {r.psic_group_desc}": (r.psic_group_code, r.psic_group_desc)
        for _, r in group_df.iterrows()
    }

    st.subheader("Selection of Industry Group")
    selected_group_label = st.selectbox("Select Industry Group", list(group_options.keys()))
    selected_group_code, selected_group_desc = group_options[selected_group_label]

    # -------------------------------
    # PSIC
    # -------------------------------
    psic_df = valid_map[
        valid_map["psic_group_code"] == selected_group_code
    ][["psic_code", "psic_desc"]].drop_duplicates()

    psic_options = {
        f"{r.psic_code} - {r.psic_desc}": (r.psic_code, r.psic_desc)
        for _, r in psic_df.iterrows()
    }

    st.subheader("Selection of Industry Class")
    selected_psic_label = st.selectbox("Select PSIC", list(psic_options.keys()))
    selected_psic, selected_psic_desc = psic_options[selected_psic_label]

    # -------------------------------
    # COW
    # -------------------------------
    cow_df = valid_map[
        valid_map["psic_code"] == selected_psic
    ][["cow_code", "cow_desc"]].drop_duplicates()

    cow_options = {
        f"{r.cow_code} - {r.cow_desc}": (r.cow_code, r.cow_desc)
        for _, r in cow_df.iterrows()
    }

    st.subheader("Selection of COW")
    selected_cow_label = st.selectbox("Select Class of Worker", list(cow_options.keys()))
    selected_cow, selected_cow_desc = cow_options[selected_cow_label]

    # -------------------------------
    # REVIEW/PREVIEW OF THE SELECTED PSOC-PSIC_GROUP-PSIC_CLASS-COW
    # -------------------------------
    st.divider()
    st.subheader("Review Final Selection")

    st.write(f"PSOC: {selected_psoc} - {psoc_desc}")
    st.write(f"PSIC Group: {selected_group_code} - {selected_group_desc}")
    st.write(f"PSIC Class: {selected_psic} - {selected_psic_desc}")
    st.write(f"COW: {selected_cow} - {selected_cow_desc}")

    # -------------------------------
    # SAVE
    # -------------------------------
    
    st.markdown("""
    <style>
    .stButton > button {
    background-color: green;
    color: white;
    }
    </style>
    """, unsafe_allow_html=True)

    
    if st.button("Save / Update Record"):

        if not respondent_id:
            st.error("Respondent ID is required")
            st.stop()

        new_data = pd.DataFrame([{
            "respondent_id": respondent_id,
            "age": age,
            "region": region,
            "province": province,
            "municipality": municipality,
            "barangay": barangay,
            "husn": husn,
            "raw_text": raw_text,

            "psoc_code": selected_psoc,
            "psoc_desc": psoc_desc,
            "psic_group_code": selected_group_code,
            "psic_group_desc": selected_group_desc,
            "psic_code": selected_psic,
            "psic_desc": selected_psic_desc,
            "cow_code": selected_cow,
            "cow_desc": selected_cow_desc,

            "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }])

        if os.path.exists(LOG_FILE):
            existing = pd.read_csv(LOG_FILE, dtype=str)

            if respondent_id in existing["respondent_id"].values:
                idx = existing[existing["respondent_id"] == respondent_id].index[0]
                existing.loc[idx] = new_data.iloc[0]
                existing.to_csv(LOG_FILE, index=False)
                st.success("✅ Record updated!")
            else:
                new_data.to_csv(LOG_FILE, mode="a", header=False, index=False)
                st.success("✅ Record saved!")
        else:
            new_data.to_csv(LOG_FILE, index=False)
            st.success("✅ Record saved!")

    # -------------------------------
    # VIEW DATA
    # -------------------------------
    
    st.markdown("""
    <style>
    .stDownloadButton > button {
        background-color: #FF5733;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

    st.divider()
    st.subheader("Saved Records")

    if os.path.exists(LOG_FILE):
        saved_df = pd.read_csv(LOG_FILE, dtype=str)
        st.dataframe(saved_df)

        with open(LOG_FILE, "rb") as f:
            st.download_button("Download CSV", f, file_name="coding_output.csv")
        
# =====================================================
# ==================== DASHBOARD =======================
# =====================================================
elif page == "Dashboard":

    st.title("Dashboard")

    if not os.path.exists(LOG_FILE):
        st.warning("No data available yet.")
        st.stop()

    df = pd.read_csv(LOG_FILE, dtype=str)

    # FILTERS
    st.sidebar.subheader("Filters")

    region_filter = st.sidebar.selectbox("Region", ["All"] + sorted(df["region"].dropna().unique()))
    if region_filter != "All":
        df = df[df["region"] == region_filter]

    province_filter = st.sidebar.selectbox("Province", ["All"] + sorted(df["province"].dropna().unique()))
    if province_filter != "All":
        df = df[df["province"] == province_filter]

    # METRICS
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Records", len(df))
    col2.metric("Unique Occupations", df["psoc_code"].nunique())
    col3.metric("Regions Covered", df["region"].nunique())

    st.divider()

    # TOP OCCUPATIONS
    st.subheader("Top Occupations")

    top_n = st.selectbox("Top N", [10, 20])
    top_psoc = df["psoc_desc"].value_counts().head(top_n).reset_index()
    top_psoc.columns = ["Occupation", "Count"]

    col1, col2 = st.columns([1, 2])
    col1.dataframe(top_psoc)

    fig, ax = plt.subplots()
    ax.barh(top_psoc["Occupation"], top_psoc["Count"])
    ax.set_title("Top Occupations")
    col2.pyplot(fig)

    st.divider()

    # REGION DISTRIBUTION
    st.subheader("Region Distribution")
    region_counts = df["region"].value_counts().head(10)

    fig2, ax2 = plt.subplots()
    ax2.bar(region_counts.index, region_counts.values)
    ax2.set_xticklabels(region_counts.index, rotation=45)
    st.pyplot(fig2)

    # AGE DISTRIBUTION
    if "age" in df.columns:
        st.subheader("Age Distribution")
        df["age"] = pd.to_numeric(df["age"], errors="coerce")

        fig3, ax3 = plt.subplots()
        ax3.hist(df["age"].dropna(), bins=10)
        st.pyplot(fig3)