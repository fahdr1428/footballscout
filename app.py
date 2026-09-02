"""
Football Player Scouting & Recruitment Intelligence Platform.

Entry point. Declares the navigation; every page lives in `pages/` and every
model lives in `src/`.

    streamlit run app.py
"""

from __future__ import annotations

import streamlit as st

st.set_page_config(
    page_title="Scouting Intelligence",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

PAGES = {
    "Overview": [
        st.Page("pages/0_🏠_Home.py", title="Home", icon="🏠", url_path="home", default=True),
    ],
    "Players": [
        st.Page("pages/1_👤_Player_Search.py", title="Player Search", icon="👤", url_path="Player_Search"),
        st.Page("pages/2_📊_Player_Profile.py", title="Player Profile", icon="📊", url_path="Player_Profile"),
        st.Page("pages/3_🔎_Similar_Players.py", title="Similar Players", icon="🔎", url_path="Similar_Players"),
        st.Page("pages/5_🆚_Compare_Players.py", title="Compare Players", icon="🆚", url_path="Compare_Players"),
    ],
    "Recruitment": [
        st.Page("pages/4_🎯_Recruitment_Finder.py", title="Recruitment Finder", icon="🎯", url_path="Recruitment_Finder"),
        st.Page("pages/7_💎_Hidden_Gems.py", title="Hidden Gems", icon="💎", url_path="Hidden_Gems"),
    ],
    "Populations": [
        st.Page("pages/6_🧬_Player_Archetypes.py", title="Player Archetypes", icon="🧬", url_path="Player_Archetypes"),
        st.Page("pages/8_🌍_League_Explorer.py", title="League Explorer", icon="🌍", url_path="League_Explorer"),
    ],
    "Under the hood": [
        st.Page("pages/9_🔬_Model_Validation.py", title="Model Validation", icon="🔬", url_path="Model_Validation"),
        st.Page("pages/10_📖_Methodology.py", title="Methodology", icon="📖", url_path="Methodology"),
    ],
}

st.navigation(PAGES).run()
