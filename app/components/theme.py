# app/components/theme.py
import streamlit as st

def apply_compact_theme():
    st.markdown(
        """
        <style>
        /* tighter layout */
        .block-container { padding-top: 1rem; padding-bottom: 2rem; }
        section.main > div { padding-top: 0rem; }
        /* compact inputs */
        .stButton>button, .stDownloadButton>button, .st-emotion-cache-1vt4y43, .stSlider { transform: scale(0.98); }
        /* smaller tables on mobile */
        @media (max-width: 640px) {
          .stDataFrame { font-size: 0.8rem; }
          h1, h2 { font-size: 1.2rem; }
          h3 { font-size: 1rem; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )