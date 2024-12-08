import streamlit as st


def apply_blue_gradient():
    st.write(
        """
        <style>
        /* Remove white gap at the top */
        [data-testid="stHeader"] {
            background: linear-gradient(135deg, #d1e8ff, #b3d1f7, #a3c8f2);
            color: #083b66;
            border-bottom: 1px solid #b3d1f7;
        }

        /* Apply gradient background to the entire app */
        html, body, .stApp {
            background: linear-gradient(135deg, #d1e8ff, #b3d1f7, #a3c8f2);
            color: #000; /* Darker text for readability */
            margin: 0;
            padding: 0;
            height: 100%;
        }

        /* Adjust text and heading colors */
        h1, h2, h3, h4, h5, h6, label, p {
            color: #083b66; /* Darker shade for text */
        }

        /* Style buttons */
        .stButton button {
            background-color: #1f77b4;
            color: white;
        }

        /* Input fields styling */
        .stTextInput input, .stSelectbox select, .stTextarea textarea {
            background-color: #f0f8ff;
            color: #083b66;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
