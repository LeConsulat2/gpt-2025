import streamlit as st


def apply_orange_gradient():
    st.write(
        """
        <style>
        /* Remove the white gap at the top */
        [data-testid="stHeader"] {
            background: linear-gradient(135deg, #fff6d1, #ffe6a3, #ffd580);
            color: #663800;
            border-bottom: 1px solid #ffd580;
        }

        /* Ensure sidebar matches gradient */
        [data-testid="stSidebar"] {
            background: linear-gradient(135deg, #fff6d1, #ffe6a3, #ffd580);
        }

        /* Apply gradient background to the entire app */
        html, body, .stApp {
            background: linear-gradient(135deg, #fff6d1, #ffe6a3, #ffd580);
            color: #663800; /* Text color */
            margin: 0;
            padding: 0;
            height: 100%;
        }

        /* Adjust text and heading colors for better visibility */
        h1, h2, h3, h4, h5, h6, label, p {
            color: #663800; /* Use darker shade for text */
        }

        /* Style buttons */
        .stButton button {
            background-color: #ff9900;
            color: white;
        }

        /* Input fields styling */
        .stTextInput input, .stSelectbox select, .stTextarea textarea {
            background-color: #fff4cc;
            color: #663800;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
