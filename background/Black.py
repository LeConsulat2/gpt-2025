import streamlit as st


def dark_theme():
    st.write(
        """
        <style>
        /* Force ALL elements to have black background and white text */
        * {
            background-color: black !important;
            color: white !important;
        }

        /* General background and font colors */
        html, body, .stApp, .main, [data-testid="stSidebar"] {
            background-color: black !important;
            color: white !important;
        }

        /* Sidebar specific styling */
        .css-1d391kg, .css-1lcbmhc, .css-1y4p8pa, .css-18e3th9,
        [data-testid="stSidebarNav"], .css-pkbazv {
            background-color: black !important;
            color: white !important;
        }

        /* Expander styling */
        .streamlit-expander {
            background-color: black !important;
            color: white !important;
            border: 1px solid #333 !important;
        }

        /* All text elements */
        p, h1, h2, h3, h4, h5, h6, span, div, label, .streamlit-expanderHeader {
            color: white !important;
        }

        /* Input fields dark mode */
        .stTextInput input, .stNumberInput input, 
        .stPasswordInput input, .stSelectbox select, 
        .stTextarea textarea, .stDateInput input {
            background-color: black !important;
            color: white !important;
            border: 1px solid #333 !important;
        }

        /* Chat message styling */
        [data-testid="stChatMessage"], 
        .stChatFloatingInputContainer {
            background-color: black !important;
            border: 1px solid #333 !important;
        }

        /* Chat message content */
        [data-testid="stChatMessage"] p {
            color: white !important;
        }

        /* Markdown content */
        .stMarkdown, .css-10trblm {
            color: white !important;
        }

        /* Remove any white borders or gaps */
        [data-testid="stHeader"], 
        [data-testid="stToolbar"],
        [data-testid="stDecoration"],
        footer, .css-1outfnb {
            background-color: black !important;
            color: white !important;
        }

        /* Expander and other interactive elements */
        .streamlit-expanderHeader:hover,
        .streamlit-expanderHeader:focus {
            background-color: #222 !important;
        }

        /* Scrollbars */
        ::-webkit-scrollbar {
            background-color: black !important;
        }
        
        ::-webkit-scrollbar-thumb {
            background-color: #333 !important;
        }

        /* Any additional elements */
        .element-container, .stMarkdown, 
        .css-ocqkz7, .css-10trblm, .css-qrbaxs {
            background-color: black !important;
            color: white !important;
        }

        /* Force white text for all nested elements */
        * div, * span, * p {
            color: white !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
