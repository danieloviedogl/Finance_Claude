# app.py (Archivo Principal - Solo Configuración y Navegación)
import streamlit as st
import pandas as pd

PAGE_NAME = "Stock Analysis Pro"

# --- Configuración de la Página (PRIMER COMANDO ST) ---
st.set_page_config(
    page_title="Stock Analysis Pro",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Definición de Páginas ---
# Página de Introducción (Contenido ahora en introduction.py o pages/0_...)
about_page = st.Page(
    "pages/0_Introduction.py",
    title="Introduction",
    icon="🏠",
    default=True
)
# Página de Análisis de Stock
stock_page = st.Page(
    "pages/1_Stock_Analysis.py",
    title="Stock Analysis          👈 Click Here!",
    icon="📈"
)
# Página de Análisis de Portafolio
portfolio_page = st.Page(
    "pages/2_Portfolio_Analysis.py",
    title="Portafolio Analysis",
    icon="📊"
)

# --- Construcción de la Navegación ---
pg = st.navigation(
    {
        "Information": [about_page],
        "Tools": [stock_page, portfolio_page],
    }
)

# --- Inicialización del Estado de Sesión Global ---
# (Mantenemos esto aquí si es necesario globalmente)
default_portfolio = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "JPM", "V"]
if 'portfolio_tickers' not in st.session_state:
    st.session_state.portfolio_tickers = default_portfolio.copy()
if 'current_ticker' not in st.session_state:
    st.session_state.current_ticker = "AAPL"
if 'risk_free_rate' not in st.session_state:
    st.session_state.risk_free_rate = 0.04
if 'benchmark' not in st.session_state:
    st.session_state.benchmark = 'SPY'

# --- SHARED ON ALL PAGES ---
st.logo("assets/logo_1.png")
st.sidebar.text(f"Made by Daniel Oviedo")

# --- Ejecuta la Lógica de la Página Seleccionada ---
pg.run()

# --- app.py TERMINA AQUÍ ---
# No más código que muestre contenido en app.py