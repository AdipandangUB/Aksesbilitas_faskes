# CSS tambahan untuk mencegah flickering
st.markdown("""
<style>
    /* Fix untuk peta agar tidak reload */
    .folium-map {
        width: 100% !important;
        height: 650px !important;
        border: 1px solid #ddd;
        border-radius: 10px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        transition: none !important;
        animation: none !important;
    }
    
    /* Mencegah re-rendering pada tab */
    .stTabs [data-baseweb="tab-panel"] {
        transition: none !important;
        animation: none !important;
    }
    
    /* Stabilkan container */
    .stContainer {
        transition: none !important;
    }
    
    /* Fix untuk metric cards */
    [data-testid="stMetricValue"] {
        transition: none !important;
    }
</style>
""", unsafe_allow_html=True)
