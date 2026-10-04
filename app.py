import streamlit as st
from ui_components import render_sidebar, render_chat_tab, render_evaluation_tab, render_pca_map_tab

# Sayfa Yapılandırması
st.set_page_config(page_title="RAG Doküman Asistanı", layout="wide")
st.title("📚 RAG Tabanlı Doküman Asistanı")

# Session State Tanımlamaları
if "vector_db" not in st.session_state:
    st.session_state.vector_db = None
if "chunks" not in st.session_state:
    st.session_state.chunks = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "last_query" not in st.session_state:
    st.session_state.last_query = None

# Sol Menü (Ayarlar ve Yükleme)
search_type = render_sidebar()

# Ana Ekran (Sekmeler)
if st.session_state.vector_db is None:
    st.info("👈 Lütfen sol menüden bir PDF yükleyip 'Belgeyi İşle' butonuna basın.")
else:
    tab1, tab2, tab3 = st.tabs(["💬 Sohbet", "📊 Normal vs MMR Kıyaslaması", "🌌 Embedding Haritası"])

    with tab1:
        # Sohbet ekranını çiz ve yeni bir soru sorulduysa bunu kaydet
        user_query = render_chat_tab(search_type)
        if user_query:
            st.session_state.last_query = user_query

    with tab2:
        # Son sorulan soruya göre değerlendirme tablosunu çiz
        render_evaluation_tab(st.session_state.last_query)

    with tab3:
        # PCA Haritasını çiz
        render_pca_map_tab()