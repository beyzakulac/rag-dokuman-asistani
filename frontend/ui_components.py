import streamlit as st
import os
import tempfile
import pandas as pd
import plotly.express as px
from sklearn.decomposition import PCA
from pdf_processor import process_pdf
from vector_db import create_vector_db, search_normal, search_mmr, embedding_model
from rag_chain import generate_answer

def render_sidebar():
    """Sol menüyü çizer, belge yükleme işlemlerini yönetir ve seçilen arama tipini döndürür."""
    with st.sidebar:
        st.header("⚙️ Ayarlar ve Yükleme")
        uploaded_file = st.file_uploader("Bir PDF dosyası yükleyin", type=["pdf"])

        if st.button("Belgeyi İşle") and uploaded_file is not None:
            with st.spinner("PDF işleniyor ve Vektör Veritabanı oluşturuluyor..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = tmp_file.name

                try:
                    chunks = process_pdf(tmp_path)
                    vector_db = create_vector_db(chunks)
                    st.session_state.chunks = chunks
                    st.session_state.vector_db = vector_db
                    st.success("Belge başarıyla işlendi!")
                except Exception as e:
                    st.error("🚨 Hata: Yüklediğiniz dosya geçerli bir PDF formatında değil veya bozuk. Lütfen gerçek ve okunabilir bir PDF dosyası yükleyin.")
                    st.stop()
                finally:
                    if 'tmp_path' in locals() and os.path.exists(tmp_path):
                        try:
                            os.remove(tmp_path)
                        except PermissionError:
                            pass

        st.divider()
        st.subheader("🔍 Arama Stratejisi")
        search_type = st.radio(
            "Nasıl bir arama yapalım?",
            ["🎯 Nokta Atışı Arama (En Yakın Sonuçlar)",
            "🌐 Çok Yönlü Arama (Farklı Sayfalardan Çeşitli Bilgiler)"],
            help="**🎯 Nokta Atışı Arama:** Sorduğunuz soruya en çok benzeyen ve doğrudan cevap içeren kısımları getirir.\n\n**🌐 Çok Yönlü Arama:** Sorunuzla ilgili belgenin farklı yerlerinde geçen çeşitli bilgileri toparlar."
        )
        return search_type

def render_chat_tab(search_type):
    """Sohbet arayüzünü çizer ve LLM ile etkileşimi yönetir."""
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_query = st.chat_input("PDF içeriği hakkında bir soru sorun...")
    if user_query:
        with st.chat_message("user"):
            st.markdown(user_query)
        st.session_state.chat_history.append({"role": "user", "content": user_query})

        with st.chat_message("assistant"):
            with st.spinner("Yanıt üretiliyor..."):
                if search_type.startswith("🎯") or search_type.startswith("Normal"):
                    retrieved_docs = search_normal(user_query, st.session_state.vector_db)
                else:
                    retrieved_docs = search_mmr(user_query, st.session_state.vector_db)

                response_data = generate_answer(user_query, retrieved_docs)
                answer = response_data["answer"]
                st.markdown(answer)
        st.session_state.chat_history.append({"role": "assistant", "content": answer})
    return user_query

def render_evaluation_tab(user_query):
    """Normal ve MMR arama sonuçlarını tablo halinde kıyaslar."""
    st.write("Son sorduğunuz soru için **Normal Arama** ve **MMR (Çeşitlilik)** yöntemlerinin getirdiği farklı kaynak parçalarının analizi:")
    if user_query:
        docs_normal = search_normal(user_query, st.session_state.vector_db)
        docs_mmr = search_mmr(user_query, st.session_state.vector_db)

        data = {
            "Yöntem": ["Normal"] * len(docs_normal) + ["MMR"] * len(docs_mmr),
            "Sayfa": [d.metadata.get("page", 0) + 1 for d in docs_normal] + [d.metadata.get("page", 0) + 1 for d in docs_mmr],
            "Metin Özeti (İlk 150 Karakter)": [d.page_content[:150].replace("\n", " ") + "..." for d in docs_normal] + [d.page_content[:150].replace("\n", " ") + "..." for d in docs_mmr]
        }
        df = pd.DataFrame(data)
        st.dataframe(df, use_container_width=True)
    else:
        st.warning("Bu tabloyu görmek için sohbette bir soru sormalısınız.")

def render_pca_map_tab():
    """Belge parçalarını 2 boyutlu uzayda çizer."""
    st.write("Belgenizdeki veri parçalarının anlamsal uzaydaki 2 boyutlu görünümünün haritası:.")
    if st.button("Haritayı Çiz (Gelişmiş)"):
        with st.spinner("Vektörler 2 boyuta indirgeniyor (PCA)..."):
            texts = [c.page_content for c in st.session_state.chunks]
            pages = [str(c.metadata.get("page", 0) + 1) for c in st.session_state.chunks]

            embeddings = embedding_model.embed_documents(texts)
            if len(embeddings) < 2:
                st.warning("⚠️ Harita çizilebilmesi için belgenizin en az 2 parçaya (chunk) bölünmüş olması gerekir. Lütfen daha uzun bir PDF belgesi yükleyin.")
            else:
                pca = PCA(n_components=2)
                reduced_embeddings = pca.fit_transform(embeddings)

                plot_df = pd.DataFrame({
                    "X": reduced_embeddings[:, 0],
                    "Y": reduced_embeddings[:, 1],
                    "Sayfa": pages,
                    "Metin": [t[:100] + "..." for t in texts]
                })

                fig = px.scatter(
                    plot_df, x="X", y="Y", color="Sayfa",
                    hover_data=["Metin"],
                    title="Doküman Semantik Vektör Haritası (PCA Uzayı)"
                )
                st.plotly_chart(fig, use_container_width=True)