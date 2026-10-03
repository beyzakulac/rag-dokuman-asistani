import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain
from trace_logger import trace_step

# .env dosyasındaki ortam değişkenlerini sisteme yükle
load_dotenv()


@trace_step(step_name="cevap_uretme")
def generate_answer(query: str, retrieved_docs: list):
    """
    Kullanıcının sorusunu ve ChromaDB'den getirilen belgeleri ücretsiz Gemini modeline göndererek
    bağlam tabanlı (RAG) bir cevap üretir.
    """
    # API anahtarını .env dosyasından çekiyoruz
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("API anahtarı bulunamadı. Lütfen .env dosyasını kontrol edin.")

    # Hızlı ve sabit sürüme sahip olğu için gemini-3.8-flash kullanmayı seçtim.
    llm = ChatGoogleGenerativeAI(model="gemini-3.8-flash", temperature=0, google_api_key=api_key)

    system_prompt = (
        "Sen profesyonel bir doküman asistanısın. Aşağıda sağlanan 'Bağlam' metinlerini "
        "kullanarak kullanıcının sorusunu yanıtla.\n"
        "Kurallar:\n"
        "1. Eğer cevabı aşağıdaki bağlamda bulamazsan, kesinlikle kendi bilgilerini uydurma ve "
        "'Sağlanan belgelerde bu bilgi bulunmamaktadır.' de.\n"
        "2. Cevabını verirken kullandığın bilgilerin hangi sayfalardan geldiğini mutlaka belirt.\n"
        "3. Cevabının en sonuna ayrı bir satırda 'Kaynak Sayfalar: [Sayfa X, Sayfa Y]' şeklinde bir not ekle.\n\n"
        "Bağlam:\n{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}")
    ])

    chain = create_stuff_documents_chain(llm, prompt)

    response = chain.invoke({
        "input": query,
        "context": retrieved_docs
    })

    # Kaynak sayfaları ayıkla
    sources = []
    for doc in retrieved_docs:
        page_num = doc.metadata.get("page", "Bilinmiyor")
        if isinstance(page_num, int):
            sources.append(page_num + 1)

    unique_sources = list(set(sources))

    return {
        "answer": response,
        "source_pages": unique_sources
    }