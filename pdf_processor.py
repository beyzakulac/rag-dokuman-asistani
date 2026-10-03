import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from trace_logger import trace_step


@trace_step(step_name="pdf_hazirligi")
def process_pdf(file_path: str, chunk_size: int = 1000, chunk_overlap: int = 200):
    """
    Belirtilen PDF dosyasını okur ve metni semantik bütünlüğü koruyarak parçalara ayırır.
     metadata (sayfa numaraları ) otomatik olarak korunur.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dosya bulunamadı: {file_path}")

    # 1. PDF dosyasını  okur.
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    # 2. Metin Parçalayıcıyı (Text Splitter) ayarla
    # RecursiveCharacterTextSplitter, paragrafları ve cümleleri bölmemeye özen gösterir
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ".", " ", ""]
    )

    # 3. Dökümanları chunks böl
    chunks = text_splitter.split_documents(documents)

    return chunks


# Test için (Sadece bu dosya doğrudan çalıştırıldığında tetiklenir)
if __name__ == "__main__":
    # Test etmek için proje dizinine 'test.pdf' adında bir dosya koyabilirsiniz
    test_pdf_path = "test.pdf"
    if os.path.exists(test_pdf_path):
        print("PDF işleniyor...")
        pdf_chunks = process_pdf(file_path=test_pdf_path, chunk_size=1000, chunk_overlap=200)
        print(f"Toplam {len(pdf_chunks)} parça oluşturuldu.")
        print(f"Örnek Parça (Sayfa {pdf_chunks[0].metadata['page']}):\n{pdf_chunks[0].page_content[:200]}...")