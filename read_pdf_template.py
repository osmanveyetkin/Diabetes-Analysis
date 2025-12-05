from pypdf import PdfReader

def extract_text_from_pdf(pdf_path):
    try:
        reader = PdfReader(pdf_path)
        print(f"--- PDF İÇERİĞİ BAŞLANGIÇ (ILK 3 SAYFA): {pdf_path} ---")
        # Only read first 3 pages
        max_pages = min(3, len(reader.pages))
        for i in range(max_pages):
            print(f"\n--- SAYFA {i+1} ---")
            print(reader.pages[i].extract_text())
        print("\n--- PDF İÇERİĞİ BİTİŞ ---")
    except Exception as e:
        print(f"Hata oluştu: {e}")

if __name__ == "__main__":
    extract_text_from_pdf("örnek_rapor_şablonu.pdf")
