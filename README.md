<img width="1819" height="819" alt="image" src="https://github.com/user-attachments/assets/f40dba45-4bba-46b5-89c0-c3431668e84a" /># streamlit-mongolian-ocr-qa
A Streamlit web application for bilingual (Mongolian/English) OCR using EasyOCR, storing results in SQLite, and enabling semantic Q&amp;A using LangChain, FAISS vector store, and Ollama (MongPT).
# 🔍 Mongolian OCR & Q&A Database System

Streamlit болон OCR ашиглан зургаас текст таньж, SQLite өгөгдлийн сан болон хиймэл оюун ухаан (LangChain + FAISS + Ollama) ашиглан хайлт хийх болон асуулт хариулт авах боломжтой вэб аппликэйшн.

## 🚀 Үндсэн боломжууд
- **Монгол ба Англи хэлний OCR:** EasyOCR ашиглан бичиг баримт болон зургаас хоёр хэлээр текст таних.
- **SQLite Database:** Танигдсан үр дүн, итгэлцүүр (confidence score), координатуудыг өгөгдлийн санд найдвартай хадгалах.
- **Semantic Q&A (MongPT):** Танигдсан баримтуудынхаа агуулга дээр үндэслэн хиймэл оюун ухаанаас асуулт асууж хариулт авах.
- **Экспорт & Удирдлага:** Өгөгдлийг JSON формат руу татаж авах болон санг цэвэрлэх, удирдах.

## 🛠️ Ашигласан технологиуд
- **Frontend/UI:** Streamlit
- **OCR Engine:** EasyOCR (PyTorch)
- **Database:** SQLite3, Pandas
- **Vector Search & LLM:** LangChain, FAISS (`sentence-transformers/all-MiniLM-L6-v2`), Ollama (MongPT)
