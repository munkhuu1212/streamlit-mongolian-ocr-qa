# OCR үр дүнг SQLite database-д хадгалах - Streamlit веб аппликэйшн
import streamlit as st
import easyocr
import sqlite3
import os
from datetime import datetime
import json
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
from PIL import Image, ImageDraw, ImageFont
import io
# Q&A системийн import-ууд
try:
    from langchain.chains import RetrievalQA
    from langchain.docstore.document import Document
    from langchain_community.vectorstores import FAISS
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.llms import Ollama
    QA_AVAILABLE = True
except ImportError:
    QA_AVAILABLE = False

class OCRDatabase:
    def __init__(self, db_name="ocr_results.db"):
        self.db_name = db_name
        self.init_database()
        
    def init_database(self):
        """Database болон table үүсгэх"""
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        
        # OCR үр дүнгийн table үүсгэх
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS ocr_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_path TEXT NOT NULL,
                image_name TEXT NOT NULL,
                detected_text TEXT,
                confidence_scores TEXT,
                bounding_boxes TEXT,
                processed_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                notes TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def convert_numpy_types(self, obj):
        """Numpy төрлүүдийг Python стандарт төрөлд хөрвүүлэх"""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, list):
            return [self.convert_numpy_types(item) for item in obj]
        elif isinstance(obj, dict):
            return {key: self.convert_numpy_types(value) for key, value in obj.items()}
        else:
            return obj
    
    def get_record_count(self):
        """Database-д хэдэн бичлэг байгааг тоолох"""
        try:
            conn = sqlite3.connect(self.db_name)
            cursor = conn.cursor()
            
            cursor.execute('SELECT COUNT(*) FROM ocr_results')
            count = cursor.fetchone()[0]
            conn.close()
            
            return count
        except Exception as e:
            st.error(f"Database тооллогын алдаа: {e}")
            return 0
    
    def save_ocr_result(self, image_path, results, notes=""):
        """OCR үр дүнг database-д хадгалах"""
        try:
            conn = sqlite3.connect(self.db_name)
            cursor = conn.cursor()
            
            # Өгөгдөл бэлтгэх
            image_name = os.path.basename(image_path)
            detected_texts = []
            confidence_scores = []
            bounding_boxes = []
            
            for bbox, text, confidence in results:
             detected_texts.append(clean_ocr_text(text))
             confidence_scores.append(self.convert_numpy_types(confidence))
             bounding_boxes.append(self.convert_numpy_types(bbox))

            
            # JSON format-аар хадгалах
            all_text = " | ".join(detected_texts)
            
            try:
                confidence_json = json.dumps(confidence_scores)
                bbox_json = json.dumps(bounding_boxes)
            except TypeError as e:
                # Backup - бүх өгөгдлийг string болгох
                confidence_json = json.dumps([str(c) for c in confidence_scores])
                bbox_json = json.dumps([str(b) for b in bounding_boxes])
            
            # Database-д оруулах
            cursor.execute('''
                INSERT INTO ocr_results 
                (image_path, image_name, detected_text, confidence_scores, bounding_boxes, notes)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (image_path, image_name, all_text, confidence_json, bbox_json, notes))
            
            result_id = cursor.lastrowid
            conn.commit()
            conn.close()
            
            return result_id
        except Exception as e:
            st.error(f"Database хадгалалтын алдаа: {e}")
            return None
    
    def get_all_results(self):
        """Бүх OCR үр дүнг DataFrame болгон авах"""
        try:
            conn = sqlite3.connect(self.db_name)
            
            df = pd.read_sql_query('''
                SELECT id, image_name, detected_text, processed_date, notes
                FROM ocr_results 
                ORDER BY processed_date DESC
            ''', conn)
            
            conn.close()
            return df
        except Exception as e:
            st.error(f"Database уншилтын алдаа: {e}")
            return pd.DataFrame()
    
    def search_by_text(self, search_text):
        """Текстээр хайх"""
        try:
            conn = sqlite3.connect(self.db_name)
            
            df = pd.read_sql_query('''
                SELECT id, image_name, detected_text, processed_date, notes
                FROM ocr_results 
                WHERE detected_text LIKE ? 
                ORDER BY processed_date DESC
            ''', conn, params=(f'%{search_text}%',))
            
            conn.close()
            return df
        except Exception as e:
            st.error(f"Хайлтын алдаа: {e}")
            return pd.DataFrame()
    
    def get_detailed_result(self, result_id):
        """Тодорхой бичлэгийн дэлгэрэнгүй мэдээлэл авах"""
        try:
            conn = sqlite3.connect(self.db_name)
            cursor = conn.cursor()
            
            cursor.execute('SELECT * FROM ocr_results WHERE id = ?', (result_id,))
            result = cursor.fetchone()
            conn.close()
            
            return result
        except Exception as e:
            st.error(f"Дэлгэрэнгүй өгөгдөл авахад алдаа: {e}")
            return None
    
    def export_to_json(self):
        """Бүх OCR үр дүнг JSON format-аар буцаах"""
        try:
            conn = sqlite3.connect(self.db_name)
            cursor = conn.cursor()
            
            cursor.execute('SELECT * FROM ocr_results ORDER BY processed_date DESC')
            results = cursor.fetchall()
            conn.close()
            
            if not results:
                return None
            
            # JSON өгөгдөл бэлтгэх
            json_data = {
                "export_info": {
                    "total_records": len(results),
                    "export_date": datetime.now().isoformat(),
                    "description": "OCR танилтын үр дүн"
                },
                "ocr_results": []
            }
            
            for row in results:
                try:
                    # JSON string-уудыг буцааж parse хийх
                    confidence_scores = json.loads(row[4]) if row[4] else []
                    bounding_boxes = json.loads(row[5]) if row[5] else []
                except json.JSONDecodeError:
                    confidence_scores = []
                    bounding_boxes = []
                
                # Statistics тооцоолох
                avg_confidence = 0
                if confidence_scores:
                    try:
                        numeric_scores = []
                        for score in confidence_scores:
                            if isinstance(score, str):
                                numeric_scores.append(float(score))
                            else:
                                numeric_scores.append(float(score))
                        avg_confidence = sum(numeric_scores) / len(numeric_scores)
                    except:
                        avg_confidence = 0
                
                record = {
                    "id": row[0],
                    "image_path": row[1],
                    "image_name": row[2],
                    "detected_text": row[3],
                    "confidence_scores": confidence_scores,
                    "bounding_boxes": bounding_boxes,
                    "processed_date": row[6],
                    "notes": row[7],
                    "statistics": {
                        "text_length": len(row[3]) if row[3] else 0,
                        "average_confidence": round(avg_confidence, 4),
                        "detected_blocks": len(confidence_scores),
                        "words_count": len(row[3].split()) if row[3] else 0
                    }
                }
                json_data["ocr_results"].append(record)
            
            return json_data
        except Exception as e:
            st.error(f"JSON экспортын алдаа: {e}")
            return None
    
    def clear_database(self):
        """Database-г цэвэрлэх"""
        try:
            conn = sqlite3.connect(self.db_name)
            cursor = conn.cursor()
            
            cursor.execute('DELETE FROM ocr_results')
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            st.error(f"Database цэвэрлэлтийн алдаа: {e}")
            return False

# OCR Reader-г кэш хийх (удаан ачаалагддаг учир)
@st.cache_resource
def load_ocr_reader():
    """OCR reader ачаалах"""
    try:
        return easyocr.Reader(['en', 'mn'])
    except Exception as e:
        st.error(f"OCR reader ачаалалтын алдаа: {e}")
        return None

# Database-г кэш хийх
@st.cache_resource
def get_database():
    """Database холболт авах"""
    return OCRDatabase()

def clean_ocr_text(text):
    """OCR-оор танигдсан текстээс латин үсгийг кирилл рүү хөрвүүлнэ, тоог ч хамт"""
    replacements = {
        'A': 'А', 'B': 'В', 'C': 'С', 'E': 'Е', 'H': 'Н', 'I': 'И', 'J': 'Ж',
        'K': 'К', 'M': 'М', 'O': 'О', 'P': 'Р', 'S': 'С', 'T': 'Т', 'X': 'Х',
        'Y': 'Ү', 'a': 'а', 'c': 'с', 'e': 'е', 'o': 'о', 'p': 'р', 'x': 'х',
        'y': 'у', 'n': 'п',
        'l': '1', 'I': '1', 'O': '0', 'Q': '0'
    }
    return ''.join(replacements.get(ch, ch) for ch in text)

def dual_language_ocr(image_path):
    """EasyOCR-г Монгол + Англи хэлээр 2 удаа ажиллуулж үр дүнг нэгтгэнэ"""
    try:
        reader_mn = easyocr.Reader(['mn'])
        reader_en = easyocr.Reader(['en'])

        results_mn = reader_mn.readtext(image_path)
        results_en = reader_en.readtext(image_path)

        # 2 үр дүнг нэгтгэх
        combined = results_mn + results_en

        # Text давхардлыг арилгах (confidence-р шүүж)
        cleaned_results = {}
        for bbox, text, conf in combined:
            text = text.strip()
            if text in cleaned_results:
                if conf > cleaned_results[text][1]:
                    cleaned_results[text] = (bbox, conf)
            else:
                cleaned_results[text] = (bbox, conf)

        final_results = [(bbox, text, conf) for text, (bbox, conf) in cleaned_results.items()]
        return final_results

    except Exception as e:
        st.error(f"OCR (2 хэлээр) алдаа: {e}")
        return []

def draw_boxes_on_image(image, results):
    """Зураг дээр text box-уудыг зурах - PIL ашиглан"""
    try:
        # PIL Image-г copy хийх
        img_copy = image.copy()
        draw = ImageDraw.Draw(img_copy)
        
        # Font загвар
        try:
            # Windows дээр ажиллана
            font = ImageFont.truetype("arial.ttf", 12)
        except:
            try:
                # Linux/Mac дээр ажиллана
                font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 12)
            except:
                # Default font
                font = ImageFont.load_default()
        
        for bbox, text, confidence in results:
            # Bounding box координатууд
            points = bbox
            
            # Rectangle зурах (4 цэгийг rectangle болгон хувиргах)
            x_coords = [point[0] for point in points]
            y_coords = [point[1] for point in points]
            min_x, max_x = min(x_coords), max(x_coords)
            min_y, max_y = min(y_coords), max(y_coords)
            
            # Box зурах
            draw.rectangle([min_x, min_y, max_x, max_y], outline='green', width=2)
            
            # Confidence score бичих
            confidence_text = f"{confidence:.2f}"
            draw.text((min_x, max(min_y-15, 0)), confidence_text, fill='red', font=font)
        
        return img_copy
    except Exception as e:
        st.error(f"Зураг зурахад алдаа: {e}")
        return image

def main():
    st.set_page_config(
        page_title="OCR Database System",
        page_icon="🔍",
        layout="wide"
    )
    
    st.title("🔍 OCR Database System")
    st.markdown("---")
    
    # Database холболт
    db = get_database()
    
    # record_count-ыг энд тодорхойлох
    record_count = db.get_record_count()
    
    # Sidebar navigation
    st.sidebar.title("🧭 Navigation")
    
    # Q&A хэсэг нэмэх эсэх
    pages = ["📷 OCR процесс", "📊 Database хайлт", "📁 Экспорт/Удирдлага"]
    if QA_AVAILABLE and record_count > 0:
        pages.append("🤖 Q&A систем")
    
    page = st.sidebar.radio("Хуудас сонгох:", pages)
    
    # Database статус
    st.sidebar.info(f"📊 Нийт бичлэг: {record_count}")
    
    # Q&A системийн статус
    if not QA_AVAILABLE:
        st.sidebar.warning("⚠️ Q&A систем ажиллахгүй (langchain суулгаагүй)")
    elif record_count == 0:
        st.sidebar.info("ℹ️ Q&A систем: Өгөгдөл хэрэгтэй")
    if page == "📷 OCR процесс":
        st.header("📷 Зураг танилт (OCR)")
        
        # Файл оруулах
        uploaded_files = st.file_uploader(
            "Зураг файл сонгоно уу:",
            type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
            accept_multiple_files=True
        )
        
        if uploaded_files:
            # OCR reader ачаалах
            with st.spinner("OCR систем ачаалж байна..."):
                reader = load_ocr_reader()
            
            if reader is None:
                st.error("OCR систем ачаалагдсангүй!")
                return
            
            for uploaded_file in uploaded_files:
                st.subheader(f"📄 {uploaded_file.name}")
                
                try:
                    # Зургийг харуулах
                    image = Image.open(uploaded_file)
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.image(image, caption="Анхны зураг", use_column_width=True)
                    
                    # OCR хийх
                    with st.spinner(f"'{uploaded_file.name}' боловсруулж байна..."):
                        # Temporary файл үүсгэх
                        with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{uploaded_file.name}") as tmp_file:
                            tmp_file.write(uploaded_file.getvalue())
                            tmp_path = tmp_file.name
                        
                        try:
                            # OCR хийх
                            results = dual_language_ocr(tmp_path)
                            
                            if results:
                                # Тэмдэглэл нэмэх
                                notes = st.text_input(f"Тэмдэглэл ({uploaded_file.name}):", key=f"notes_{uploaded_file.name}")
                                
                                # Үр дүн харуулах
                                st.success(f"✅ {len(results)} текст олдлоо!")
                                
                                # Танигдсан текстүүд
                                detected_texts = []
                                for i, (bbox, text, confidence) in enumerate(results):
                                    cleaned_text = clean_ocr_text(text)
                                    detected_texts.append({
                                         "№": i+1,
                                         "Текст": cleaned_text,
                                         "Итгэлцүүр": f"{confidence:.3f}"
                                 })

                                
                                df_results = pd.DataFrame(detected_texts)
                                st.dataframe(df_results, use_container_width=True)
                                
                                # Box-тай зураг харуулах
                                with col2:
                                    boxed_image = draw_boxes_on_image(image, results)
                                    st.image(boxed_image, caption="Танигдсан хэсгүүд", use_column_width=True)
                                
                                # Database-д хадгалах товч
                                if st.button(f"💾 Database-д хадгалах", key=f"save_{uploaded_file.name}"):
                                    result_id = db.save_ocr_result(tmp_path, results, notes)
                                    if result_id:
                                        st.success(f"✅ Амжилттай хадгалагдлаа! ID: {result_id}")
                                        st.rerun()
                                    else:
                                        st.error("❌ Хадгалахад алдаа гарлаа!")
                            else:
                                st.warning("⚠️ Текст олдсонгүй!")
                        
                        except Exception as e:
                            st.error(f"❌ OCR алдаа гарлаа: {e}")
                        
                        finally:
                            # Temporary файл устгах
                            if os.path.exists(tmp_path):
                                os.unlink(tmp_path)
                
                except Exception as e:
                    st.error(f"❌ Зураг боловсруулахад алдаа: {e}")
                
                st.markdown("---")
    
    elif page == "📊 Database хайлт":
        st.header("📊 Database хайлт ба харуулалт")
        
        if record_count == 0:
            st.warning("⚠️ Database хоосон байна! Эхлээд OCR процесс ажиллуулна уу.")
            return
        
        # Хайлтын сонголт
        search_option = st.radio("Хайлтын төрөл:", ["Бүгдийг харах", "Текстээр хайх"])
        
        if search_option == "Текстээр хайх":
            search_text = st.text_input("🔍 Хайх текст:")
            if search_text:
                df = db.search_by_text(search_text)
                if not df.empty:
                    st.success(f"✅ {len(df)} үр дүн олдлоо!")
                else:
                    st.warning("⚠️ Хайлтад тохирох үр дүн олдсонгүй!")
            else:
                df = pd.DataFrame()
        else:
            df = db.get_all_results()
        
        # Үр дүн харуулах
        if not df.empty:
            st.subheader(f"📋 Үр дүн ({len(df)} бичлэг)")
            
            # DataFrame харуулах
            st.dataframe(df, use_container_width=True)
            
            # Дэлгэрэнгүй харах
            if len(df) > 0:
                selected_id = st.selectbox("Дэлгэрэнгүй харах:", df['id'].tolist())
                
                if st.button("👁️ Дэлгэрэнгүй харах"):
                    detailed_result = db.get_detailed_result(selected_id)
                    if detailed_result:
                        st.subheader("📄 Дэлгэрэнгүй мэдээлэл")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write(f"**ID:** {detailed_result[0]}")
                            st.write(f"**Зургийн нэр:** {detailed_result[2]}")
                            st.write(f"**Огноо:** {detailed_result[6]}")
                            st.write(f"**Тэмдэглэл:** {detailed_result[7] or 'Байхгүй'}")
                        
                        with col2:
                            # Confidence scores-ийг харуулах
                            try:
                                scores = json.loads(detailed_result[4])
                                if scores:
                                    avg_confidence = sum(float(s) for s in scores) / len(scores)
                                    st.write(f"**Дундаж итгэлцүүр:** {avg_confidence:.3f}")
                                    st.write(f"**Олдсон блок:** {len(scores)}")
                            except:
                                pass
                        
                        st.text_area("📝 Танигдсан бүх текст:", detailed_result[3], height=200)
    
    elif page == "🤖 Q&A систем":
        st.header("🤖 Монгол OCR Q&A Систем (MongPT)")
        
        if not QA_AVAILABLE:
            st.error("❌ Q&A систем ажиллахгүй байна!")
            st.info("Дараах packages суулгана уу:")
            st.code("pip install langchain langchain-community faiss-cpu sentence-transformers")
            return
        
        if record_count == 0:
            st.warning("⚠️ Database хоосон байна! Эхлээд OCR процесс ажиллуулна уу.")
            return
        
        # Vector database ачаалах функц
        @st.cache_resource
        def load_vector_db_from_sqlite():
            try:
                # Database-аас бүх өгөгдөл авах
                json_data = db.export_to_json()
                if not json_data or not json_data.get("ocr_results"):
                    return None
                
                docs = []
                for item in json_data["ocr_results"]:
                    text = item["detected_text"]
                    if not text or not text.strip():
                        continue
                    
                    metadata = {
                        "image": item["image_name"],
                        "confidence": item["statistics"]["average_confidence"],
                        "processed_date": item["processed_date"],
                        "id": item["id"]
                    }
                    docs.append(Document(page_content=text, metadata=metadata))
                
                if not docs:
                    return None
                
                st.info(f"📚 {len(docs)} баримт vector database-д ачаалж байна...")
                
                # Embeddings үүсгэх
                embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2"
                )
                
                # FAISS vector store үүсгэх
                vectorstore = FAISS.from_documents(docs, embeddings)
                retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
                
                st.success("✅ Vector database бэлэн боллоо!")
                return retriever
            
            except Exception as e:
                st.error(f"Vector database ачаалахад алдаа: {e}")
                return None
        
        # LLM ачаалах
        @st.cache_resource
        def load_llm():
            try:
                llm = Ollama(model="mongpt")
                return llm
            except Exception as e:
                st.error(f"LLM ачаалахад алдаа: {e}")
                st.info("Ollama болон MongPT model суулгасан эсэхээ шалгана уу.")
                return None
        
        # Системийг ачаалах
        with st.spinner("🤖 Q&A систем ачаалж байна..."):
            retriever = load_vector_db_from_sqlite()
            llm = load_llm()
        
        if not retriever:
            st.error("❌ Vector database ачаалагдсангүй!")
            return
        
        if not llm:
            st.error("❌ Language model ачаалагдсангүй!")
            return
        
        st.success("🚀 Q&A систем бэлэн!")
        
        # Асуулт оруулах хэсэг
        st.subheader("💬 Асуулт асуух")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            query = st.text_input(
                "📝 Асуултаа оруулна уу:",
                placeholder="Жишээ: Энэ баримтад ямар төрлийн албан бичиг байна вэ?",
                key="qa_query"
            )
        
        with col2:
            search_mode = st.selectbox(
                "Хайлтын төрөл:",
                ["Ерөнхий", "Нарийвчилсан"]
            )
        
        if query:
            with st.spinner("🔍 Хайж байна..."):
                try:
                    # Relevant documents олох
                    if search_mode == "Нарийвчилсан":
                        docs = retriever.get_relevant_documents(query, k=5)
                    else:
                        docs = retriever.get_relevant_documents(query, k=3)
                    
                    if not docs:
                        st.warning("⚠️ Холбогдох мэдээлэл олдсонгүй!")
                        return
                    
                    # Context бэлтгэх
                    context_parts = []
                    for i, doc in enumerate(docs):
                        context_parts.append(f"Баримт {i+1} ({doc.metadata.get('image', 'Unknown')}):\n{doc.page_content}")
                    
                    context = "\n\n".join(context_parts)
                    
                    # Prompt бэлтгэх
                    prompt = f"""Та дараах баримтын текстүүд дээр үндэслэн Монгол хэл дээр хариулна уу:

--- Баримтын контекст ---
{context}

--- Асуулт ---
{query}

--- Заавар ---
Хариултаа Монгол хэл дээр өгнө үү. Баримтад байгаа мэдээлэл дээр л үндэслэнэ үү. Хэрэв хариулт олдохгүй бол тэгж хэлнэ үү.

--- Хариулт ---"""
                    
                    # LLM-д асуух
                    with st.spinner("🧠 Хариулт үүсгэж байна..."):
                        response = llm.invoke(prompt)
                    
                    # Хариултыг харуулах
                    st.success("🤖 Хариулт:")
                    st.write(response)
                    
                    # Ашигласан баримтуудыг харуулах
                    with st.expander("📚 Ашигласан баримтууд"):
                        for i, doc in enumerate(docs):
                            st.write(f"**Баримт {i+1}:** {doc.metadata.get('image', 'Unknown')}")
                            st.write(f"**Итгэлцүүр:** {doc.metadata.get('confidence', 'N/A'):.3f}")
                            st.text_area(f"Текст {i+1}:", doc.page_content, height=100, key=f"doc_{i}")
                            st.markdown("---")
                
                except Exception as e:
                    st.error(f"❌ Q&A системд алдаа гарлаа: {e}")
        
        # Системийн тохиргоо
        with st.expander("⚙️ Систем тохиргоо"):
            st.info("**Ашигласан технологи:**")
            st.write("- **Embeddings:** sentence-transformers/all-MiniLM-L6-v2")
            st.write("- **Vector Store:** FAISS")
            st.write("- **LLM:** Ollama MongPT")
            st.write("- **Database:** SQLite + OCR Results")
    
    elif page == "📁 Экспорт/Удирдлага":
        st.header("📁 Экспорт ба Database удирдлага")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📤 JSON Экспорт")
            
            if record_count > 0:
                if st.button("📁 JSON файл татаж авах"):
                    json_data = db.export_to_json()
                    if json_data:
                        json_string = json.dumps(json_data, ensure_ascii=False, indent=2)
                        
                        st.download_button(
                            label="💾 JSON файл татах",
                            data=json_string,
                            file_name=f"ocr_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                            mime="application/json"
                        )
                        
                        st.success("✅ JSON файл бэлэн!")
                        
                        # Preview харуулах
                        with st.expander("👁️ JSON өгөгдлийн preview"):
                            st.json(json_data)
                    else:
                        st.error("❌ JSON экспорт амжилтгүй!")
            else:
                st.warning("⚠️ Экспортлох өгөгдөл байхгүй!")
        
        with col2:
            st.subheader("🗑️ Database удирдлага")
            
            st.warning("⚠️ Анхаарах: Энэ үйлдэл буцаах боломжгүй!")
            
            # Checkbox болон button-г тусад нь хийх (session state ашиглан)
            if 'clear_confirmed' not in st.session_state:
                st.session_state.clear_confirmed = False
            
            confirm_clear = st.checkbox("Тийм, би мэдэж байна", key="confirm_clear")
            
            if st.button("🗑️ Database бүгдийг устгах", type="secondary", disabled=not confirm_clear):
                if db.clear_database():
                    st.success("✅ Database цэвэрлэгдлээ!")
                    # Checkbox-г reset хийх
                    st.session_state.clear_confirmed = False
                    st.rerun()
                else:
                    st.error("❌ Database цэвэрлэхэд алдаа гарлаа!")
        
        # Статистик
        if record_count > 0:
            st.subheader("📊 Статистик")
            df = db.get_all_results()
            
            if not df.empty:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Нийт бичлэг", record_count)
                with col2:
                    total_chars = df['detected_text'].fillna('').str.len().sum()
                    st.metric("Нийт тэмдэгт", total_chars)
                with col3:
                    avg_chars = total_chars / record_count if record_count > 0 else 0
                    st.metric("Дундаж тэмдэгт", f"{avg_chars:.0f}")
                
                # Огнооны график
                try:
                    df['processed_date'] = pd.to_datetime(df['processed_date'])
                    daily_counts = df.groupby(df['processed_date'].dt.date).size()
                    
                    if len(daily_counts) > 1:
                        st.subheader("📈 Өдрийн боловсруулалт")
                        st.line_chart(daily_counts)
                except:
                    pass

if __name__ == "__main__":
    main()