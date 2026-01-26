"""
KOKOA Memory Viewer
===================
벡터스토어에 저장된 문서들을 웹 UI로 열람

Usage: python view_memory.py
"""

import os
import webbrowser
import json
from http.server import HTTPServer, SimpleHTTPRequestHandler
import threading

# Paths (tools/ is inside KOKOA/)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Go up to KOKOA/
PDF_STORE = os.path.join(BASE_DIR, "initial_state", "pdf_store")


def get_collection_documents(collection_path: str) -> list:
    """Chroma collection에서 문서들을 가져옴"""
    try:
        import chromadb
        client = chromadb.PersistentClient(path=collection_path)
        
        all_docs = []
        # Get all collection names first
        collection_names = [col.name for col in client.list_collections()]
        
        for col_name in collection_names:
            try:
                col = client.get_collection(col_name)
                results = col.get(include=["documents", "metadatas"])
                docs = results.get("documents", [])
                metas = results.get("metadatas", [])
                
                for doc, meta in zip(docs, metas):
                    all_docs.append({
                        "collection": col_name,
                        "content": doc if doc else "No content",
                        "metadata": meta
                    })
            except Exception as e:
                all_docs.append({"error": f"Collection {col_name}: {e}"})
        
        return all_docs
    except Exception as e:
        return [{"error": str(e)}]


def get_all_data() -> dict:
    """모든 컬렉션 데이터 수집 (Simple is Best)"""
    data = {
        "pdfs": [],
        "technical_reports": []
    }
    
    # PDF RAG
    if os.path.exists(PDF_STORE):
        data["pdfs"] = get_collection_documents(PDF_STORE)
        print(f"  - PDFs: {len(data['pdfs'])} docs")
    
    # Technical reports (directly in initial_state/)
    reports_path = os.path.join(BASE_DIR, "initial_state", "technical_reports")
    if os.path.exists(reports_path):
        data["technical_reports"] = get_collection_documents(reports_path)
        print(f"  - Reports: {len(data['technical_reports'])} docs")
    
    return data


def generate_html(data: dict) -> str:
    """HTML 페이지 생성"""
    
    def render_items(items: list) -> str:
        if not items:
            return "<p class='empty'>저장된 문서가 없습니다.</p>"
        
        html = ""
        for i, item in enumerate(items):
            if "error" in item:
                html += f"<div class='item error'>Error: {item['error']}</div>"
            else:
                content = item.get("content", "")[:500]
                meta = json.dumps(item.get("metadata", {}), indent=2, ensure_ascii=False)
                html += f"""
                <div class='item'>
                    <div class='item-header'>Document #{i+1}</div>
                    <div class='content'>{content}...</div>
                    <details>
                        <summary>Metadata</summary>
                        <pre>{meta}</pre>
                    </details>
                </div>
                """
        return html
    
    return f"""
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <title>KOKOA Memory Viewer</title>
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{ 
            font-family: 'Segoe UI', sans-serif; 
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            min-height: 100vh;
            color: #e0e0e0;
            padding: 20px;
        }}
        h1 {{ 
            text-align: center; 
            margin: 20px 0 30px;
            color: #00d9ff;
            text-shadow: 0 0 20px rgba(0, 217, 255, 0.3);
        }}
        .tabs {{
            display: flex;
            justify-content: center;
            gap: 10px;
            margin-bottom: 20px;
            flex-wrap: wrap;
        }}
        .tab-btn {{
            padding: 12px 24px;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            font-size: 16px;
            font-weight: 600;
            transition: all 0.3s;
            background: linear-gradient(145deg, #2d2d44, #1f1f35);
            color: #aaa;
            box-shadow: 0 4px 15px rgba(0,0,0,0.3);
        }}
        .tab-btn:hover {{
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(0,217,255,0.2);
        }}
        .tab-btn.active {{
            background: linear-gradient(145deg, #00d9ff, #0099cc);
            color: #000;
        }}
        .tab-content {{
            display: none;
            max-width: 1000px;
            margin: 0 auto;
        }}
        .tab-content.active {{ display: block; }}
        .item {{
            background: rgba(255,255,255,0.05);
            border: 1px solid rgba(255,255,255,0.1);
            border-radius: 12px;
            padding: 16px;
            margin-bottom: 15px;
            backdrop-filter: blur(10px);
        }}
        .item-header {{
            font-weight: bold;
            color: #00d9ff;
            margin-bottom: 10px;
        }}
        .content {{
            font-size: 14px;
            line-height: 1.6;
            white-space: pre-wrap;
            color: #ccc;
        }}
        details {{
            margin-top: 10px;
            font-size: 12px;
        }}
        summary {{
            cursor: pointer;
            color: #888;
        }}
        pre {{
            background: rgba(0,0,0,0.3);
            padding: 10px;
            border-radius: 6px;
            overflow-x: auto;
            margin-top: 8px;
        }}
        .empty {{
            text-align: center;
            color: #666;
            padding: 40px;
        }}
        .error {{ color: #ff6b6b; }}
        .count {{
            font-size: 12px;
            color: #888;
            margin-left: 5px;
        }}
    </style>
</head>
<body>
    <h1>🧠 KOKOA Memory Viewer</h1>
    
    <div class="tabs">
        <button class="tab-btn active" onclick="showTab('pdfs')">📄 PDFs <span class="count">({len(data['pdfs'])})</span></button>
        <button class="tab-btn" onclick="showTab('technical_reports')">📊 Reports <span class="count">({len(data['technical_reports'])})</span></button>
    </div>
    
    <div id="pdfs" class="tab-content active">{render_items(data['pdfs'])}</div>
    <div id="technical_reports" class="tab-content">{render_items(data['technical_reports'])}</div>
    
    <script>
        function showTab(tabId) {{
            document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
            document.querySelectorAll('.tab-btn').forEach(el => el.classList.remove('active'));
            document.getElementById(tabId).classList.add('active');
            event.target.classList.add('active');
        }}
    </script>
</body>
</html>
"""


def main():
    print("[KOKOA Memory Viewer]")
    print("=" * 40)
    
    # 데이터 수집
    print("Loading data from vectorstores...")
    data = get_all_data()
    
    # HTML 생성 및 저장
    html_path = os.path.join(BASE_DIR, "memory_view.html")
    html_content = generate_html(data)
    
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    
    print(f"Generated: {html_path}")
    
    # 브라우저로 열기
    webbrowser.open(f"file://{html_path}")
    print("[OK] Opened in browser!")


if __name__ == "__main__":
    main()
