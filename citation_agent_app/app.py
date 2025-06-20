from flask import Flask, request, jsonify, render_template, send_file
from flask_cors import CORS
import os
import json
import tempfile
from werkzeug.utils import secure_filename
from paper_citation_verifier import PaperCitationVerifier
from citation_utils import extract_citations_from_text
import fitz
import threading
import time
from datetime import datetime
from config import WebConfig, ModelConfig, create_directories, validate_config

app = Flask(__name__)
CORS(app)

# 从配置文件加载配置
app.config['MAX_CONTENT_LENGTH'] = WebConfig.MAX_CONTENT_LENGTH
app.config['UPLOAD_FOLDER'] = WebConfig.UPLOAD_FOLDER
app.config['ALLOWED_EXTENSIONS'] = WebConfig.ALLOWED_EXTENSIONS
app.config['SECRET_KEY'] = WebConfig.SECRET_KEY

# 创建必要的目录
create_directories()

# 全局变量
verifier = None
task_status = {}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def read_pdf_text_fitz(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text() + "\n"
        doc.close()
        return text
    except Exception as e:
        return f"PDF读取失败: {e}"

def read_text_file(file_path):
    try:
        with open(file_path, encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"文件读取失败: {e}"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/init', methods=['POST'])
def init_verifier():
    """初始化验证器"""
    global verifier
    try:
        data = request.get_json()
        chroma_dir = data.get('chroma_dir', './chroma_db')
        embedding_model = data.get('embedding_model', './local_models/all-MiniLM-L6-v2')
        
        verifier = PaperCitationVerifier(chroma_dir=chroma_dir, embedding_model=embedding_model)
        return jsonify({'success': True, 'message': '系统初始化成功'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/upload_papers', methods=['POST'])
def upload_papers():
    """上传并添加论文到知识库"""
    if not verifier:
        return jsonify({'success': False, 'error': '请先初始化系统'}), 400
    
    if 'files' not in request.files:
        return jsonify({'success': False, 'error': '没有文件被上传'}), 400
    
    files = request.files.getlist('files')
    results = []
    
    for file in files:
        if file.filename == '':
            continue
            
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            try:
                success = verifier.add_paper(filepath)
                results.append({
                    'filename': filename,
                    'success': success,
                    'message': '添加成功' if success else '添加失败'
                })
            except Exception as e:
                results.append({
                    'filename': filename,
                    'success': False,
                    'message': f'处理失败: {str(e)}'
                })
            finally:
                # 清理临时文件
                if os.path.exists(filepath):
                    os.remove(filepath)
    
    # 更新RAG集合
    try:
        verifier.update_rag_collection()
    except Exception as e:
        return jsonify({'success': False, 'error': f'更新知识库失败: {str(e)}'}), 500
    
    return jsonify({'success': True, 'results': results})

@app.route('/api/verify_citations', methods=['POST'])
def verify_citations():
    """验证文章引用"""
    if not verifier:
        return jsonify({'success': False, 'error': '请先初始化系统'}), 400
    
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': '没有文件被上传'}), 400
    
    file = request.files['file']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({'success': False, 'error': '无效的文件'}), 400
    
    # 获取参数
    sim_thresh = float(request.form.get('sim_thresh', 0.05))
    keyword_overlap_thresh = float(request.form.get('keyword_overlap_thresh', 0.3))
    verification_method = request.form.get('verification_method', 'api')  # 默认为api
    
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    try:
        # 读取文件内容
        if filepath.lower().endswith('.pdf'):
            text = read_pdf_text_fitz(filepath)
            _, _, _, article_references = verifier.extract_title_and_abstract_authors_references(text, filepath)
        else:
            text = read_text_file(filepath)
            article_references = []
        
        if not text or text.startswith('PDF读取失败') or text.startswith('文件读取失败'):
            return jsonify({'success': False, 'error': text}), 400
        
        # 提取引用
        citations = extract_citations_from_text(text)
        
        # 验证每个引用
        results = []
        for idx, cit in enumerate(citations):
            # 提取上下文
            context_fragments = []
            import re
            for m in re.finditer(re.escape(cit), text):
                sentence_start = text.rfind('.', 0, m.start()) + 1
                if sentence_start == 0:
                    sentence_start = text.rfind('\n\n', 0, m.start()) + 2
                
                sentence_end = text.find('.', m.end()) + 1
                if sentence_end == 0:
                    sentence_end = text.find('\n\n', m.end())
                    if sentence_end == -1:
                        sentence_end = len(text)
                
                fragment = text[sentence_start:sentence_end].strip()
                if len(fragment) < 30 or '\n' in fragment:
                    start_idx = max(0, m.start() - 200)
                    end_idx = min(len(text), m.end() + 200)
                    fragment = text[start_idx:end_idx].strip()
                
                fragment = re.sub(r'\n\s*\n+', '\n\n', fragment).strip()
                context_fragments.append(fragment)
            
            if not context_fragments:
                results.append({
                    "citation_marker": cit,
                    "status": "skipped",
                    "reason": "未在文中找到引用标记或有效上下文"
                })
                continue
            
            merged_context = "\n\n".join(context_fragments)
            
            # 根据选择的验证方式进行验证
            if verification_method == 'api':
                raw_result = verifier.verify_citation(
                    citation_marker=cit,
                    cited_context=merged_context,
                    article_references=article_references,
                    sim_thresh=sim_thresh,
                    keyword_overlap_thresh=keyword_overlap_thresh,
                    change=True
                )
            else:  # local
                raw_result = verifier.verify_citation(
                    citation_marker=cit,
                    cited_context=merged_context,
                    article_references=article_references,
                    sim_thresh=sim_thresh,
                    keyword_overlap_thresh=keyword_overlap_thresh,
                    change=False
                )
            
            # 转换为前端期望的格式
            evidence = raw_result.get("evidence", [])
            max_similarity = max([e.get("semantic_similarity", 0) for e in evidence], default=0)
            max_keyword_overlap = max([e.get("keyword_overlap", 0) for e in evidence], default=0)
            
            formatted_result = {
                "citation_marker": cit,
                "status": "verified" if raw_result.get("is_valid") else "not_found",
                "reason": "; ".join(raw_result.get("issues", [])),
                "matched_paper": raw_result.get("resolved_title"),
                "similarity_score": max_similarity,
                "keyword_overlap": max_keyword_overlap,
                "method": verification_method  # 返回使用的验证方式
            }
            results.append(formatted_result)
        
        return jsonify({
            'success': True,
            'total_citations': len(citations),
            'results': results
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500
    finally:
        # 清理临时文件
        if os.path.exists(filepath):
            os.remove(filepath)

@app.route('/api/query_knowledge_base', methods=['POST'])
def query_knowledge_base():
    """查询知识库"""
    if not verifier:
        return jsonify({'success': False, 'error': '请先初始化系统'}), 400
    
    try:
        data = request.get_json()
        query_text = data.get('query_text', '')
        n_results = data.get('n_results', 5)
        sim_thresh = data.get('sim_thresh', 0.6)
        
        if not query_text:
            return jsonify({'success': False, 'error': '查询文本不能为空'}), 400
        
        results = verifier.query_knowledge_base(query_text, n_results=n_results, sim_thresh=sim_thresh)
        
        return jsonify({
            'success': True,
            'query': query_text,
            'results': results
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/update_knowledge_base', methods=['POST'])
def update_knowledge_base():
    """更新知识库"""
    if not verifier:
        return jsonify({'success': False, 'error': '请先初始化系统'}), 400
    
    try:
        verifier.update_rag_collection()
        return jsonify({'success': True, 'message': 'RAG 知识库已更新'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/status')
def get_status():
    """获取系统状态"""
    return jsonify({
        'initialized': verifier is not None,
        'timestamp': datetime.now().isoformat()
    })

if __name__ == '__main__':
    print("启动文献引用验证系统...")
    
    # 验证配置
    if not validate_config():
        print("配置验证失败，请检查config.py文件")
        exit(1)
    
    print(f"访问 http://localhost:{WebConfig.PORT} 使用Web界面")
    app.run(debug=WebConfig.DEBUG, host=WebConfig.HOST, port=WebConfig.PORT)