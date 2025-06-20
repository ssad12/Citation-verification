import os
import json
import argparse
import re
from paper_citation_verifier import PaperCitationVerifier, call_llm_api # Import call_llm_api
from citation_utils import extract_citations_from_text
import fitz 


def read_pdf_text_fitz(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text() + "\n"
        doc.close()
        return text
    except Exception as e:
        print(f"PDF读取失败: {e}")
        return ""

def read_text_file(file_path):
    try:
        with open(file_path, encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        print(f"文件读取失败: {e}")
        return ""

def verify_article(verifier, article_path, output_path, sim_thresh=0.6, keyword_overlap_thresh=0.3):
    if article_path.lower().endswith(".pdf"):
        text = read_pdf_text_fitz(article_path)
        # For PDF, we also need to extract its own references section
        _, _, _, article_references = verifier.extract_title_and_abstract_authors_references(text, article_path)
        print(f"Extracted {len(article_references)} references from article bibliography.")
    else:
        text = read_text_file(article_path)
        article_references = [] # No bibliography for text files, or need to parse it if available

    if not text:
        print(f"无法读取文章内容: {article_path}")
        return

    citations = extract_citations_from_text(text)
    print(f"检测到文中引用 {len(citations)} 条")

    results = []
    for idx, cit in enumerate(citations):
        print(f"验证引用 {idx+1}/{len(citations)}: {cit}")
        context_fragments = []
        for m in re.finditer(re.escape(cit), text):
            # Attempt to get the whole sentence containing the citation for better context
            sentence_start = text.rfind('.', 0, m.start()) + 1
            if sentence_start == 0: # If no dot before, go to start of text
                sentence_start = text.rfind('\n\n', 0, m.start()) + 2 # Or start of paragraph
            
            sentence_end = text.find('.', m.end()) + 1
            if sentence_end == 0: # If no dot after, go to end of text
                sentence_end = text.find('\n\n', m.end()) # Or end of paragraph
                if sentence_end == -1: sentence_end = len(text)


            fragment = text[sentence_start:sentence_end].strip()
            # If the sentence is too short or clearly broken, take a wider window
            if len(fragment) < 30 or '\n' in fragment:
                start_idx = max(0, m.start() - 200) 
                end_idx = min(len(text), m.end() + 200) 
                fragment = text[start_idx:end_idx].strip()
            
            fragment = re.sub(r'\n\s*\n+', '\n\n', fragment).strip() 
            context_fragments.append(fragment)

        if not context_fragments:
            results.append({"citation_marker": cit, "status": "skipped", "reason": "未在文中找到引用标记或有效上下文"})
            continue
        
        merged_context = "\n\n".join(context_fragments)
        
        # Pass article_references to resolve_citation_marker_to_title
        result = verifier.verify_citation(
            citation_marker=cit, 
            cited_context=merged_context, 
            article_references=article_references, # NEW: pass article's bibliography
            sim_thresh=sim_thresh, 
            keyword_overlap_thresh=keyword_overlap_thresh
        )
        result["citation_marker"] = cit 
        results.append(result)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"引用验证完成，结果写入：{output_path}")

def query_knowledge_base_mode(verifier, query_text, output_path, n_results=5, sim_thresh=0.6):
    print(f"🔍 正在查询知识库: '{query_text}'")
    results = verifier.query_knowledge_base(query_text, n_results=n_results, sim_thresh=sim_thresh)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"✅ 查询完成，结果写入：{output_path}")


def add_papers_from_dir(verifier, pdf_dir, recursive=False):
    count_added, count_failed = 0, 0
    if not os.path.exists(pdf_dir):
        print(f"错误: 文件夹 '{pdf_dir}' 不存在。")
        return

    for root, _, files in os.walk(pdf_dir):
        for file in files:
            if file.lower().endswith(".pdf"):
                full_path = os.path.join(root, file)
                print(f"--- 尝试添加: {full_path} ---")
                if verifier.add_paper(full_path):
                    count_added += 1
                else:
                    count_failed += 1
        if not recursive:
            break 
    print(f"添加完成：成功 {count_added} 篇，失败 {count_failed} 篇")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="📘 文献引用验证与知识库查询系统")
    parser.add_argument("mode", choices=["add", "verify_citations", "query_kb", "update"], help="选择运行模式")
    parser.add_argument("--pdf_dir", help="PDF 文件夹路径 (用于 add 模式)")
    parser.add_argument("--article", help="待验证的文章路径 (PDF 或 TXT，用于 verify_citations 模式)")
    parser.add_argument("--query_text", help="查询知识库的文本 (用于 query_kb 模式)")
    parser.add_argument("--output", default="results.json", help="输出 JSON 文件路径 (验证或查询结果)")
    parser.add_argument("--chroma_dir", default="./chroma_db", help="RAG 存储路径")
    parser.add_argument("--embedding_model", default="./local_models/all-MiniLM-L6-v2", help="本地嵌入模型路径")
    parser.add_argument("--recursive", action="store_true", help="是否递归遍历 pdf_dir (用于 add 模式)")
    parser.add_argument("--sim_thresh", type=float, default=0.05, help="语义相似度阈值 (用于 verify_citations 和 query_kb 模式)")
    parser.add_argument("--keyword_overlap_thresh", type=float, default=0.3, help="关键词重合度阈值 (用于 verify_citations 模式)") 
    args = parser.parse_args()

    verifier = PaperCitationVerifier(chroma_dir=args.chroma_dir, embedding_model=args.embedding_model)

    if args.mode == "add":
        if not args.pdf_dir:
            print("❗ 'add' 模式需要提供 --pdf_dir 参数。")
        else:
            add_papers_from_dir(verifier, args.pdf_dir, recursive=args.recursive)
            verifier.update_rag_collection()

    elif args.mode == "verify_citations":
        if not args.article:
            print("❗ 'verify_citations' 模式需要提供 --article 参数。")
        else:
            verify_article(verifier, args.article, args.output, sim_thresh=args.sim_thresh, keyword_overlap_thresh=args.keyword_overlap_thresh)

    elif args.mode == "query_kb":
        if not args.query_text:
            print("❗ 'query_kb' 模式需要提供 --query_text 参数。")
        else:
            query_knowledge_base_mode(verifier, args.query_text, args.output, sim_thresh=args.sim_thresh)

    elif args.mode == "update":
        verifier.update_rag_collection()
        print("📚 RAG 知识库已手动更新。")