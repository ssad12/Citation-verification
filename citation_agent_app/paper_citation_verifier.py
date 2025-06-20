import os
import re
import pickle
import torch
from sentence_transformers import SentenceTransformer
from thefuzz import fuzz
import fitz  # PyMuPDF
import chromadb
from collections import Counter
import arxiv # For arXiv download
from typing import List, Tuple, Optional, Dict
from transformers import AutoTokenizer, AutoModelForCausalLM
# --- DeepSeek API Setup ---
from openai import OpenAI

client = OpenAI(
    base_url='',
    api_key='', 
)
# --- End DeepSeek API Setup ---


def call_local_api(prompt_llm_check, max_tokens=150, temperature=0.1, model_path="D:\agent\home\wuxiao\test\citation_agent_app\fine_tuned_qwen2.5_3B"):
    """
    调用本地大模型的函数
    
    参数:
        prompt_llm_check: 输入的提示文本
        max_tokens: 生成的最大token数
        temperature: 控制生成随机性的温度参数
        model_path: 本地模型路径
    
    返回:
        模型的响应文本
    """
    # 初始化tokenizer和model
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, 
                                               device_map="auto", 
                                               torch_dtype=torch.float16)
    model.eval()
    
    # 准备输入
    inputs = tokenizer(prompt_llm_check, return_tensors="pt", max_length=2048, truncation=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # 生成响应
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # 解码响应
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # 只返回assistant生成的部分（去掉输入的prompt）
    if "<|im_start|>assistant" in response:
        response = response.split("<|im_start|>assistant")[-1].strip()
    
    return response

def call_llm_api(prompt, model_name='deepseek-ai/DeepSeek-R1-0528', max_tokens=200, temperature=0.1):
    """
    Calls the DeepSeek API to get a response.
    Includes a retry mechanism for robustness.
    """
    try:
        messages = [
            {'role': 'system', 'content': 'You are a precise academic assistant. Respond concisely and accurately.'},
            {'role': 'user', 'content': prompt}
        ]

        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            stream=False
        )
        
        full_response_content = response.choices[0].message.content
        if not full_response_content: 
             full_response_content = response.choices[0].message.reasoning_content

        return full_response_content.strip()

    except Exception as e:
        print(f"Error calling DeepSeek LLM API: {e}")
        return "LLM API Error: " + str(e)


class PaperCitationVerifier:
    def __init__(self, chroma_dir="./chroma_db", embedding_model="./local_models/all-MiniLM-L6-v2",
                 device="cuda:0" if torch.cuda.is_available() else "cpu"):
        os.makedirs(chroma_dir, exist_ok=True)
        self.chroma_dir = chroma_dir
        self.metadata_path = os.path.join(chroma_dir, "metadata.pkl")
        self.device = device

        self.stored_titles = []
        self.paper_summaries = {}
        self.paper_paragraphs = {}  # title -> list of paras
        self.paper_references = {} # title -> list of reference strings (from paper's bibliography)
        self.init_embedding_model(embedding_model)

        self.client = chromadb.PersistentClient(path=self.chroma_dir)
        self.collection = self.client.get_or_create_collection("papers")
        self.load_metadata()

    def init_embedding_model(self, model_path):
        self.embedding_model = SentenceTransformer(model_path)
        self.embedding_model.to(self.device)

    def load_metadata(self):
        if os.path.exists(self.metadata_path):
            with open(self.metadata_path, "rb") as f:
                meta = pickle.load(f)
                self.stored_titles = meta.get("titles", [])
                self.paper_summaries = meta.get("summaries", {})
                self.paper_paragraphs = meta.get("paragraphs", {})
                self.paper_references = meta.get("references", {})

    def save_metadata(self):
        with open(self.metadata_path, "wb") as f:
            pickle.dump({
                "titles": self.stored_titles,
                "summaries": self.paper_summaries,
                "paragraphs": self.paper_paragraphs,
                "references": self.paper_references
            }, f)

    def parse_pdf(self, pdf_path):
        try:
            doc = fitz.open(pdf_path)
            full_text = ""
            for page in doc:
                full_text += page.get_text() + "\n"
            doc.close()
            return full_text
        except Exception as e:
            print(f"❌ PyMuPDF failed to read PDF {pdf_path}: {e}")
            return ""

    def extract_title_and_abstract_authors_references(self, text, pdf_path):
        title = None
        abstract = None
        authors = []
        references = [] 
        
        try:
            doc = fitz.open(pdf_path)
            title = None

            if doc.metadata and doc.metadata.get("title"):
                meta_title = doc.metadata["title"].strip()
                if meta_title and 5 < len(meta_title) < 200:
                    title = meta_title

            if title is None and doc.page_count > 0:
                first_page = doc[0]
                page_dict = first_page.get_text("dict")
                blocks = page_dict.get("blocks", [])

                potential_titles_with_info = [] 
                for b in blocks:
                    if b.get("type") == 0: 
                        for line in b.get("lines", []):
                            line_text = ""
                            max_line_size = 0
                            for span in line.get("spans", []):
                                line_text += span.get("text", "")
                                if span.get("size", 0) > max_line_size:
                                    max_line_size = span.get("size", 0)
                            
                            cleaned_line_text = line_text.strip()
                        
                            if (cleaned_line_text and 
                                    10 < len(cleaned_line_text) < 200 and
                                    not re.search(r'^\d+$', cleaned_line_text) and 
                                    not re.search(r'\b(abstract|introduction|doi|keywords|author|page|table of contents|contents)\b', cleaned_line_text.lower()) and
                                    not re.match(r'arxiv:\d{4}\.\d{4,5}(v\d+)?', cleaned_line_text.lower()) and
                                    not re.match(r'\[.+\]\s+\d{1,2}\s+\w+\s+\d{4}', cleaned_line_text.lower()) and
                                    not re.search(r'\barxiv\b', cleaned_line_text.lower())):
                                    
                                    potential_titles_with_info.append((cleaned_line_text, max_line_size, line['bbox'][1]))


                potential_titles_with_info.sort(key=lambda x: (-x[1], x[2])) 
                
                if potential_titles_with_info:
                    best_candidate_text = potential_titles_with_info[0][0]
                    best_candidate_size = potential_titles_with_info[0][1]
                    best_candidate_y = potential_titles_with_info[0][2]

                    current_title_candidates = [(best_candidate_y, best_candidate_text)] 
                    for i in range(1, len(potential_titles_with_info)):
                        next_line_text, next_line_size, next_line_y = potential_titles_with_info[i]
                        if (0 < (next_line_y - best_candidate_y) < best_candidate_size * 1.5) and \
                        (abs(best_candidate_size - next_line_size) <= 5):
                            
                            current_title_candidates.append((next_line_y, next_line_text))
                            best_candidate_y = next_line_y 
                        else:
                            break
                    
                    current_title_candidates.sort(key=lambda x: x[0])
                    title_lines = [text for y, text in current_title_candidates]

                    title = " ".join(title_lines).strip()
                    title = re.sub(r'^\s*[\d\.§•]+\s*|\s*[\d\.§•]+\s*$', '', title).strip()
                    
                    if len(title) < 15: 
                        title = None 

                if not title and len(doc[0].get_text().strip().split('\n')) > 0:
                    first_line_raw = doc[0].get_text().strip().split('\n')[0].strip()
                    if 10 < len(first_line_raw) < 200 and not any(x in first_line_raw.lower() for x in ["abstract", "doi", "introduction"]):
                        title = first_line_raw



            lower_text = text.lower()
            abstract_start_match = re.search(r'\b(abstract|摘要)\b', lower_text)
            if abstract_start_match:
                abstract_start_index = abstract_start_match.end()
                abstract_end_patterns = [
                    r'\b(introduction|引言|i\. introduction|1\.\s*introduction)\b',
                    r'\b(keywords|关键词)\b',
                    r'\b(background|背景)\b',
                    r'\b(section)\b' 
                ]
                abstract_end_index = len(text)
                for pattern in abstract_end_patterns:
                    match = re.search(pattern, lower_text[abstract_start_index:])
                    if match:
                        if abstract_start_index + match.start() < abstract_end_index:
                            abstract_end_index = abstract_start_index + match.start()
                
                abstract = text[abstract_start_index:abstract_end_index].strip()
                abstract = re.sub(r'^(?:author|authors|doi|keywords|摘要|abstract)[:\s]*.*?\n', '', abstract, flags=re.IGNORECASE|re.DOTALL).strip()
                abstract = re.sub(r'^(?:[A-Z][a-z]+\s+){1,4}(?:et al\.)?\s*.*?\n', '', abstract).strip() 

            author_search_area = text[len(title or ''): abstract_start_match.start() if abstract_start_match else min(len(text), len(title or '') + 500)]
            author_pattern = re.compile(
                r"(?:[A-Z][a-z'-]+(?:,\s*[A-Z][a-z'-]+)*\s+(?:[A-Z]\.\s*)*[A-Z][a-z'-]+|" 
                r"\b[A-Z]\.\s*[A-Z]\.\s*[A-Z][a-z'-]+(?:,\s*[A-Z]\.\s*[A-Z]\.\s*[A-Z][a-z'-]+)*|" 
                r"\b[A-Z][a-z'-]+(?:\s+[A-Z][a-z'-]+)+)" 
            )
            potential_authors = author_pattern.findall(author_search_area)
            
            authors = sorted(list(set([a.strip() for a in potential_authors 
                                       if len(a.strip()) > 5 and ' ' in a.strip() 
                                       and not re.search(r'\d|@|\b(?:university|institute|department)\b', a.lower())])))
            
            references = []

            references_start_match = re.search(r'(?i)\b(references|参考文献|bibliography|literatures)\b', lower_text)
            if references_start_match:
                references_start_index = references_start_match.end()
                references_end_index = len(text)

                appendix_match = re.search(
                    r'(?i)\b(appendix|附录|acknowledg(e)?ments?|致谢|鸣谢)\b',
                    lower_text[references_start_index:])
                if appendix_match:
                    references_end_index = references_start_index + appendix_match.start()

                references_section = text[references_start_index:references_end_index].strip()
                ref_pattern = re.compile(
                    r'^\s*(\[\d+\][\s\S]*?)(?=^\s*\[\d+\]|\Z)',  
                    re.MULTILINE
                )

                matches = ref_pattern.findall(references_section) 
                references = [m.strip() for m in matches]

            doc.close()

        except Exception as e:
            print(f"❌ Error during title, abstract, author, or references extraction: {e}")

        return title, abstract, authors, references


    def split_into_paragraphs(self, text, min_len=100):
        paras = re.split(r"(\n\s*\n\s*){1,}", text) 
        paras = [p.strip() for p in paras if len(p.strip()) >= min_len]
        
        if len(paras) < 3 and len(text) > 500: 
            sents = re.split(r"(?<=[.!?])\s+", text)
            paras, buf = [], ""
            for s in sents:
                buf += " " + s
                if len(buf.strip()) > min_len:
                    paras.append(buf.strip())
                    buf = ""
            if buf.strip():
                paras.append(buf.strip())
        return paras


    def add_paper(self, pdf_path):
        text = self.parse_pdf(pdf_path)
        if not text:
            print(f"❌ No text extracted from: {pdf_path}")
            return False

        title, abstract, authors, references = self.extract_title_and_abstract_authors_references(text, pdf_path)
        
        if not title:
            print(f"⚠️ Could not extract a reliable title, skipping: {pdf_path}")
            return False

        for existing_title in self.stored_titles:
            if fuzz.ratio(title.lower(), existing_title.lower()) >= 90: 
                print(f"⏭ Already exists (similar title: {existing_title}): {title}")
                return False

        self.paper_summaries[title] = abstract if abstract else (text[:200] + "...") 
        self.paper_paragraphs[title] = self.split_into_paragraphs(text)
        self.paper_references[title] = references
        self.stored_titles.append(title)
        self.save_metadata()
        print(f"📥 Text cached: {title}")
        return True

    def update_rag_collection(self):
        print("🔄 Updating RAG...")
        existing_rag_titles = set(m['title'] for m in self.collection.get(limit=self.collection.count(), include=['metadatas'])['metadatas'])

        added_to_rag_count = 0
        for title, paras in self.paper_paragraphs.items():
            if title in existing_rag_titles:
                continue 

            ids = [f"{title}_{i}" for i in range(len(paras))]
            metas = [{"title": title, "paragraph_id": i} for i in range(len(paras))]
            
            embeds = self.embedding_model.encode(paras, convert_to_tensor=True).tolist()
            
            self.collection.add(ids=ids, embeddings=embeds, documents=paras, metadatas=metas)
            added_to_rag_count += 1

        print(f"✅ Synced {added_to_rag_count} new papers to RAG. Total {len(self.stored_titles)} papers.")

    def _get_text_ngrams(self, text, n_min=1, n_max=3):
        words = re.findall(r'\b\w+\b', text.lower())
        ngrams = []
        for n in range(n_min, n_max + 1):
            ngrams.extend([' '.join(words[i:i+n]) for i in range(len(words) - n + 1)])
        return Counter(ngrams)

    def _download_arxiv_paper(self, arxiv_id, download_dir="./downloaded_arxiv_papers"):
        """
        Downloads a paper from arXiv given its ID.
        Returns the path to the downloaded PDF if successful, None otherwise.
        """
        os.makedirs(download_dir, exist_ok=True)
        try:
            print(f"Attempting to download arXiv:{arxiv_id}...")
            search = arxiv.Search(id_list=[arxiv_id])
            paper = next(search.results())
            
            # Use the paper's title for a clean filename
            filename = re.sub(r'[^\w\s-]', '', paper.title).strip()[:100] + ".pdf"
            filepath = os.path.join(download_dir, filename)

            paper.download_pdf(dirpath=download_dir, filename=filename)
            print(f"✅ Downloaded arXiv:{arxiv_id} to {filepath}")
            return filepath
        except Exception as e:
            print(f"❌ Failed to download arXiv:{arxiv_id}: {e}")
            return None

    def resolve_citation_marker_to_title(self, citation_marker, article_references):
        """
        Resolves a citation marker (e.g., '[1]', '(Smith, 2020)') to a full paper title
        by matching against the article's bibliography and then to the knowledge base.
        Incorporates individual LLM calls and arXiv download.
        """
        # Step 1: Try to extract title directly if marker contains it (e.g., [Smith2020])
        match = re.search(r'\[([A-Za-z]+)\d{4}\]', citation_marker)
        if match:
            potential_author_year_title = match.group(1) + " " + citation_marker[-5:-1]
            for stored_title in self.stored_titles:
                if fuzz.partial_ratio(potential_author_year_title.lower(), stored_title.lower()) > 70:
                    print(f"💡 Resolved '{citation_marker}' to '{stored_title}' via direct marker match.")
                    return stored_title

        target_ref_entry = None
        
        # For numerical citations like [1], [10]
        if re.match(r'\[(\d+)\]', citation_marker):
            ref_num = citation_marker.strip('[]')
            for ref_entry in article_references:
                if re.match(rf"^{re.escape(ref_num)}\.\s|^\[{re.escape(ref_num)}\]\s", ref_entry) or \
                   re.match(rf"^{re.escape(ref_num)}\.\s*\S", ref_entry) or \
                   re.match(rf"^\[{re.escape(ref_num)}\]\s*\S", ref_entry):
                    target_ref_entry = ref_entry
                    print(f"Found potential bibliography entry for '{citation_marker}': '{target_ref_entry[:100]}...'")
                    break
        
        # For author-year citations like (Smith, 2020)
        author_year_match = re.search(r'\(([^,]+),\s*(\d{4})\)', citation_marker)
        if author_year_match:
            author_part = author_year_match.group(1).split('et al.')[0].strip().lower()
            year_part = author_year_match.group(2)
            
            for ref_entry in article_references:
                if author_part in ref_entry.lower() and year_part in ref_entry:
                    target_ref_entry = ref_entry
                    print(f"Found potential bibliography entry for '{citation_marker}': '{target_ref_entry[:100]}...'")
                    break

        if target_ref_entry:
            # --- Reverted to Individual LLM Call for Title Extraction ---
            prompt = f"""
            From the following academic reference entry, extract ONLY the precise and full title. 
            Do NOT include authors, publication year, journal name, volume, page numbers, DOI, or any other metadata.
            Your output should be the title itself, with no introductory phrases or explanations.

            Reference Entry:
            "{target_ref_entry}"
            
            Precise Title:
            """
            print(f"Calling DeepSeek LLM to extract title from bibliography entry for '{citation_marker}'...")
            llm_extracted_title = call_llm_api(prompt, max_tokens=100, temperature=0.0)
            
            if llm_extracted_title and "LLM API Error" not in llm_extracted_title and llm_extracted_title.strip():
                llm_extracted_title = re.sub(r'^(Precise )?Title:\s*', '', llm_extracted_title, flags=re.IGNORECASE).strip()
                llm_extracted_title = llm_extracted_title.strip('"').strip('“').strip('”')

                print(f"DeepSeek LLM extracted title: '{llm_extracted_title}' for marker '{citation_marker}'.")
                
                for stored_title in self.stored_titles:
                    similarity_ratio = fuzz.ratio(llm_extracted_title.lower(), stored_title.lower())
                    partial_similarity_ratio = fuzz.partial_ratio(llm_extracted_title.lower(), stored_title.lower())

                    if similarity_ratio >= 85 or partial_similarity_ratio >= 90:
                        print(f"💡 Resolved '{citation_marker}' to '{stored_title}' (Fuzz Ratio: {similarity_ratio}, Partial Ratio: {partial_similarity_ratio}).")
                        return stored_title
                
                # --- ADDED: arXiv Download Logic for unresolved titles ---
                print(f"⚠️ Extracted title '{llm_extracted_title}' did not match any stored papers for '{citation_marker}'. Checking for arXiv...")
                arxiv_id_match = re.search(r'(?:arxiv:|ARXIV:)\s*(\d{4}\.\d{5}(?:v\d+)?)|\[?(\d{4}\.\d{5}(?:v\d+)?)\s*\]?', target_ref_entry.lower())
                
                if arxiv_id_match:
                    arxiv_id = arxiv_id_match.group(1) or arxiv_id_match.group(2)
                    if arxiv_id:
                        print(f"🚀 Detected arXiv ID: {arxiv_id} for '{citation_marker}'. Attempting to download and add to RAG.")
                        downloaded_pdf_path = self._download_arxiv_paper(arxiv_id)
                        if downloaded_pdf_path:
                            print(f"Adding downloaded paper '{downloaded_pdf_path}' to RAG.")
                            if self.add_paper(downloaded_pdf_path):
                                print("Successfully added arXiv paper to RAG. Retrying citation resolution.")
                                # Update ChromaDB immediately to ensure the new paper is available for query
                                self.update_rag_collection() 
                                # Re-attempt resolution by matching the newly added title
                                for stored_title in self.stored_titles: # Check all titles, including the new one
                                    if fuzz.ratio(llm_extracted_title.lower(), stored_title.lower()) >= 85: # Use a reasonable threshold for newly added
                                        print(f"💡 Resolved '{citation_marker}' to newly downloaded arXiv paper '{stored_title}'.")
                                        return stored_title
                                print(f"⚠️ Newly added arXiv paper title did not match extracted title for '{citation_marker}'.")
                            else:
                                print(f"❌ Failed to add downloaded arXiv paper to RAG for '{citation_marker}'.")
                        else:
                            print(f"❌ Download of arXiv paper failed for '{citation_marker}'.")
                else:
                    print(f"No arXiv ID detected in reference entry for '{citation_marker}'.")

            else:
                print(f"⚠️ DeepSeek LLM failed to extract a valid title for '{citation_marker}' from entry: '{target_ref_entry[:100]}...'. LLM Response: {llm_extracted_title}")

        return None 
    
    import re

    def parse_llm_xml_response(response):
        """Parses LLM XML output and extracts <answer> and <reason>."""
        answer_match = re.search(r"<answer>\s*(.*?)\s*</answer>", response, re.IGNORECASE | re.DOTALL)
        reason_match = re.search(r"<reason>\s*(.*?)\s*</reason>", response, re.IGNORECASE | re.DOTALL)
        answer = answer_match.group(1).strip().lower() if answer_match else "unknown"
        reason = reason_match.group(1).strip() if reason_match else "No reason provided."
        return answer, reason


    def verify_citation(self, citation_marker, cited_context, article_references, sim_thresh=0.05, keyword_overlap_thresh=0.00,change=True): # <-- Adjusted thresholds
        res = {"citation_marker": citation_marker, "is_valid": False, "issues": [], "evidence": [], "resolved_title": None}
        
        # Step 1: Resolve citation marker to a known title in the knowledge base
        resolved_title = self.resolve_citation_marker_to_title(citation_marker, article_references)

        if not resolved_title:
            res["issues"].append(f"❌ Citation marker '{citation_marker}' could not be resolved to a known paper title in the knowledge base.")
            return res
        
        res["resolved_title"] = resolved_title
        
        print(f"🔍 Resolved '{citation_marker}' to '{resolved_title}'. Proceeding with content verification.")

        if cited_context:
            try:
                qr = self.collection.query(
                    query_texts=[cited_context],
                    n_results=3, # Get top 3 most relevant paragraphs
                    where={"title": resolved_title}, # Use the RESOLVED title for query
                    include=['documents', 'distances', 'metadatas']
                )
                
                docs = qr.get("documents", [[]])[0]
                dists = qr.get("distances", [[]])[0]
                metadatas = qr.get("metadatas", [[]])[0]

                if docs:
                    final_llm_confirmed_validity = False # This flag now dictates overall validity
                    
                    cited_context_ngrams = self._get_text_ngrams(cited_context) # Calculate ngrams once
                    
                    for doc, dist, meta in zip(docs, dists, metadatas):
                        semantic_sim = 1 - dist / 2 
                        
                        doc_ngrams = self._get_text_ngrams(doc)
                        common_ngrams = cited_context_ngrams & doc_ngrams 
                        
                        if cited_context_ngrams: 
                            overlap_score = sum(common_ngrams.values()) / (sum(cited_context_ngrams.values()) + sum(doc_ngrams.values()) - sum(common_ngrams.values()))
                        else:
                            overlap_score = 0.0

                        combined_score = (semantic_sim * 0.7) + (overlap_score * 0.3) 

                        current_evidence = {
                            "text_fragment": doc,
                            "semantic_similarity": semantic_sim,
                            "keyword_overlap": overlap_score,
                            "combined_score": combined_score,
                            "paragraph_id": meta.get("paragraph_id"),
                            "source_title": meta.get("title")
                        }
                        res["evidence"].append(current_evidence) # Always add evidence for inspection

                        # Lower thresholds mean more potential RAG matches are passed to LLM for final check
                        if semantic_sim >= min(sim_thresh, -1) and overlap_score >= min(keyword_overlap_thresh, -1):
                            # Step 3: LLM for Semantic Consistency and Distortion Check (High Weight)
                            prompt_llm_check = f"""
                            You are a scientific research assistant. Your task is to assess whether a citation in a paper accurately reflects the content of the cited source.

                            Given the following statement from an article:
                            "{cited_context}"

                            And the paragraph from the cited document ("{resolved_title}"):
                            "{doc}"

                            Determine if the article statement accurately reflects or is supported by the cited paragraph. Answer "yes" if it is accurately represented, or "no" if it is misrepresented, distorted, or fabricated.

                            Respond in the following format:
                            <xml><answer>yes/no</answer><reason>Brief explanation...</reason></xml>
                                """

                            
                            if change==True:
                                print("Calling DeepSeek LLM for semantic check...")
                                llm_response = call_llm_api(prompt_llm_check, max_tokens=150, temperature=0.1)
                                print(f"DeepSeek LLM Response: {llm_response}")
                            elif change==False:
                                print("Local LLM for semantic check...")
                                llm_response = call_local_api(prompt_llm_check, max_tokens=150, temperature=0.1)
                                print(f"Lcoal; LLM Response: {llm_response}")
                            

                            # Parse LLM response
                            answer, reason = parse_llm_xml_response(llm_response)

                            # Save training data for finetuning
                            with open("citation_data.jsonl", "a", encoding="utf-8") as f_out:
                                json.dump({
                                    "x1": cited_context,
                                    "x2": doc,
                                    "label": {
                                        "answer": answer,
                                        "reason": reason
                                    }
                                }, f_out, ensure_ascii=False)
                                f_out.write("\n")

                            # Append to issue log
                            if answer == "yes":
                                final_llm_confirmed_validity = True
                                res["issues"].append(f" LLM confirms support: {reason}")
                                break  # Stop checking further matches
                            else:
                                res["issues"].append(
                                    f" LLM suggests distortion or lack of support (sem_sim: {semantic_sim:.2f}, overlap: {overlap_score:.2f}): {reason}"
                                )
                        else:
                            res["issues"].append(
                                f"RAG match (sem_sim: {semantic_sim:.2f}, overlap: {overlap_score:.2f}) below initial thresholds (LLM not consulted for this specific match)."
                            )

                    # The overall validity of the citation is now determined solely by whether
                    # the LLM confirmed any strong RAG match.
                    res["is_valid"] = final_llm_confirmed_validity

                    if not res["is_valid"] and not res["issues"]:
                        res["issues"].append("No sufficiently strong RAG match found, or LLM did not confirm any RAG match as supportive.")
                    elif not res["is_valid"] and not final_llm_confirmed_validity:
                        # This covers cases where RAG matches were found and checked, but LLM didn't give a "YES" for any
                        res["issues"].append("While potential evidence was found (see 'evidence' for scores), the LLM did not confirm the statement's support for any match.")
                else:
                    res["issues"].append("No relevant paragraphs found in the cited document for the given context.")
            except Exception as e:
                res["issues"].append(f"Error during RAG query or DeepSeek LLM semantic check: {e}")
                
        else: 
            # If no cited context is provided, we can't verify content, so we just assume validity for marker resolution
            res["is_valid"] = True
            res["issues"].append("No cited context provided for content verification. Only citation resolution performed.")

        return res

    def query_knowledge_base(self, query_text, n_results=5, sim_thresh=0.6):
        results = {"query": query_text, "matches": []}
        
        if not query_text:
            results["issues"] = ["Query text cannot be empty."]
            return results
        
        try:
            qr = self.collection.query(
                query_texts=[query_text],
                n_results=n_results,
                include=['documents', 'distances', 'metadatas']
            )
            
            docs = qr.get("documents", [[]])[0]
            dists = qr.get("distances", [[]])[0]
            metadatas = qr.get("metadatas", [[]])[0]

            if docs:
                query_ngrams = self._get_text_ngrams(query_text)
                for doc, dist, meta in zip(docs, dists, metadatas):
                    semantic_sim = 1 - dist / 2
                    
                    doc_ngrams = self._get_text_ngrams(doc)
                    common_ngrams = query_ngrams & doc_ngrams
                    if query_ngrams:
                        overlap_score = sum(common_ngrams.values()) / (sum(query_ngrams.values()) + sum(doc_ngrams.values()) - sum(common_ngrams.values()))
                    else:
                        overlap_score = 0.0

                    combined_score = (semantic_sim * 0.7) + (overlap_score * 0.3) 
                    
                    if combined_score >= sim_thresh: 
                        results["matches"].append({
                            "doc_title": meta.get("title"),
                            "matched_paragraph": doc,
                            "semantic_score": semantic_sim,
                            "keyword_overlap_score": overlap_score,
                            "combined_score": combined_score,
                            "paragraph_id": meta.get("paragraph_id") 
                        })
                results["matches"] = sorted(results["matches"], key=lambda x: x["combined_score"], reverse=True)
                if not results["matches"]:
                    results["issues"] = [f"No matches found with combined score above {sim_thresh:.2f}."]
            else:
                results["issues"] = ["No matching content found."]
                
        except Exception as e:
            results["issues"] = [f"Error querying knowledge base: {e}"]
            
        return results