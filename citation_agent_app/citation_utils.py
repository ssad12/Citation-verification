import re

def extract_citations_from_text(text):
    """
    提取文中所有疑似引用的标注，包括：[1]、(Smith et al., 2020)、[Smith2020]、【王2020】等
    改进：更精确的正则表达式，避免误伤日期等
    """
    patterns = [
        r'\[\d+\]',                                  # [1], [12]
        r'\[[A-Za-z]+(?:\s*et al\.)?\s*,\s*\d{4}\]', # [Smith, 2020], [Smith et al., 2020]
        r'\[[A-Za-z]+[0-9]{4}\]',                    # [Smith2020]
        r'\((?:[A-Z][a-z]+(?:,\s*[A-Z][a-z]+)*\s*(?:and|&)\s*)?[A-Z][a-z]+(?:\s*et al\.)?,\s*\d{4}[a-z]?\)', # (Smith, 2020), (Smith et al., 2020), (Smith and Jones, 2020a)
        r'【[^】]{2,40}\s*(?:20\d{2}|19\d{2})】'      # 【王2020】, 【王等 2020】 - 增加年份范围
    ]

    citations = set()
    for pat in patterns:
        matches = re.findall(pat, text)
        citations.update(match.strip() for match in matches)

    return sorted(list(citations))



