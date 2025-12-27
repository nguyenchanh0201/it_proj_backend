import re

def extract_mermaid_content(text):
    mermaid_pattern = r"```mermaid\s*([\s\S]*?)\s*```"
    match = re.search(mermaid_pattern, text, re.IGNORECASE)
    
    if match:
        return match.group(1).strip()
    fallback_pattern = r"```\s*([\s\S]*?)\s*```"
    match_fallback = re.search(fallback_pattern, text)
    
    if match_fallback:
        return match_fallback.group(1).strip()
    return text.strip()

