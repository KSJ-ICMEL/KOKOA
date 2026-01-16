"""
KOKOA Tools Module
==================
CASCADE 스타일의 도구 모음:
- Web Search (DuckDuckGo - 무료)
- Code Extraction from URLs
- Python Introspection
"""

import re
import os
import ast
import inspect
import importlib
import pkgutil
from typing import List, Dict, Any, Optional
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup


def web_search(query: str, max_results: int = 5, search_depth: str = "advanced") -> List[Dict[str, str]]:
    """
    Web search using Tavily (AI-optimized search engine used in CASCADE)
    
    Requires TAVILY_API_KEY environment variable.
    Get API key at: https://app.tavily.com
    
    Args:
        query: Search query
        max_results: Maximum number of results (default: 5)
        search_depth: "basic" or "advanced" (default: advanced for better code examples)
    
    Returns:
        List of {"title": ..., "url": ..., "snippet": ..., "content": ...}
    """
    import os
    
    api_key = os.getenv("TAVILY_API_KEY")
    
    if api_key:
        try:
            from langchain_community.tools.tavily_search import TavilySearchResults
            
            tavily = TavilySearchResults(
                max_results=max_results,
                search_depth=search_depth,
                include_answer=True,
                include_raw_content=False
            )
            
            raw_results = tavily.invoke(query)
            
            results = []
            if isinstance(raw_results, list):
                for r in raw_results:
                    results.append({
                        "title": r.get("title", ""),
                        "url": r.get("url", ""),
                        "snippet": r.get("content", "")[:500],
                        "content": r.get("content", "")
                    })
            
            print(f"   [Tavily] Found {len(results)} results for: {query[:50]}...")
            return results
            
        except ImportError:
            print("   [Tavily] langchain-community not installed. Falling back to DuckDuckGo.")
        except Exception as e:
            print(f"   [Tavily] Error: {e}. Falling back to DuckDuckGo.")
    
    try:
        from duckduckgo_search import DDGS
        
        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                results.append({
                    "title": r.get("title", ""),
                    "url": r.get("href", ""),
                    "snippet": r.get("body", ""),
                    "content": r.get("body", "")
                })
        
        print(f"   [DuckDuckGo] Found {len(results)} results for: {query[:50]}...")
        return results
        
    except ImportError:
        print("   [WebSearch] No search backend available. Install: pip install duckduckgo-search")
        return []
    except Exception as e:
        print(f"   [WebSearch] Error: {e}")
        return []


def extract_code_from_url(url: str, language: str = "python") -> List[str]:
    """
    Extract code blocks from a URL (GitHub, docs, etc.)
    
    Args:
        url: URL to extract code from
        language: Programming language to filter (default: python)
    
    Returns:
        List of code snippets
    """
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        code_blocks = []
        
        for pre in soup.find_all('pre'):
            code = pre.get_text()
            if code.strip():
                code_blocks.append(code.strip())
        
        for code_tag in soup.find_all('code'):
            classes = code_tag.get('class', [])
            if any(language in c.lower() for c in classes):
                code = code_tag.get_text()
                if code.strip() and len(code) > 50:
                    code_blocks.append(code.strip())
        
        md_pattern = rf'```{language}\s*(.*?)\s*```'
        md_matches = re.findall(md_pattern, response.text, re.DOTALL | re.IGNORECASE)
        code_blocks.extend(md_matches)
        
        unique_blocks = list(dict.fromkeys(code_blocks))
        
        print(f"   [CodeExtract] Found {len(unique_blocks)} code blocks from {urlparse(url).netloc}")
        return unique_blocks[:10]
        
    except Exception as e:
        print(f"   [CodeExtract] Error: {e}")
        return []


def quick_introspect(
    package_name: str = None,
    class_hint: str = None,
    method_hint: str = None,
    function_hint: str = None
) -> Dict[str, Any]:
    """
    Quick introspection of Python packages/classes/methods
    
    Args:
        package_name: Package to inspect (e.g., "pymatgen.core")
        class_hint: Class name to find
        method_hint: Method name to find
        function_hint: Function name to find
    
    Returns:
        Dict with found symbols and their signatures
    """
    result = {
        "package": package_name,
        "classes": [],
        "functions": [],
        "methods": [],
        "suggestions": []
    }
    
    if not package_name:
        return result
    
    try:
        module = importlib.import_module(package_name)
        
        for name, obj in inspect.getmembers(module):
            if name.startswith('_'):
                continue
            
            if inspect.isclass(obj):
                class_info = {
                    "name": name,
                    "full_path": f"{package_name}.{name}",
                    "methods": []
                }
                
                for method_name, method_obj in inspect.getmembers(obj, predicate=inspect.isfunction):
                    if not method_name.startswith('_'):
                        try:
                            sig = str(inspect.signature(method_obj))
                            class_info["methods"].append(f"{method_name}{sig}")
                        except:
                            class_info["methods"].append(method_name)
                
                if class_hint and class_hint.lower() in name.lower():
                    result["classes"].insert(0, class_info)
                else:
                    result["classes"].append(class_info)
            
            elif inspect.isfunction(obj):
                try:
                    sig = str(inspect.signature(obj))
                    func_info = f"{name}{sig}"
                except:
                    func_info = name
                
                if function_hint and function_hint.lower() in name.lower():
                    result["functions"].insert(0, func_info)
                else:
                    result["functions"].append(func_info)
        
        result["classes"] = result["classes"][:5]
        result["functions"] = result["functions"][:10]
        
        print(f"   [Introspect] {package_name}: {len(result['classes'])} classes, {len(result['functions'])} functions")
        
    except ImportError as e:
        result["error"] = f"Cannot import {package_name}: {e}"
        print(f"   [Introspect] Import error: {e}")
    except Exception as e:
        result["error"] = str(e)
        print(f"   [Introspect] Error: {e}")
    
    return result


def runtime_probe(code_snippet: str, variable_name: str = None) -> Dict[str, Any]:
    """
    Execute a code snippet and probe runtime values
    
    Args:
        code_snippet: Python code to execute
        variable_name: Optional variable to inspect after execution
    
    Returns:
        Dict with execution result and variable info
    """
    result = {
        "success": False,
        "output": "",
        "variables": {},
        "error": None
    }
    
    try:
        local_vars = {}
        exec(code_snippet, {"__builtins__": __builtins__}, local_vars)
        
        result["success"] = True
        
        for name, value in local_vars.items():
            if not name.startswith('_'):
                result["variables"][name] = {
                    "type": type(value).__name__,
                    "value": str(value)[:200] if not callable(value) else f"<{type(value).__name__}>"
                }
        
        if variable_name and variable_name in local_vars:
            obj = local_vars[variable_name]
            result["probed"] = {
                "name": variable_name,
                "type": type(obj).__name__,
                "attributes": [a for a in dir(obj) if not a.startswith('_')][:20],
                "methods": [m for m in dir(obj) if callable(getattr(obj, m, None)) and not m.startswith('_')][:20]
            }
        
        print(f"   [Probe] Executed successfully, {len(result['variables'])} variables")
        
    except Exception as e:
        result["error"] = str(e)
        print(f"   [Probe] Execution error: {e}")
    
    return result


def get_package_structure(package_name: str, max_depth: int = 2) -> Dict[str, Any]:
    """
    Get the structure of a Python package
    
    Args:
        package_name: Package name (e.g., "pymatgen")
        max_depth: Maximum depth to explore
    
    Returns:
        Dict with package structure
    """
    structure = {
        "name": package_name,
        "submodules": [],
        "error": None
    }
    
    try:
        package = importlib.import_module(package_name)
        
        if hasattr(package, '__path__'):
            for importer, modname, ispkg in pkgutil.walk_packages(
                package.__path__, 
                prefix=package_name + ".",
            ):
                depth = modname.count('.') - package_name.count('.')
                if depth <= max_depth:
                    structure["submodules"].append({
                        "name": modname,
                        "is_package": ispkg
                    })
        
        structure["submodules"] = structure["submodules"][:30]
        print(f"   [Structure] {package_name}: {len(structure['submodules'])} submodules")
        
    except Exception as e:
        structure["error"] = str(e)
        print(f"   [Structure] Error: {e}")
    
    return structure


def format_search_results(results: List[Dict[str, str]]) -> str:
    """Format web search results for LLM context"""
    if not results:
        return "No search results found."
    
    lines = []
    for i, r in enumerate(results, 1):
        lines.append(f"[{i}] {r['title']}")
        lines.append(f"    URL: {r['url']}")
        lines.append(f"    {r['snippet'][:200]}")
        lines.append("")
    
    return "\n".join(lines)


def format_code_blocks(blocks: List[str]) -> str:
    """Format extracted code blocks for LLM context"""
    if not blocks:
        return "No code blocks found."
    
    lines = []
    for i, code in enumerate(blocks[:3], 1):
        lines.append(f"--- Code Block {i} ---")
        lines.append(code[:1000])
        lines.append("")
    
    return "\n".join(lines)
