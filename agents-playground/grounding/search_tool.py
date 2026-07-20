from ddgs import DDGS


def web_search(query: str) -> str:
    """Search the web and return the top results as readable text.

    Args:
      query: what to search for

    Returns:
      A formatted string of results (title, url, snippet), or a plain
      message if the search failed or found nothing.
    """
    # any failure (rate limit, timeout, network) must come back as a
    # readable observation the model can react to.
    try:
        results = DDGS().text(query, max_results=5)
    except Exception as e:
        return f"Search failed: {type(e).__name__}. Try rephrasing or searching again."

    if not results:
        return "No results found."

    lines = []
    for i, r in enumerate(results, start=1):
        lines.append(f"[{i}] {r['title']}")
        lines.append(f"    {r['href']}")
        lines.append(f"    {r['body']}")
        lines.append("") 
    return "\n".join(lines)