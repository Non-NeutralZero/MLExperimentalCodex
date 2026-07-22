import requests


def find_in_artic(query: str) -> str:
    """
    Search the art institure of chicago and return the top results as readable text.
    Used when the user wants actual artworks they can view or print (returns title, artist, public-domain status, image URL)

    Args:
        query: what to search for

    Returns:
        A formatted string of results (title of artwork, artist, date of display, public domain status, image url), or a plain
        message if the search failed or found nothing.
    
    """
    fetch_url = "https://api.artic.edu/api/v1/artworks/search"
    query_params = {
        "q": query,
        "limit": 5,
        "fields": "id,title,artist_display,date_display,is_public_domain,image_id",
    }
    try:
        response = requests.get(fetch_url, params=query_params)
    except Exception as e:
        return f"Search failed: {type(e).__name__}. Try another key word."

    data = response.json()
    if not data["data"]:
        return "No results found."

    lines = []
    for i, r in enumerate(data["data"], start=1):
        if r["image_id"]:
            image = f"https://www.artic.edu/iiif/2/{r['image_id']}/full/843,/0/default.jpg"
        else:
            image = "no image available"

        public_domain = "yes" if r["is_public_domain"] else "no"

        lines.append(f"[{i}] {r['title']}")
        lines.append(f"    artist: {r['artist_display']}")
        lines.append(f"    date: {r['date_display']}")
        lines.append(f"    public domain: {public_domain}")
        lines.append(f"    image: {image}")
        lines.append("")

    return "\n".join(lines)


#if __name__ == "__main__":
#    print(find_in_artic("monet"))
