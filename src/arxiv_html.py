import requests
from bs4 import BeautifulSoup

ARXIV_HTML_URL = "https://arxiv.org/html/{arxiv_id}"


def get_h2_sections(arxiv_id: str, timeout: float = 15) -> list[dict]:
    response = requests.get(ARXIV_HTML_URL.format(arxiv_id=arxiv_id), timeout=timeout)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    sections = []

    for header in soup.find_all("h2"):
        title = header.get_text(" ", strip=True)
        content_parts = []

        for sibling in header.next_siblings:
            if getattr(sibling, "name", None) in {"h1", "h2"}:
                break

            if not getattr(sibling, "get_text", None):
                continue

            text = sibling.get_text(" ", strip=True)
            if text:
                content_parts.append(text)

        sections.append(
            {
                "header": title,
                "content": " ".join(content_parts),
            }
        )

    return sections
