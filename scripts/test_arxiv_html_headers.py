import requests
from bs4 import BeautifulSoup

ARXIV_ID = "2501.12345"  # replace with your paper ID
PREVIEW_LENGTH = 100


def get_h2_sections(arxiv_id):
    url = f"https://arxiv.org/html/{arxiv_id}"

    response = requests.get(url)
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

        content = " ".join(content_parts)
        sections.append(
            {
                "header": title,
                "preview": content[:PREVIEW_LENGTH],
            }
        )

    return sections


if __name__ == "__main__":
    sections = get_h2_sections(ARXIV_ID)

    if not sections:
        print("No h2 headers found. This paper may not have an HTML version.")
    else:
        print("\n--- H2 HEADERS ---\n")
        for index, section in enumerate(sections, start=1):
            print(f"{index}. {section['header']}")
            print(f"   {section['preview']}")
            print()
