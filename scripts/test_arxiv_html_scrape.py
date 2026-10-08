import requests
from bs4 import BeautifulSoup

ARXIV_ID = "2501.12345"  # replace with your paper ID


def get_abstract_from_html(arxiv_id):
    url = f"https://arxiv.org/html/{arxiv_id}"

    response = requests.get(url)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    # ArXiv HTML uses the abstract environment
    abstract = soup.find("div", class_="ltx_abstract")

    if abstract:
        return abstract.get_text(" ", strip=True)

    return "Abstract not found. This paper may not have an HTML version."


if __name__ == "__main__":
    abstract = get_abstract_from_html(ARXIV_ID)

    print("\n--- ABSTRACT ---\n")
    print(abstract)
