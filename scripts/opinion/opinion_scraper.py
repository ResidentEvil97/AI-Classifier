import pandas as pd
from newspaper import Article
import os

urls = [
    "https://www.gse.harvard.edu/ideas/usable-knowledge/24/10/what-causing-our-epidemic-loneliness-and-how-can-we-fix-it",
    "https://www.nytimes.com/2024/09/09/learning/should-schools-ban-student-phones.html",
    "https://www.psypost.org/what-makes-someone-a-perfect-friend-heres-what-new-research-says/",
    "https://news.harvard.edu/gazette/story/2019/01/perspectives-on-gene-editing/",
    "https://www.psychologytoday.com/us/blog/artificial-intelligence-in-behavioral-and-mental-health-care/202402/could-artificial",
    "https://www.psychologytoday.com/us/blog/insight-therapy/202401/sleep-is-more-important-than-you-think",
    "https://www.nature.com/articles/d41586-024-00998-6",
    "https://www.pewtrusts.org/en/trend/archive/fall-2024/5-ways-to-rebuild-trust-in-government",
    "https://www.linkedin.com/pulse/hidden-cost-remote-work-lawrence-coburn/",
    "https://www.psychologytoday.com/us/blog/the-freedom-change/201811/attachment-theory-elections-and-the-politics-fear",
    "https://medium.com/@frankbreslin41/the-case-for-philosophy-in-americas-high-schools-33ef79167f1f",
    "https://medium.com/@i.samiprem/the-power-of-nostalgia-why-90s-and-2000s-trends-are-making-a-comeback-5fde17faf3f3",
]

data = []
for url in urls:
    try:
        article = Article(url)
        article.download()
        article.parse()
        text = article.text.strip()
        if len(text) > 300:  # skip broken articles
            data.append({"text": text, "label": "human"})
    except Exception as e:
        print(f"Failed to parse {url}: {e}")

# Save to CSV
os.makedirs("data/opinion", exist_ok=True)
pd.DataFrame(data).to_csv("data/opinion/opinion_human.csv", index=False)
print("Saved opinion_human.csv")