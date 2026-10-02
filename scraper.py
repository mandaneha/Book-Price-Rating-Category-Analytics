import requests
from bs4 import BeautifulSoup
import pandas as pd

all_books = []

rating_map = {
    "One": 1,
    "Two": 2,
    "Three": 3,
    "Four": 4,
    "Five": 5
}

for page in range(1, 51):

    url = f"https://books.toscrape.com/catalogue/page-{page}.html"

    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")

    books = soup.find_all("article", class_="product_pod")

    for book in books:
       
        title = book.h3.a["title"]
     
        price = book.find("p", class_="price_color").text.strip()
        
        rating_text = book.p["class"][1]
        rating = rating_map[rating_text]
        
        availability = book.find(
            "p", class_="instock availability"
        ).text.strip()
        
        product_url = (
            "https://books.toscrape.com/catalogue/"
            + book.h3.a["href"].replace("../", "")
        )
        
        product_response = requests.get(product_url)

        product_soup = BeautifulSoup(
            product_response.text, "html.parser"
        )

        breadcrumb = product_soup.find(
            "ul", class_="breadcrumb"
        )

        category = breadcrumb.find_all("li")[2].text.strip()

        
        all_books.append({
            "Title": title,
            "Category": category,
            "Price": price,
            "Rating": rating,
            "Availability": availability,
            "Product URL": product_url
        })

    print(f"Page {page} completed")

df = pd.DataFrame(all_books)

df.to_csv(
    "books_data.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\nScraping completed!")
print("Total books:", len(df))
print("CSV file created: books_data.csv")
print("\nDataset shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())
print("\nMissing values:")
print(df.isnull().sum())