from selenium import webdriver
from selenium.webdriver.common.by import By
import time

from vectorization import get_text_chunks, get_embedding, insert_data

# Initialize Selenium WebDriver (Make sure you have ChromeDriver installed)
driver = webdriver.Chrome()

# Open the webpage
driver.get("https://heyharper.com/eu/en/products")

# Wait a bit for JavaScript to load
time.sleep(3)

def get_product_names():
    # Find product names
    products = driver.find_elements(By.XPATH, '//a[@aria-label="Product name"]')

    # Print each product title
    for product in products:
        print(product.text)

def get_all_text():
    # Get all text that is visible
    visible_text = driver.find_element(By.TAG_NAME, "body").text
    print(visible_text)
    return visible_text

def main():
  get_raw_text=get_all_text()
  chunks=get_text_chunks(get_raw_text)
  vectors=get_embedding(chunks)
  insert_data(vectors)

# Close the browser
driver.quit()
