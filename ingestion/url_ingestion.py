import base64
import os
import re
import requests
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains

import difflib
import time

from llms.openai_utils import get_openai_client, openai_chat_completion
from chatbot import make_conversation_file, retrieve_from_vdb
from vectorization import get_points, get_text_chunks, insert_data, create_batches_of_text


def setup_driver(start_url):
    # Initialize Selenium WebDriver (Make sure you have ChromeDriver installed)
    driver = webdriver.Chrome()
    driver.get(start_url)
    time.sleep(2)  # Allow JS to load
    return driver


"""-------------Products related functions------------------"""
def get_product_names(driver):
    # Find product names
    products = driver.find_elements(By.XPATH, '//a[@aria-label="Product name"]')

    # Print each product title
    for product in products:
        print(product.text)

def get_all_product_urls(driver):
    print("entrou no get_all_product_urls")
    product_elements = driver.find_elements(By.XPATH, "//a[starts-with(@href, '/eu/en/products/')]")

    products_urls = []
    for element in product_elements:
        try:
            product_url = element.get_attribute("href")
            if product_url and (product_url.startswith("https://heyharper.com/eu/en/products/") or product_url.startswith("https://checkout-eu.heyharper.com/")) and not product_url.startswith(product_url, "https://heyharper.com/eu/en/products/") and product_url not in products_urls:
                print(product_url)
                products_urls.append(product_url)
        except:
                continue  # Ignore elements that fail

    return products_urls

"""--------Helper functions for other files----------"""
#it is used on csv_ingestion
def scrape_page(url):
    time.sleep(0.1)
    driver = setup_driver(url)
    try:
        text = get_all_text(driver)
        return text
    finally:
        driver.quit()

"""-------------Specific functions------------------"""
def find_show_more_id(driver):
    # Find all buttons or elements you want to check
    button_elements = driver.find_elements(By.TAG_NAME, "button")
    print("entered the click_show_more")
    # Iterate through the elements and look for the one with the text "Show More"
    for button in button_elements:
        if button.text == "Show more":
            print("button id is ", button.id)
            return button.id
    
    # If the button can't be found or clicked (e.g., no more products), return False
    print("No more 'Show More' button found or error occurred:")
    return False

def click_show_more(driver):
    # Find all buttons or elements you want to check
    button_elements = driver.find_elements(By.TAG_NAME, "button")
    #print("entered the click_show_more")
    # Iterate through the elements and look for the one with the text "Show More"
    for button in button_elements:
        try:
            if button.text == "Show more":
                #scrolling until the button
                #print("button id is ", button.id)
                driver.execute_script("arguments[0].scrollIntoView(true);", button)
                # Once you find the button, click it
                #print("Clicked the 'Show more' button.")
                button.click()
                # Wait a bit for the content to load
                time.sleep(0.5)
                return True
        except:
            continue
    
    # If the button can't be found or clicked (e.g., no more products), return False
    print("No more 'Show More' button found or error occurred:")
    return False

def get_image_knowing_the_src(driver):
    openai_client = get_openai_client()

    #images = driver.find_elements(By.TAG_NAME, "img")
    #image_srcs = [image.get_attribute("src") for image in images]

    image_url = "https://a.storyblok.com/f/237022/2964x1040/fdbef14142/01_offer_desktop-us-1.png/m/2048x0/"

    # Download the image
    img_data = requests.get(image_url).content

    # Convert image data to base64
    base64_image = base64.b64encode(img_data).decode("utf-8")

    # Call GPT-4 Vision model
    response = openai_client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            { "role": "user", "content": [{ "type": "text", "text": "Please retrun the text of the image."},{ "type": "image_url", "image_url": { "url": f"data:image/png;base64,{base64_image}" }}]}],
        max_tokens=1000
    )
    return response.choices[0].message.content
"""-----------What to get functions------------------"""
#not in use, since get_all_hrefs to replace it
def get_all_clickable_buttons(driver):
    #print("entrou no get_all_clickable")
    clickables = driver.find_elements(By.CSS_SELECTOR, "a, button, [role='button'], [onclick], [tabindex='0'], [href]")
    #print(f"Clickables found: {clickables}")
    really_clickable = []
    clickable_with_hrefs = []
    for element in clickables:
        try:
            # Use JavaScript to get the href without clicking
            link = element.get_attribute("href")
            if link not in clickable_with_hrefs:
                clickable_with_hrefs.append(link)

            if element.is_displayed() and element.is_enabled():
                #element.click()
                really_clickable.append(element)
                #print(element.text, "is clickable!")
                #time.sleep(3)
        except Exception as e:
            # risky code probably
            if "element not interactable" in str(e).lower():
                print("Not interactable\nDisplayed: ", element.is_displayed(), " Enabled: ", element.is_enabled())
    
    print("clickables with hrefsfound :")
    print(*clickable_with_hrefs, sep="\n")
    print("clickckables with hrefs found: ", len(clickable_with_hrefs))
    print("really clickable found: ", len(really_clickable))
    #time.sleep(0.1)
    return really_clickable

def get_all_hrefs(driver):
    print("entrou no get_all_hrefs")
    elements = driver.find_elements(By.CSS_SELECTOR, "a, button, [role='button'], [onclick], [tabindex='0'], [href]")
    elements_list = []
    for element in elements:
        try:
            # Use JavaScript to get the href without clicking
            link = element.get_attribute("href")
            if link not in elements_list:
                elements_list.append(link)
        except Exception as e:
            # risky code probably
            if "element not interactable" in str(e).lower():
                print("Not interactable\nDisplayed: ", element.is_displayed(), " Enabled: ", element.is_enabled())
    print("hrefs found: ", len(elements_list))
    return elements_list

def get_all_text(driver):
    # Get all text that is visible
    visible_text = driver.find_element(By.TAG_NAME, "body").text
    #print("visible data", visible_text)
    return visible_text

"""-------------Effeciency functions------------------"""
def get_new_content(initial_text, new_text):
    """
    Compares two strings and returns the content present in the new text
    that was not in the initial text.
    """
    # Split the initial and new text into sets of lines for efficient comparison.
    initial_lines = set(initial_text.splitlines())
    new_lines = new_text.splitlines()

    # Find the lines that are in new_lines but not in initial_lines.
    newly_added_lines = [line for line in new_lines if line not in initial_lines]
    
    # Join the new lines back into a single string, separated by newlines.
    return "\n".join(newly_added_lines)

#expensive operation
def element_diff(list1, list2):
    unique_elements = []
    for element in list2:
        if element not in list1:
            unique_elements.append(element)
    return unique_elements

#not in use
def element_common(list1, list2):
    unique_elements = []
    for element in list2:
        if element in list1:
            unique_elements.append(element)
    return unique_elements

def filter_links_from_website(start_url, list_of_links):
    return [link for link in list_of_links if link.startswith(start_url)]

"""---------Deal with interroption of the process--------"""
def read_and_split_last_line(filename):
    """
    Reads a file, separates the last line, and returns both parts.
    
    Returns:
        A tuple containing (all_lines_but_last, last_line).
        If the file is empty, returns ([], None).
    """
    try:
        # Open the file in read mode to get all lines.
        with open(filename, 'r') as file:
            lines = file.readlines()
        
        # Check if the file is not empty.
        if lines:
            # The last line is the last element of the list.
            last_line = lines[-1].strip()
            # All other lines are everything up to the last element.
            all_but_last_line = lines[:-1]
            return all_but_last_line, last_line
        else:
            return [], None
    except FileNotFoundError:
        print(f"Error: The file '{filename}' was not found.")
        return [], None

def write_number_visited_links_at_the_end(number_visited_links, filename):
    with open(filename, "a") as file:
        file.write(str(number_visited_links))
"""-------------Search functions------------------"""          
#not in use
def get_dynamic_elements(driver):
    elements_1 = driver.find_elements(By.XPATH, "//*")  # Get all elements
    
    time.sleep(2)  # Wait for dynamic changes (adjust based on site behavior)

    elements_2 = driver.find_elements(By.XPATH, "//*")  # Get all elements

    time.sleep(3)

    elements_3 = driver.find_elements(By.XPATH, "//*")  # Get all elements

    dynamic_elements = element_diff(elements_1, elements_2)
    dynamic_elements.append(element_diff(elements_1, elements_3))
    dynamic_elements.append(element_diff(elements_2, elements_3))

    return dynamic_elements

def mouse_hover(driver, original_text, original_clickable_elements):
    hover_elements = driver.find_elements(By.XPATH, "//*[contains(@onmouseover, '') or contains(@class, 'hover')]")
    print("entrou no hover, numero de hover elements: ", len(hover_elements))
    hover_text = ""
    hover_clickable_elements = []
    count = 0

    for element in hover_elements:
        if element.is_displayed() and element.is_enabled():
            try:
                #scrolling until the button
                driver.execute_script("arguments[0].scrollIntoView(true);", element)

                count += 1
                print("vai fazer o perform now ", count)
                ActionChains(driver).move_to_element(element).perform()
                time.sleep(2)
                hover_page_text = get_all_text(driver)
                #to get the added text with the mouse over
                print("previous text lenght ", len(original_text), " new text lenght ", len(hover_page_text))
                diff = list(difflib.ndiff(original_text.split(), hover_page_text.split()))
                print("DIFF:", diff)
                print("orginal text ", original_text)
                print("hover page text ", hover_page_text)
                #hover_text += [word[2:] for word in diff if word.startswith('+ ')]

                hover_page_clickable_elements = get_all_clickable_buttons(driver)
                print("hover page clickable elements ", len(hover_page_clickable_elements))
                hover_clickable_elements = element_diff(original_clickable_elements, hover_page_clickable_elements)
                print("actually new clickable", len(hover_clickable_elements))
                hover_clickable_elements.append(hover_clickable_elements)
            except Exception as e:
                print(e)
                continue
    return hover_text, hover_clickable_elements

def open_all_toggles(driver):
    toggles = driver.find_elements(By.XPATH, "//button[@aria-label='accordion']")

    initial_text = get_all_text(driver)
    initial_clickables = get_all_hrefs(driver)
    for toggle in toggles:
        try:
            if toggle.is_displayed() and toggle.is_enabled():
                driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", toggle)
                #print("devia estar no toggle")
                #time.sleep(0.1)
                # Wait until it's actually clickable
                toggle = WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.ID, toggle.get_attribute("id"))))
                toggle.click()
                time.sleep(0.1)
                new_text = get_all_text(driver)
                initial_text += get_new_content(initial_text, new_text)
                clickables = get_all_hrefs(driver)
                new_clickables = element_diff(initial_clickables, clickables)
                initial_clickables += new_clickables
        except:
            continue

    return initial_text, initial_clickables

"""-----------pos-scrapping functions-------------"""
def apply_filters(text):
    print("Initial text number characters:                    ", len(text))

    filtered_text = remove_end_of_page_info(text)
    print("After removing end of page number characters:      ", len(filtered_text))

    filtered_text = remove_product_info_regex(filtered_text)
    print("After removing product info number characters:     ", len(filtered_text))

    filtered_text = remove_influencers_tags(filtered_text)
    print("After removing influencers tags number characters: ", len(filtered_text))

    return filtered_text

def remove_product_info_ai(text):
    examples = ["""4.7
€45 with 40% Off
Daphne
Bracelet
€75
Add""",
"""New
€30 with 40% Off
Daphne Alice
Bracelet
€50
Add""",
"""Bestseller
€27 with 40% Off
Nassau
Bracelet
€44
Add"""]
    new_text = text
    chunk_size = 1000
    chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
    for chunk in chunks:
        new_chunk = openai_chat_completion(f"""Please remove products from the previous scrapped text.
                                      I want you to return just the text only removing a group of lines like these:
                                      {examples}""", chunk)
        new_text = new_text.replace(chunk, new_chunk)
    
    print("remove_product_info made the lenght go from this: ", len(text), " to ", len(new_text))
    return new_text

def optional_regex_line(regex):
    return "(?:" + regex + ")?"

def remove_product_info_regex(text):
    status_line = r"(?:New|Bestseller|Save on the Set|\d\.\d|50% Off|Out of stock)"
    pos_status_line = r"(?:Save on the Set|Game Day Glow|Overtime Ready|All-Weather Ready)"
    promotion_line = r"(?:€\d+(?:[ \t]+with[ \t]+\d+%[ \t]+Off)?|Final Sale)"
    category_line = r"(?:Bracelet|Set|Set 2|Watch|Anklet|Subscription Box|Huggies|Earrings|Ring|Rings|Necklace|Jewelry Case|[\w \t\-]*Bikini|Shorts|Dress|Silver|Extenders|Pendant|Choker)"
    name_line = r"[\w \t\-\&]+"
    other_price_line = r"€\d+"
    price_line = r"€\d+"
    add_line = r"(?:Add|Notify me)"
    composed_regex = ( "(?:" + #more normal structure
                    optional_regex_line(status_line + r"\s*") +
                    optional_regex_line(pos_status_line + r"\s*") +
                    optional_regex_line(promotion_line + r"\s*") +
                    name_line + r"\s*" +
                    category_line + r"\s*" +
                    optional_regex_line(other_price_line + r"\s*") +
                    price_line + r"\s*" +
                    add_line + r"\s*"
                    + "|" + #found this structure sometimes
                    name_line + r"\s*" +
                    price_line + r"\s*" +
                    promotion_line + r"\s*" +
                    add_line + r"\s*"
                    + ")"
                    )
    #regex = r"(?:(?:New|Bestseller|Save on the Set|\d\.\d|50% Off)?\s*)?(?:€\d+(?:[ \t]+with[ \t]+\d+%[ \t]+Off)?\s*)?[\w \t\-]+\s*(?:Bracelet|Set|Watch|Anklet|Subscription Box|Huggies|Earrings|Ring|Rings|Necklace|Jewelry Case|[\w \t\-]*Bikini|Shorts)\s*€\d+\s*(?:€\d+\s*)?Add\s*"
    #print("regex: ", regex)
    #print("composed regex: ", composed_regex)
    new_text = re.sub(composed_regex, "", text, flags=re.DOTALL)
    return new_text

def remove_end_of_page_info(text):
    pattern = """Free gift with 1st order
Join our newsletter to claim it
Product
Brand
Resources
Support
Join us
Social
EUR €
English
Terms of service
·
Privacy Policy"""

    # Use re.sub to remove the matched section
    cleaned_text = re.sub(pattern, "", text)

    return cleaned_text

#It can be dangerous if important information is starting with a @ for some reason
#It is also important for privacy matters
def remove_influencers_tags(text):
    print("Performing influencers tag removal, be careful as a new line starting with @ will be removed")
    pattern = r'^@.*\n'
    cleaned_text = re.sub(pattern, "", text, flags=re.MULTILINE)
    return cleaned_text

"""-------------Main functions-------------"""
def crawl(driver, start_url, number_links_visited, filename):
    """ Recursively crawls through all clickable elements and extracts text. """
    stack = [start_url]  # URLs to visit
    visited_urls = []
    text = ""

    links_count = 0

    while stack:
        url = stack.pop()
        
        if url in visited_urls:
            continue  # Skip already visited pages
        
        driver.get(url)
        #WebDriverWait(driver, 10).until(lambda d: d.execute_script("return document.readyState") == "complete")
        time.sleep(2)  # Wait for JS to load
        visited_urls.append(url)


        if number_links_visited > links_count:
            links_count += 1
            print("Skipping link ", links_count, " of ", number_links_visited, " : ", url)

        else:
            try:
                #print(f"\n--- Extracting from {url} ---")
                new_text, hrefs = open_all_toggles(driver)
            except:
                write_number_visited_links_at_the_end(links_count, filename)
            """
            try:      
                new_text = get_all_text(driver)
            except:
                write_number_visited_links_at_the_end(links_count, filename)
            """
            #print("new text: ", new_text)
            print("Count of visited URLs: ", len(visited_urls), " Current URL: ", url)
            with open(filename, 'a') as file:
                file.write(new_text + "\n")
            #text += "\n" + new_text
            links_count += 1
        
        """
        #I did this but understood I need to call this everytime I open a toggle
        try:
            clickable_elements = get_all_clickable_buttons(driver)
        except:
            write_number_visited_links_at_the_end(links_count, filename)
        """
        
        
        '''
        hover_text, hover_clickable_elements = mouse_hover(driver, text, clickable_elements)

        text += hover_text
        print(len(text))  # Print extracted text lenght

        clickable_elements.append(hover_clickable_elements)
'''
        #print(f"Length of stack before: {len(stack)}")
        print(f"Length of hrefs: {len(hrefs)}")
        #print("Actual clickable elements:")
        #print(*hrefs, sep="\n")

        for link in hrefs:
            if link and (link.startswith("https://store.peixefresco.com.pt/receitas")) and link not in visited_urls and link not in stack:
                stack.append(link)

        print(f"Length of stack after: {len(stack)}")
        print("Actual stack:")
        print(*stack, sep="\n")
    print(f"\nCrawling completed. Visited {len(visited_urls)} pages.")
    return text

def main():
    start_url = "https://store.peixefresco.com.pt/blog-de-receitas-peixe-fresco/"
    driver = setup_driver(start_url)

    #openai_client = get_openai_client()
    try:
            
        filename = "ingestion/peixefresco.txt"
        #filename = "ingestion/aux_text_to_test_filters.txt"
        """os.remove(filename)
        
        existing_text, number_links_visited = read_and_split_last_line(filename)
        if number_links_visited is None:
            number_links_visited = 0
        else:
            number_links_visited = int(number_links_visited)
            with open(filename, 'w') as file:
                file.writelines(existing_text)

        text = crawl(driver, start_url, number_links_visited, filename)
        """

        """
        # Process to filter the scraped text
        chunks = []
        with open(filename, 'r') as file:
            text = file.read()

        filtered_text = apply_filters(text)       

        # Write the filtered text to a new file
        filename = "ingestion/heyharper_helper_text_reading_and_get_links_at_all_toggles_regex_filtered.txt"
        with open(filename, 'w') as file:
            file.write(filtered_text)
        """
        """os.remove(filename)
        text = crawl(driver, start_url, 0, filename)

        """
        

        """
        text = ""
        
        open_all_toggles(driver)
        text += get_all_text(driver)
        #text += get_image_knowing_the_src(driver)
        print("scraping done")

        """
        #print(openai_chat_completion("From the text the user will give you understand if bracelets are sold or not by this website.", text))
        
        """
        #Create Chunks with batches
        filename = "ingestion/heyharper_helper_text_reading_and_get_links_at_all_toggles_regex_filtered.txt"
        with open(filename, 'r') as file:
            text = file.read()

        
        batches = create_batches_of_text(text, 10000, 100)
        chunks = []
        for batch in batches:
            chunks += get_text_chunks(batch)
        """

        with open(filename, 'r') as file:
            text = file.read()
        batches = create_batches_of_text(text, 10000, 100)
        chunks = []
        for batch in batches:
            chunks += get_text_chunks(batch)
        #chunks = get_text_chunks(text)
        #print(*chunks, sep="\n")
        print("number of chunks: ", len(chunks))
        

        """
        #chunksToKeep = manual_chunks_filter(chunks, text)
        #print("chunks to keep: \n", chunksToKeep)
        #vectors=get_points(chunksToKeep)
        #insert_data(vectors)

        """
        return chunks
        
    finally:
        # Close the browser
        driver.quit()
        pass


if __name__ == '__main__':
    main()