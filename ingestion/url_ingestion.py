import base64
import os
import requests
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains

import difflib
import time

from llms.openai_utils import get_openai_client, openai_chat_completion
from chatbot import make_conversation_file
from vectorization import get_points, get_text_chunks, insert_data


def setup_driver(start_url):
    # Initialize Selenium WebDriver (Make sure you have ChromeDriver installed)
    driver = webdriver.Chrome()
    driver.get(start_url)
    time.sleep(2)  # Allow JS to load
    return driver


def get_product_names(driver):
    # Find product names
    products = driver.find_elements(By.XPATH, '//a[@aria-label="Product name"]')

    # Print each product title
    for product in products:
        print(product.text)


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




#expensive operation
def element_diff(list1, list2):
    unique_elements = []
    for element in list2:
        if element not in list1:
            unique_elements.append(element)
    return unique_elements

def element_common(list1, list2):
    unique_elements = []
    for element in list2:
        if element in list1:
            unique_elements.append(element)
    return unique_elements


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

def get_all_text(driver):
    # Get all text that is visible
    visible_text = driver.find_element(By.TAG_NAME, "body").text
    #print("visible data", visible_text)
    return visible_text

def clean_page_text(text):
    return 

def ask_questions(text, questions):
    """ Receives a list of questions and returns a list of answers. """
    openai_client = get_openai_client()
    answers = []
    for question in questions:
        response = openai_chat_completion("""Use the provided text to answer the question, notice that the text was generated from a website scrapping process.
                                          If it is a yes or not questions only answer with Yes, No or IDontKnow.
                                          If the provided text does not contain the answer, do not explain that the text does have the answer just answer IDontKnow.
                                          If it is a question to list something, just name the items.
                                          Overall try to be brieve, unless there is question on a process, in that case, explain all the steps""", "Question: " + question + "\nContext: " + text)
        answers.append(response)
    return answers

def crawl(driver, start_url, number_links_visited, filename):
    """ Recursively crawls through all clickable elements and extracts text. """
    stack = [start_url]  # URLs to visit
    visited_urls = []
    text = ""

    links_count = 0

    while stack:
        #if links_count == 10:
        #    break
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
            if link and (link.startswith(start_url) or link.startswith("https://checkout-eu.heyharper.com/")) and not link.startswith("https://heyharper.com/eu/en/products/") and link not in visited_urls and link not in stack:
                stack.append(link)
        print(f"Length of stack after: {len(stack)}")
        print("Actual stack:")
        print(*stack, sep="\n")
    print(f"\nCrawling completed. Visited {len(visited_urls)} pages.")
    return text

def add_context(chunk, text):
    openai_client = get_openai_client()
    response = openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": f"""I need you to rewrite this preposition: {chunk}. 
                           In order to include a bit more context so that will better mach when used for embeddings
                           The text is: {text}
                           Note, I don't want to include any more information than the one that makes it so you do not need to know anything else to understand it.
                           Please answer only the rewritten sentence.
                           Example:
                           if this was the preposition: "Options are: Minimalist, Trendy, or Surprise Me."
                           your answer should be something like this: "For the product subscription, customers can choose the style of jewelry pieces they want: Minimalist, Trendy, or Surprise Me."
                           This is a good example because it includes the context that we are refering to the product subscription, we refer who has the option to choose, and we use words that are used on the rest of the text, like style instead of options"""}
                          ]
    )
    return response.choices[0].message.content

def manual_chunks_filter(chunks, text):
    chunksToKeep = []
    for chunk in chunks:
        print("\nchunk: ", chunk, "\n")
        toKeep = input("""Do you want to keep this chunk as it is? if Yes type y
        If it is too summarized and needs context, type 1
        If you want to rewrite it yourself, type r""")
        if toKeep == "y":
            chunksToKeep.append(chunk)
        elif toKeep == "1":
            new_chunk = add_context(chunk, text)
            print("new chunk: ", new_chunk)
            toKeep = input("Write y to include like this, b to include it as before and r to rewrite it yourself")
            if toKeep == "y":
                chunksToKeep.append(new_chunk)
            elif toKeep == "b":
                chunksToKeep.append(chunk)
        elif toKeep == "r":
            new_chunk = input("Write the new chunk: ")
            chunksToKeep.append(new_chunk)
    return chunksToKeep

def filter_links_from_website(start_url, list_of_links):
    return [link for link in list_of_links if link.startswith(start_url)]

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

def scrape_page(url):
    time.sleep(0.1)
    driver = setup_driver(url)
    try:
        text = get_all_text(driver)
        return text
    finally:
        driver.quit()

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

def main():
    start_url = "https://heyharper.com/eu/en"
    driver = setup_driver(start_url)

    openai_client = get_openai_client()
    try:
            
        filename = "ingestion/heyharper_helper_text_reading_and_get_links_at_all_toggles.txt"
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
        with open(filename, 'r') as file:
            text = file.read()


        questions = ["What products do you sell?",
"Do you sell bracelets?",
"Do you sell rings?",
"Do you sell dresses?",
"Apart  from products rings, bracelets, earrings, what other products do you offer?",
"What payment methods do you offer?",
"Is it possible to pay with VISA?",
"Is it possible to pay with multiple credit cards?",
"Apart from VISA, AMEX and Pay Pal, what other payment methods do you offer?",
"Explain the domestic return delivery process including details of the time and cost.",
"What is the fee for domestic return delivery?",
"Explain the international return delivery process including details of the time and cost.",
"What is the fee for international return delivery?",
"What is the process to return a product?",
"Which products, if any, require a different return process?"]
        
        answers = ask_questions(text, questions)

        formatted_lines = [
    f"Q: {question}\nA: {answer}\n" for question, answer in zip(questions, answers)
]
        formatted_output = "\n".join(formatted_lines)

        make_conversation_file(formatted_output, "conversation_logs/onboarding")
        print(formatted_output)

        """
        text = ""
        
        open_all_toggles(driver)
        text += get_all_text(driver)
        #text += get_image_knowing_the_src(driver)
        print("scraping done")

        """
        
        print(openai_chat_completion("From the text the user will give you understand if bracelets are sold or not by this website.", text))
        
        """chunks=get_text_chunks(text)

        print("chunks: \n", chunks)

        chunksToKeep = manual_chunks_filter(chunks, text)

        print("chunks to keep: \n", chunksToKeep)
        #chunksToKeep = ['Hey Harper offers monthly surprise jewelry subscriptions.', 'Customers can choose the style of jewelry pieces they want: Minimalist, Trendy, or Surprise Me.', 'Customers can subscribe and save money by paying monthly.', 'By paying monthly, customers can save 50%.', 'Hey Harper offers free delivery from March 17th to 24th.', 'The subscription costs $30 and can be cancelled or paused anytime without commitments.', "The jewelry pieces are either from Hey Harper's core collection or upcoming new drops.", "Each jewelry piece is chosen by Hey Harper's design team.", 'Customers can choose their style, add their address, and the first piece ships immediately.', 'The subscription service is available only to the USA and Canada.', 'Images displayed are examples of jewelry pieces.', 'Monthly subscription pieces ship monthly on the same date as the first order to the given address.', 'Prepaid subscriptions ship on the first week of each month to the given address.', 'Customers receive a confirmation email with tracking information for each new shipment.', 'Customers can cancel or pause their subscription anytime, easily and for free.', 'All subscription pieces are non-refundable and non-exchangeable.', "Hey Harper's jewelry is made from stainless steel metal with 14K gold PVD coating.", 'The 14K gold PVD coating is durable and waterproof.', 'Hey Harper offers a lifetime color warranty for their jewelry.', "Hey Harper's waterproof jewelry is designed to endure daily routines, including showering, working out, and swimming.", 'If the jewelry loses color, customers can contact customer support with a visible picture of their item to claim the warranty.', 'Hey Harper offers a monthly surprise jewelry piece at a fraction of the price.', 'Customers can pay monthly and cancel anytime easily and for free.', 'Customers can select their subscription style.']
        #chunksToKeep = ['Please note that the Heart Jewelry Box is not included.', 'To ensure perfect fit, we are only offering necklaces and earrings. ']

        
        vectors=get_points(chunksToKeep)
        #insert_data(vectors)
        """

        """
        chunks=get_text_chunks(text_2)
        vectors=get_embedding(chunks)
        insert_data(vectors)
        """

    finally:
        # Close the browser
        driver.quit()


if __name__ == '__main__':
    main()