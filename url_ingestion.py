from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains

import difflib
import time

from openai_utils import get_openai_client
from vectorization import get_text_chunks, get_embedding, insert_data


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
    print("entrou no get_all_clickable")
    clickables = driver.find_elements(By.CSS_SELECTOR, "a, button, [role='button'], [onclick], [tabindex='0'], [href]")
    really_clickable = []
    for element in clickables:
        try:
            if element.is_displayed() and element.is_enabled():
                #element.click()
                really_clickable.append(element)
                #print(element.text, "is clickable!")
                #time.sleep(3)
        except Exception as e:
            # risky code probably
            if "element not interactable" in str(e).lower():
                print("Not interactable\nDisplayed: ", element.is_displayed(), " Enabled: ", element.is_enabled())
    
    return really_clickable


def get_all_product_urls(driver):
    print("entrou no get_all_product_urls")
    product_elements = driver.find_elements(By.XPATH, "//a[starts-with(@href, '/eu/en/products/')]")

    products_urls = []
    for element in product_elements:
        try:
            product_url = element.get_attribute("href")
            if product_url and product_url.startswith("https://heyharper.com/eu/en/products/")and product_url not in products_urls:
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

def open_all_toggles(driver):
    toggles = driver.find_elements(By.XPATH, "//button[@aria-label='accordion']")

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
        except:
            continue

def get_all_text(driver):
    # Get all text that is visible
    visible_text = driver.find_element(By.TAG_NAME, "body").text
    #print("visible data", visible_text)
    return visible_text

def clean_page_text(text):
    return 

def crawl(driver, start_url):
    """ Recursively crawls through all clickable elements and extracts text. """
    stack = [start_url]  # URLs to visit
    visited_urls = []
    text = ""

    while stack:
        url = stack.pop()
        
        if url in visited_urls:
            continue  # Skip already visited pages
        
        driver.get(url)
        #WebDriverWait(driver, 10).until(lambda d: d.execute_script("return document.readyState") == "complete")
        time.sleep(2)  # Wait for JS to load
        visited_urls.append(url)

        #print(f"\n--- Extracting from {url} ---")

        text += "\n" + get_all_text(driver)
        clickable_elements = get_all_clickable_buttons(driver)
        
        '''
        hover_text, hover_clickable_elements = mouse_hover(driver, text, clickable_elements)

        text += hover_text
        print(len(text))  # Print extracted text lenght

        clickable_elements.append(hover_clickable_elements)
'''

        open_all_toggles(driver)

        for element in clickable_elements:
            try:
                # Use JavaScript to get the href without clicking
                link = element.get_attribute("href")
                if link and link.startswith("http") and link not in visited_urls:
                    stack.append(link)
            except:
                continue  # Ignore elements that fail

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
        print("chunk: ", chunk)
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

def list_of_chunks_to_numbered_string(chunks):
    string = ""
    for chunk_ix, chunk in enumerate(chunks):
        #single_chunk_string = f"""Chunk ({chunk['chunk_id']}): {chunk['title']}\nSummary: {chunk['summary']}\n\n"""
        single_chunk_string = f"""Chunk ({chunk_ix}): {chunk}\n\n"""
        string += single_chunk_string
    return string

def main():
    start_url = "https://heyharper.com/us/en/products/surprise-jewelry-subscription-box"
    driver = setup_driver(start_url)

    openai_client = get_openai_client()
    try:
        
        text = ""
        
        open_all_toggles(driver)
        text += get_all_text(driver)
        print("scraping done")

        chunks=get_text_chunks(text)

        chunksToKeep = manual_chunks_filter(chunks, text)

        print("chunks to keep: \n", chunksToKeep)
        #chunksToKeep = ['Hey Harper offers monthly surprise jewelry subscriptions.', 'Customers can choose the style of jewelry pieces they want: Minimalist, Trendy, or Surprise Me.', 'Customers can subscribe and save money by paying monthly.', 'By paying monthly, customers can save 50%.', 'Hey Harper offers free delivery from March 17th to 24th.', 'The subscription costs $30 and can be cancelled or paused anytime without commitments.', "The jewelry pieces are either from Hey Harper's core collection or upcoming new drops.", "Each jewelry piece is chosen by Hey Harper's design team.", 'Customers can choose their style, add their address, and the first piece ships immediately.', 'The subscription service is available only to the USA and Canada.', 'Images displayed are examples of jewelry pieces.', 'Monthly subscription pieces ship monthly on the same date as the first order to the given address.', 'Prepaid subscriptions ship on the first week of each month to the given address.', 'Customers receive a confirmation email with tracking information for each new shipment.', 'Customers can cancel or pause their subscription anytime, easily and for free.', 'All subscription pieces are non-refundable and non-exchangeable.', "Hey Harper's jewelry is made from stainless steel metal with 14K gold PVD coating.", 'The 14K gold PVD coating is durable and waterproof.', 'Hey Harper offers a lifetime color warranty for their jewelry.', "Hey Harper's waterproof jewelry is designed to endure daily routines, including showering, working out, and swimming.", 'If the jewelry loses color, customers can contact customer support with a visible picture of their item to claim the warranty.', 'Hey Harper offers a monthly surprise jewelry piece at a fraction of the price.', 'Customers can pay monthly and cancel anytime easily and for free.', 'Customers can select their subscription style.']
        #chunksToKeep = ['Please note that the Heart Jewelry Box is not included.', 'To ensure perfect fit, we are only offering necklaces and earrings. ']

        #assuming there is at least one chunk
        groupedChunks = [chunksToKeep[0]]

        for chunk in chunksToKeep[1:]:
            
            string_of_chunks = list_of_chunks_to_numbered_string(groupedChunks)
            # TO-DO add the response format to also include an explanation
            completion = openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": f"""You are classifier which porpuse is to group prepositions together depending on their meaning.
                           You will receive a list of groups of prepositions and a new one.
                           Each group will be identified by a number.
                           Please answer onlywith the number of the group in which the new preposition fits well, or -1 if it doesn't fit well in any of them.
                           I rreally just want you to answer with a umber for me to be able to conver it to an integer on my code. Examples of answers:
                           2
                           -1
                           The groups are:
                           {string_of_chunks}
                           New preposition: {chunk}"""}
                          ]
            )
            response = completion.choices[0].message.content
            print("grouped chunks: ", string_of_chunks)
            print("chunk: ", chunk)
            print("response: ", response)
            agreed = input("agree?")
            if agreed == "y":
                if response == "-1":
                    groupedChunks.append(chunk)
                else:
                    groupedChunks[int(response)] += "\n" + chunk
            else:
                user_response = input("then what?")
                if user_response == "-1":
                    groupedChunks.append(chunk)
                else:
                    groupedChunks[int(user_response)] += "\n" + chunk

        print("final chunks: ", groupedChunks)
        vectors=get_embedding(groupedChunks)
        insert_data(vectors)

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