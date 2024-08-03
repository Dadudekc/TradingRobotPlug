import os
import logging
import time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import WebDriverException, TimeoutException
from PIL import Image
import openai

# User-friendly introduction
print("Welcome to the HTML Screenshot and Explanation Script!")
print("This script will help you capture screenshots of HTML files and explain them using AI.")
print("Let's get started!\n")

# Ask user for paths and API key interactively (or use defaults)
chrome_driver_path = input("Please enter the full path to your ChromeDriver executable (e.g., /path/to/chromedriver): ")
html_files_directory = input("Please enter the directory where your HTML files are stored: ")
base_output_directory = input("Please enter the base directory where you want to store outputs: ")
openai_api_key = input("Please enter your OpenAI API key: ")

# Set up logging
logging.basicConfig(
    filename="process_log.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Set up OpenAI
openai.api_key = openai_api_key

def setup_driver():
    """
    Initializes the Selenium WebDriver with headless Chrome options.
    This is necessary for capturing screenshots of HTML files without opening a browser window.
    """
    chrome_options = Options()
    chrome_options.add_argument("--headless")  # Runs Chrome in headless mode.
    chrome_options.add_argument("--disable-gpu")  # Disables GPU hardware acceleration.
    chrome_options.add_argument("--window-size=1920,1080")  # Sets the window size for consistent screenshots.

    # Initialize the WebDriver service with the path to the ChromeDriver executable.
    service = Service(executable_path=chrome_driver_path)
    
    try:
        driver = webdriver.Chrome(service=service, options=chrome_options)
        logging.info("WebDriver initialized successfully.")
        print("WebDriver initialized successfully. We are ready to start capturing screenshots!")
        return driver
    except WebDriverException as e:
        logging.critical(f"Failed to initialize WebDriver: {str(e)}")
        print("There was an error initializing the WebDriver. Please check the ChromeDriver path and try again.")
        raise

def capture_screenshot(driver, html_file_path, output_image_path):
    """
    Captures a screenshot of an HTML file using the WebDriver.
    """
    try:
        driver.get(f"file:///{html_file_path}")
        time.sleep(2)  # Waits for the page to fully load before taking a screenshot.
        driver.save_screenshot(output_image_path)
        logging.info(f"Screenshot captured: {output_image_path}")
        print(f"Screenshot captured successfully: {output_image_path}")
        return output_image_path
    except (WebDriverException, TimeoutException) as e:
        logging.error(f"Error capturing screenshot for {html_file_path}: {str(e)}")
        print(f"Sorry, something went wrong while trying to capture a screenshot of {html_file_path}. Please try again.")
        return None

def process_image_with_chatgpt(image_path):
    """
    Sends the screenshot to OpenAI's API to get an explanation of the content.
    """
    try:
        with open(image_path, "rb") as image_file:
            image_data = image_file.read()

        # Sending the image to OpenAI for a detailed explanation
        response = openai.Image.create(
            prompt="Describe the content of this screenshot in simple terms:",
            images=image_data,
            n=1,
            size="1024x1024"
        )

        explanation_text = response['choices'][0]['text']
        logging.info(f"Image processed successfully: {image_path}")
        print("Screenshot processed successfully. Explanation generated.")
        return explanation_text
    except openai.error.OpenAIError as e:
        logging.error(f"Error processing image with OpenAI: {str(e)}")
        print(f"Sorry, there was an error processing the screenshot with OpenAI. Please check your API key or try again later.")
        return None

def save_explanation_to_file(explanation, output_path):
    """
    Saves the AI-generated explanation to a text file.
    """
    try:
        with open(output_path, "w") as file:
            file.write(explanation)
        logging.info(f"Explanation saved: {output_path}")
        print(f"Explanation saved successfully to {output_path}")
    except IOError as e:
        logging.error(f"Error saving explanation to file: {str(e)}")
        print(f"Sorry, we couldn't save the explanation to {output_path}. Please check the directory and try again.")

def guide_user_on_next_steps(explanation, html_filename):
    """
    Provides dynamic guidance to the user on what to do next with the generated explanation.
    """
    print("\nNext Steps:")
    print(f"1. Review the generated explanation for {html_filename}.")
    print(f"2. If the explanation provides insights or information you need, consider the following:")
    print("   - If you are analyzing data, use the explanation to understand the content of the HTML page.")
    print("   - If you are building a model, use this explanation as part of your training data.")
    print("   - If you want to refine the explanations, consider adding more details to the prompt or tweaking the HTML content.")
    print("3. Save and organize these explanations if they are useful for your project or research.")
    print("4. If you want to process more HTML files, you can run this script again.")
    print("5. For further analysis or model training, you might consider:")
    print("   - Aggregating multiple explanations into a dataset.")
    print("   - Using these explanations as input to another model for more complex tasks.")
    print("   - Sharing this data with team members or using it in presentations.")
    print("\nFeel free to explore and experiment with the outputs!")

def compare_and_update_best_explanation(current_best, new_explanation):
    """
    Compares the new explanation with the current best one and updates if the new one is better.
    Currently, it uses the length of the explanation as a proxy for quality.
    """
    if current_best is None or len(new_explanation) > len(current_best):
        return new_explanation
    return current_best

def main():
    # Define paths for organized storage
    screenshots_directory = os.path.join(base_output_directory, "screenshots")
    explanations_directory = os.path.join(base_output_directory, "explanations")
    best_explanations_directory = os.path.join(base_output_directory, "best_explanations")
    
    # Ensure the output directories exist, or create them
    os.makedirs(screenshots_directory, exist_ok=True)
    os.makedirs(explanations_directory, exist_ok=True)
    os.makedirs(best_explanations_directory, exist_ok=True)

    # Initialize the WebDriver
    try:
        driver = setup_driver()
    except Exception as e:
        print("The program will now exit due to a critical error.")
        return

    current_best_explanation = None
    best_html_filename = None

    # Process each HTML file in the directory
    for filename in os.listdir(html_files_directory):
        if filename.endswith(".html"):
            print(f"\nProcessing file: {filename}")
            html_file_path = os.path.join(html_files_directory, filename)
            output_image_path = os.path.join(screenshots_directory, filename.replace(".html", ".png"))
            explanation_output_path = os.path.join(explanations_directory, filename.replace(".html", ".txt"))

            screenshot_path = capture_screenshot(driver, html_file_path, output_image_path)

            if screenshot_path:
                explanation = process_image_with_chatgpt(screenshot_path)
                if explanation:
                    save_explanation_to_file(explanation, explanation_output_path)
                    current_best_explanation = compare_and_update_best_explanation(current_best_explanation, explanation)
                    if current_best_explanation == explanation:
                        best_html_filename = filename
                    guide_user_on_next_steps(explanation, filename)
                else:
                    print(f"Skipping saving the explanation for {filename} due to processing failure.")
            else:
                print(f"Skipping {filename} due to screenshot failure.")

    # Save the best explanation of the session
    if current_best_explanation and best_html_filename:
        best_explanation_output_path = os.path.join(best_explanations_directory, best_html_filename.replace(".html", ".txt"))
        save_explanation_to_file(current_best_explanation, best_explanation_output_path)
        print(f"\nThe best explanation of this session was saved to {best_explanation_output_path}.")

    # Clean up and close the WebDriver
    try:
        driver.quit()
        logging.info("WebDriver closed successfully.")
        print("Process completed successfully! WebDriver has been closed.")
    except WebDriverException as e:
        logging.warning(f"Error while closing WebDriver: {str(e)}")
        print("The WebDriver encountered an issue while closing, but the process is complete.")

if __name__ == "__main__":
    try:
        main()
