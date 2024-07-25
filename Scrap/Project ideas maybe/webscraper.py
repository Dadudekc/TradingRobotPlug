import requests
from bs4 import BeautifulSoup
import re

import requests
from bs4 import BeautifulSoup
import re

def fetch_comments(url):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        raise Exception(f"Failed to fetch page: {response.status_code}")
    
    soup = BeautifulSoup(response.content, 'html.parser')
    comments = soup.find_all('div', class_='comment-content')  # Adjust this based on the actual HTML structure
    return comments

def extract_values(comments):
    values = []
    for comment in comments:
        text = comment.get_text()
        matches = re.findall(r'\$\d+|\d+', text)
        for match in matches:
            value = match.replace('$', '')
            values.append(float(value))
    return values

def calculate_average(values):
    if not values:
        return 0
    return sum(values) / len(values)

def main():
    url = "YOUR_URL_HERE"
    comments = fetch_comments(url)
    values = extract_values(comments)
    average_value = calculate_average(values)
    print(f"The average value of Tesla mentioned in the comments is: ${average_value:.2f}")

if __name__ == "__main__":
    main()


def extract_values(comments):
    values = []
    for comment in comments:
        text = comment.get_text()
        matches = re.findall(r'\$\d+|\d+', text)
        for match in matches:
            value = match.replace('$', '')
            values.append(float(value))
    return values

def calculate_average(values):
    if not values:
        return 0
    return sum(values) / len(values)

def main():
    url = "https://stocktwits.com/Stocktwits/message/580425211"
    comments = fetch_comments(url)
    values = extract_values(comments)
    average_value = calculate_average(values)
    print(f"The average value of Tesla mentioned in the comments is: ${average_value:.2f}")

if __name__ == "__main__":
    main()
