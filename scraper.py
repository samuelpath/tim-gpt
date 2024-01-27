import requests
from bs4 import BeautifulSoup
import os
import episodes_metadata

# Specify the directory for the output files
output_dir = 'data'

# Create the directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Iterate over each object in the array
for episode in episodes_metadata:
    # Send a GET request to the URL
    response = requests.get(episode['url'])

    # Parse the response content with BeautifulSoup
    soup = BeautifulSoup(response.content, 'html.parser')

    # Retrieve the content of "document.getElementsByClassName('entry-content')[0].innerText"
    content = soup.find(class_='entry-content').get_text()

    # Create a filename using the id and title, truncate if necessary
    title = episode['title'][:240] + '...' if len(episode['title']) > 240 else episode['title']
    filename = f"{episode['id']} - {title}.txt"

    # Replace any characters in the filename that are not allowed in file names with a space
    filename = "".join(i if i not in r'\/:*?"<>|' else ' ' for i in filename)

    # Write the content to a file in the specified directory
    with open(os.path.join(output_dir, filename), 'w', encoding='utf-8') as f:
        f.write(content)
