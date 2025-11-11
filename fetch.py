import requests

def fetch_headlines():
    url = "https://api.gdeltproject.org/api/v2/doc/doc?query=india&mode=ArtList&format=json&maxrecords=5"

    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        print(data)
    else:
        print("Error:", response.status_code, response.text)

if __name__ == "__main__":
    fetch_headlines()
