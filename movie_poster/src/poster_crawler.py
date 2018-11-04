import os
import pandas as pd
from urllib.request import urlopen
from urllib.error import HTTPError
from pandas import read_csv
from pathlib import Path

def read_data():
    df = pd.read_csv(python_platform_path+'/data/MovieGenre.csv', encoding="ISO-8859-1", usecols=['imdbId', 'Poster'], keep_default_na=False)
    print(df.head())
    return df

def download(imdb_id, url):
    path = Path(f"{download_path}/{imdb_id}.jpg")

    # Skip this download if the file already exists
    # so you can run this in several runs if necessary.
    if path.is_file():
        print("Skipping", str(path), "as it already exists")

    # Not every movie has a poster or a valid URL for the poster.
    elif not url.startswith("http"):
        print(f"Skipping invalid url: '{url}'")

    else:
        try:
            response = urlopen(url)
            with path.open("wb") as f:
                f.write(response.read())
        except HTTPError:
            print(f"Error downloading {url}")


if __name__ == "__main__":
    global python_platform_path
    python_platform_path = os.path.abspath(__file__ + "/../")
    movies = read_data()
    global download_path
    download_path =python_platform_path+ "/MoviePosters"
    for index, row in movies.iterrows():
        print(f"Downloading {index} {row['Poster']} ...")
        download(row['imdbId'], row['Poster'])