import os
import re
import pandas as pd
import asyncio
import aiohttp

from pathlib import Path


ROOT_DIRECTORY = Path(os.path.dirname(os.path.abspath(__file__)))
FILENAME_PATTERN = re.compile(r"(\d{9}.\w+)")
URL_PATTERN = "https://storage.googleapis.com/lun-ua/images/{}".format


async def download_by_url(session, url: str, file_name: str, save_to: Path):
    save_to = Path(save_to)

    async with session.get(url) as response:
        if response.status == 200:
            img = await response.read()
            with open(save_to / file_name, "wb") as f:
                f.write(img)
        else:
            print(f"Error at url: {url}\nfile_name: {file_name}\n")


def _get_filenames(dataframe: pd.DataFrame):
    file_names = []
    for row in dataframe.iterrows():
        data = row[1]

        url1 = data["image_url1"]
        url2 = data["image_url2"]
        
        file_name1 = FILENAME_PATTERN.search(url1).group(1)
        file_name2 = FILENAME_PATTERN.search(url2).group(1)

        file_names.append(file_name1)
        file_names.append(file_name2)
    
    return file_names


async def download_from_csv(csv_path: str, download_path: str):
    csv_path = Path(csv_path)
    download_path = Path(download_path)
    os.makedirs(download_path, exist_ok=True)
    df = pd.read_csv(csv_path)

    file_names = set(_get_filenames(df))
    already_downloaded = os.listdir(download_path)

    file_names = file_names.difference(already_downloaded)

    async with aiohttp.ClientSession() as session:
        tasks = [
            download_by_url(session, URL_PATTERN(file_name), file_name, download_path) 
            for file_name in file_names
        ]
        await asyncio.gather(*tasks)


if __name__ == "__main__":
    asyncio.run(download_from_csv("../data/test.csv", "../data/images"))
