import os
import time
import pandas as pd
import asyncio
import aiohttp

from pathlib import Path
from asyncio.exceptions import TimeoutError
from aiohttp.client_exceptions import ClientOSError
from sources.utils.url_utils import FILENAME_PATTERN, URL_PATTERN


async def download_by_url(session, url: str, file_name: str, save_to: Path):
    save_to = Path(save_to)

    async with session.get(url) as response:
        if response.status == 200:
            img = await response.read()
            with open(save_to / file_name, "wb") as f:
                f.write(img)
                print(f"Done {file_name}")
        else:
            print(f"Error at url: {url}\nfile_name: {file_name}\n")


def _get_filenames_from_df(df):
    file_names = []
    for row in df.iterrows():
        data = row[1]

        url1 = data["image_url1"]
        url2 = data["image_url2"]

        file_name1 = FILENAME_PATTERN.search(url1).group(1)
        file_name2 = FILENAME_PATTERN.search(url2).group(1)

        file_names.append(file_name1)
        file_names.append(file_name2)

    return file_names


async def download_from_csv(csv_path, download_path):
    csv_path = Path(csv_path)
    download_path = Path(download_path)
    df = pd.read_csv(csv_path)

    file_names = set(_get_filenames_from_df(df))
    already_downloaded = os.listdir(download_path)

    file_names = file_names.difference(already_downloaded)

    async with aiohttp.ClientSession() as session:
        tasks = [
            download_by_url(session, URL_PATTERN(file_name), file_name, download_path)
            for file_name in file_names
        ]
        await asyncio.gather(*tasks)


if __name__ == "__main__":
    while True:
        try:
            asyncio.run(download_from_csv("../../data/test.csv", "../../data/images"))
        except (TimeoutError, ClientOSError):
            time.sleep(30)
