import io
import os
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from google.oauth2 import service_account

# Load credentials
credentials = service_account.Credentials.from_service_account_file('credentials.json')

# Build the Google Drive API client
drive_service = build('drive', 'v3', credentials=credentials)

# Folder ID from the URL you provided
folder_id = '1SwSgrCsWKMug5G9UJibBFiAHEbfY_LH8'

# Destination folder path where you want to save the downloaded files
destination_folder = 'lol'

# Retrieve the list of files in the folder
results = drive_service.files().list(q=f"'{folder_id}' in parents", fields="files(id, name)").execute()
files = results.get('files', [])

# Download each file in the folder
for file in files:
    file_id = file['id']
    file_name = file['name']
    destination_path = os.path.join(destination_folder, file_name)

    request = drive_service.files().get_media(fileId=file_id)
    fh = io.FileIO(destination_path, 'wb')
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while done is False:
        status, done = downloader.next_chunk()
        print(f'Download progress of {file_name}: {int(status.progress() * 100)}%')
    print(f'Download of {file_name} complete!')
