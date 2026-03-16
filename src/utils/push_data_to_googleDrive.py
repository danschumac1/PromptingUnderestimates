from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from tqdm import tqdm
import pickle
import os

SCOPES = ['https://www.googleapis.com/auth/drive.file']

FOLDER_ID = "16iGDAkaZzzUY9V1OjbsyOuTyVDaSrAqq"
name = "features_llama.zip"
FILE_PATH = name


def authenticate():
    creds = None

    if os.path.exists("token.pickle"):
        with open("token.pickle", "rb") as f:
            creds = pickle.load(f)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                "credentials.json",
                SCOPES
            )
            creds = flow.run_local_server(port=0)

        with open("token.pickle", "wb") as f:
            pickle.dump(creds, f)

    return creds


def upload_zip():
    creds = authenticate()
    service = build("drive", "v3", credentials=creds)

    file_metadata = {
        "name": name,
        "parents": [FOLDER_ID]
    }

    media = MediaFileUpload(
        FILE_PATH,
        mimetype="application/zip",
        resumable=True,
        chunksize=1024 * 1024  # 1MB chunks
    )

    request = service.files().create(
        body=file_metadata,
        media_body=media,
        fields="id"
    )

    file_size = os.path.getsize(FILE_PATH)
    progress_bar = tqdm(total=file_size, unit="B", unit_scale=True, desc="Uploading")

    response = None
    while response is None:
        status, response = request.next_chunk()
        if status:
            progress_bar.n = int(status.resumable_progress)
            progress_bar.refresh()

    progress_bar.close()

    print("\n✅ Upload successful")
    print("File ID:", response["id"])


if __name__ == "__main__":
    upload_zip()
