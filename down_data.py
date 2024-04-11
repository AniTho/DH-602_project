import base64
import requests
import json
import urllib.request
import shutil
from pathlib import Path
from tqdm import tqdm 



# Encode our credentials then convert it to a string.
credentials = base64.b64encode(b'aniketthomas27:ANDroid@1234!!').decode('utf-8')

# Create the headers we will be using for all requests.
headers = {
    'Authorization': 'Basic ' + credentials,
    'User-Agent': 'Example Client',
    'Accept': 'application/json'
}

# Send Http request
response = requests.get('https://nda.nih.gov/api/package/auth', headers=headers)

# Business Logic.

# If the response status code does not equal 200
# throw an exception up.
if response.status_code != requests.codes.ok:
    print('failed to authenticate')
    response.raise_for_status()
else:
    print('Connected')

# Assume code in authentication section is present.

packageId = 113460

# Construct the request to get the files of package 1234
# URL structure is: https://nda.nih.gov/api/package/{packageId}/files
response = requests.get('https://nda.nih.gov/api/package/' + str(packageId) + '/files', headers=headers)

# Get the results array from the json response.
results = response.json()['results']

# Business Logic.

files = {}

# Add important file data to the files dictionary.
for f in results:
    files[f['package_file_id']] = {'name': f['download_alias']}




# Assume code in authentication section is present.
# Assume that one of the retrieving files implementations is present too

# Create a post request to the batch generate presigned urls endpoint.
response = requests.post('https://nda.nih.gov/api/package/' + str(packageId) + '/files/batchGeneratePresignedUrls', json=list(files.keys()), headers=headers)

# Get the presigned urls from the response.
results = response.json()['presignedUrls']

# Business Logic.

# Add a download key to the file's data.
for url in results:
    files[url['package_file_id']]['download'] = url['downloadURL']



# Iterate on file id and its data to perform the downloads.
for file_id, data in tqdm(files.items(), desc="Downloading files"):
    name = data['name']
    downloadUrl = data['download']
    # Create a downloads directory
    file = 'downloads/' + name
    # Strip out the file's name for creating non-existent directories
    directory = file[:file.rfind('/')]
    
    # Create non-existent directories, package files have their
    # own directory structure, and this will ensure that it is
    # kept intact when downloading.
    Path(directory).mkdir(parents=True, exist_ok=True)
    
    # Initiate the download.
    with urllib.request.urlopen(downloadUrl) as dl:
        total_size = int(dl.headers.get('content-length', 0))
        with open(file, 'wb') as out_file:
            # Initialize the progress bar for this file
            with tqdm(total=total_size, desc=name, unit='iB', unit_scale=True, unit_divisor=1024, leave=False) as bar:
                while True:
                    buffer = dl.read(1024*1024)  # Read chunks of 1 MB
                    if not buffer:
                        break
                    out_file.write(buffer)
                    bar.update(len(buffer))

print('Done.')
