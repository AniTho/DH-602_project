import os
import pathlib
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut

def dicom_validity(path): # In path variable provide path to X-ray image
    path=path.rstrip('_1x1.jpg')+'/001'
    file_name = pathlib.Path(path)
    ds = pydicom.dcmread(str(file_name))
    print(ds)
    if ds[0x0018, 0x0015].value=='KNEE':
        return True
    return False    

path='datasets/OAI12Month/downloads/results/1.C.2/9752412/20051003/01133104_1x1.jpg'
print(dicom_validity(path))