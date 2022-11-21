import streamlit as st
from google.oauth2 import service_account
from google.cloud import storage
import traceback
import logging
import os

credentials = service_account.Credentials.from_service_account_info(st.secrets["gcp_service_account"])
client = storage.Client(credentials=credentials)





# Retrieve file contents.
# Uses st.experimental_memo to only rerun when the query changes or after 10 min.
st.experimental_memo(ttl=600)
def read_file(bucket_name, file_path):
    bucket = client.bucket(bucket_name)
    content = bucket.blob(file_path).download_to_filename('./tmp')
    return content

def connect_google_strorage():
    # Create API client.
    bucket_name = "odir-datascientest"
    #file_path = "odir_model_weights_Xception_2022_10_21_multiclass_fine_tuning.h5"
    file_path = 'requirements.txt'

    content = read_file(client, bucket_name, file_path)

    # print(content)
st.experimental_memo(ttl=600)
def load_model(h5_name):
    target = f'./{h5_name}'
    if os.path.isfile(target):
        print(f'model file already exist')
        return target

    bucket_name = "odir-datascientest"
    file_path = f'model/{h5_name}'
    print('loading from bucket ----')
    logging.info(f'loading {file_path} from bucket {bucket_name} ---------')
    bucket = client.bucket(bucket_name)
    #target = f'./data/storage_target/{h5_name}'
    target = f'./{h5_name}'
    logging.info(f'download_to_filename to  {target} ')
    try:
        bucket.blob(file_path).download_to_filename(target)
        logging.info(f'end bucket.blob')
    except Exception as e:
        logging.error(traceback.format_exc())
        print('exception ', e)
        target = ''

    logging.info(f'end loading from bucket ----')
    print('end loading from bucket ----')
    return target


# if __name__ == '__main__':
#     h5 = 'odir_model_weights_Xception_2022_10_21_multiclass_fine_tuning.h5'
#     load_model(h5)