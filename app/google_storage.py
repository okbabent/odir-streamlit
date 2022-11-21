import streamlit as st
from google.oauth2 import service_account
from google.cloud import storage
import traceback
import logging

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
    bucket_name = "odir-datascientest"
    file_path = f'model/{h5_name}'
    print('loading from bucket ----')
    bucket = client.bucket(bucket_name)
    target = f'./data/storage_target/{h5_name}'
    try:
        bucket.blob(file_path).download_to_filename(target)
    except Exception as e:
        logging.error(traceback.format_exc())
        print('exception ', e)
        target = ''
    print('end loading from bucket ----')
    return target


# if __name__ == '__main__':
#     h5 = 'odir_model_weights_Xception_2022_10_21_multiclass_fine_tuning.h5'
#     load_model(h5)