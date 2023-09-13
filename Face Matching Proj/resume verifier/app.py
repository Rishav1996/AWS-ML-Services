import streamlit as st
import fitz
from PIL import Image
import io
import boto3


file_upload = st.file_uploader(label='Upload CV', accept_multiple_files=False, type='pdf')

rekog_client = boto3.client('rekognition')

if file_upload:
   bytes_data = file_upload.read()
   with open('./temp/temp.pdf', 'wb') as f:
      f.write(bytes_data)
   col1, col2 = st.columns(2)

   with col1:
      pdf_file = fitz.open('./temp/temp.pdf')
      page_nums = len(pdf_file)
      images_list = []
      for page_num in range(page_nums):
         page_content = pdf_file[page_num]
         images_list.extend(page_content.get_images())
      img = images_list[0]
      xref = img[0]
      base_image = pdf_file.extract_image(xref)
      image_bytes = base_image["image"]
      image = Image.open(io.BytesIO(image_bytes))
      st.image(image)

   with col2:
      img_file_buffer = st.camera_input("Take a picture")

      if img_file_buffer is not None:
         bytes_data = img_file_buffer.getvalue()
         cv2_img = Image.open(io.BytesIO(bytes_data))
         response = rekog_client.compare_faces(
            SourceImage={'Bytes': image_bytes,},
            TargetImage={'Bytes': bytes_data,},)
         if len(response['FaceMatches']) > 0:
            face_match = response['FaceMatches'][0]['Similarity']
            face_match = round(face_match, 2)
            st.write(f'Similarity Confidence : {face_match}%')
         else:
            st.write(f'Unable to match face')