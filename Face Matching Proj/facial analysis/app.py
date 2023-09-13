from streamlit_webrtc import webrtc_streamer
import av
import streamlit as st
import cv2
import boto3


rekog_client = boto3.client('rekognition')


def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    try:
        bytes_data = cv2.imencode('.png', img)[1].tobytes()
        response = rekog_client.detect_faces(Image={'Bytes': bytes_data,}, Attributes=['ALL'])
        text = ""
        alert = "Alert"
        if len(response['FaceDetails']) > 0:
            text = 'Emotion : '+','.join([i['Type'] for i in response['FaceDetails'][0]['Emotions'] if i['Confidence'] >= 50])
            pose_yaw = response['FaceDetails'][0]['Pose']['Yaw']
            pose_pitch = response['FaceDetails'][0]['Pose']['Pitch']
            eye_yaw = response['FaceDetails'][0]['EyeDirection']['Yaw']
            eye_pitch = response['FaceDetails'][0]['EyeDirection']['Pitch']
            if abs(pose_pitch) < 15 and abs(pose_yaw) < 20 and abs(eye_yaw) < 20 and abs(eye_pitch) < 15:
                alert = ""
        else:
            text = "No Face Detected"
        font = cv2.FONT_HERSHEY_SIMPLEX
        org = (30, 20)
        fontScale = 0.8
        color = (255, 0, 0)
        thickness = 2
        image = cv2.putText(img, text, org, font, fontScale, color, thickness, cv2.LINE_AA)
        if alert != "":
            image = cv2.putText(image, alert, (30, 45), font, fontScale, (0, 0, 255), thickness, cv2.LINE_AA)
    except Exception as e:
        print(e)
        return av.VideoFrame.from_ndarray(img, format="bgr24")
    return av.VideoFrame.from_ndarray(image, format="bgr24")


webrtc_streamer(key="example", 
                video_frame_callback=video_frame_callback, 
                media_stream_constraints={'audio':False, 'video':True})