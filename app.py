import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as transforms
import numpy as np
import cv2
import datetime as dt
import random as rd
from modulize.model import *
from modulize.crawl import *
from segmentation import *
  # Make sure this imports your model correctly

# Load your model and set it to evaluation mode
model = torch.load("vitdetection2.pth", map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
model.eval()

date = dt.datetime.today().strftime('%Y-%m-%d')
st.title("Emotion and Face Detection Web App🤩")
st.write(f"안녕하세요👋🏻 {date} 오늘의 셀카를 업로드해주세요🥰")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    width, height = image.size
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("잠시만 기다려주세요 감정을 분석중이에요...🧐")

    # Image transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image_np = np.array(image)
    image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    image_with_boxes = np.copy(image_np)

    # Convert BGR image to grayscale for face detection
    gray_image = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
    face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    faces = face_classifier.detectMultiScale(
        gray_image, scaleFactor=1.1, minNeighbors=1, minSize=(40, 40)
    )

    if len(faces) > 0:
        x, y, w, h = faces[0]
    
    # Apply transformations
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension

    # Move tensor to GPU if available
    if torch.cuda.is_available():
        image_tensor = image_tensor.to('cuda')

    # Make predictions
    predictions = model(image_tensor)
    emotion = torch.argmax(predictions[0], dim=1).cpu().item()
    # Prepare the image for drawing
    image_draw = np.array(image)
    image_draw = cv2.cvtColor(image_draw, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR for OpenCV

    # Correcting the start and end points
    #start_point = (int(bbox[2]*width/224), int(bbox[3]*height/224))
    #end_point = (int(bbox[0]*width/224),int(bbox[1]*height/224))


    color = (255, 0, 0)  # BGR format for a blue box
    thickness = 4
    if len(faces) > 0:
    # Find the largest face by area (w*h)
        largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
        x, y, w, h = largest_face
        
        color = (255, 0, 0)  # BGR format for a blue box
        thickness = 2
        # Draw the rectangle around the largest detected face
        image_with_boxes = cv2.rectangle(image_with_boxes, (int(x), int(y)), (int(x+w), int(y+h)), color, thickness)

        # Convert back to RGB for displaying in Streamlit if necessary
        image_with_boxes = cv2.cvtColor(image_with_boxes, cv2.COLOR_BGR2RGB)

        # Convert the NumPy array back to PIL Image for Streamlit
        final_image = Image.fromarray(image_with_boxes)
        st.image(final_image, caption='Uploaded Image with Detected Face', use_column_width=True)
      
    dict2 = {
        3:'기쁨:heart_eyes:이',
        4:'상처:pensive:',
        0:'분노:rage:',
        2:'당황:frowning:이',
        1:'불안:worried:이',
        6:'중립:neutral_face:이',
        5:'슬픔:cry:이'
    }
    
    input_dict = {
        0: ['긍정적인 생각으로 분노를 가라앉혀보세요🫠', '분노 Stop!', '내일은 오늘보다 훨씬 나아질거예요~'],
        1: ['불안한 감정은 그만🖐🏻 다 잘 될 거예요!', '불안함은 결국 나를 성장시킬거예요!', '그만 불안하기~'],
        2: ['당황스럽겠지만 릴렉스~', '여유로운 티타임 어떠세요?', '모든일은 마음먹기에 달려있어요!'],
        3: ['어제보다 기쁨이 ✌🏻배!!', '오늘보다 내일 더 기쁠거예요🌞', '기쁘니까 기쁘다.'],
        4: ['노래로 상처를 치유해보세요🎆', '상처는 금방 괜찮아진다~', '별일 아니에요. 걱정 마세요!'],
        5: ['슬픔은 나누어요!', '외로워도 슬퍼도 나는 안울어🥲', '금방 사라질 슬픔...'],
        6: ['언제나 중간~', '좋지도 나쁘지도 않지만 해피해피해피~', '오늘 기분도 🍗도 반반!']
    }
    
    combined_entries = {**dict2, **input_dict}
    random_input = rd.choice(combined_entries[emotion])
    st.write(f"오늘의 감정은 {dict2[emotion]}네요! {random_input}")
    st.write(f"당신의 오늘의 감정에 알맞는 노래를 추천해드릴게요!")
    # add 2 buttons ad if the button is clicked "예", run if or "아니오" else
    st.write("노래 목록을 업데이트 할까요?")
    col1, col2 = st.columns(2)
    with col1:
        yes_button = st.button("예")
    with col2:
        no_button = st.button("아니오")

    # If "예" is clicked
    if yes_button:
        # Placeholder for the code to execute when "예" is clicked
        song_dataframe = crawl_analyze()
        # Example action: Show a message or perform some operation
        # You can replace this with your own logic or function call
        selected_song = random_song(emotion,song_dataframe)
        st.write("="*30)
        st.write(f"추천 노래는 \n {selected_song.iloc[0]['가수']}의 {selected_song.iloc[0]['제목']} 입니다.")
        song_url = f"https://www.melon.com/song/detail.htm?songId={selected_song.iloc[0]['ID']}"
        link_html = f"<a href='{song_url}' target='_blank'><button style='margin: 10px; padding: 5px; border: none; color: white; background-color: #009688;'>노래 듣기</button></a>"
        st.markdown(link_html, unsafe_allow_html=True)
        diary_entry = st.text_input("오늘의 간단한 일기를 적어주세요!")
        
    # If "아니오" is clicked
    elif no_button:
        # Placeholder for the code to execute when "아니오" is clicked
        # Example: Load a DataFrame and display or perform some operation
        song_dataframe = pd.read_csv("updated_dataset.csv")
        selected_song = random_song(emotion,song_dataframe)
        st.write(f"추천 노래는 {selected_song.iloc[0]['가수']}의 {selected_song.iloc[0]['제목']} 입니다.")
        song_url = f"https://www.melon.com/song/detail.htm?songId={selected_song.iloc[0]['ID']}"
        link_html = f"<a href='{song_url}' target='_blank'><button style='margin: 10px; padding: 5px; border: none; color: white; background-color: #009688;'>노래 듣기</button></a>"
        st.markdown(link_html, unsafe_allow_html=True)
        diary_entry = st.text_input("오늘의 간단한 일기를 적어주세요!")
        


    

    
        
