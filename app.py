import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as transforms
import numpy as np
import cv2
from modulize.model import *
from modulize.crawl import *  # Make sure this imports your model correctly

# Load your model and set it to evaluation mode
model = torch.load("../KimJeongHyeon/models/vitClassification10.pth", map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
model.eval()

st.title("Emotion and Face Detection Web App")
st.write("안녕하세요. 안부인사 어쩌구저쩌고. 오늘의 셀카를 업로드해주세요.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    width, height = image.size
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("감정을 분석중이에요...")

    # Image transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Apply transformations
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension

    # Move tensor to GPU if available
    if torch.cuda.is_available():
        image_tensor = image_tensor.to('cuda')

    # Make predictions
    predictions = model(image_tensor)
    #st.write(predictions,predictions[0],predictions[1])
    emotion = torch.argmax(predictions[0], dim=1).cpu().item()
    bbox = predictions[1][0].cpu().detach().numpy()
    #st.write(bbox)
    # Prepare the image for drawing
    image_draw = np.array(image)
    image_draw = cv2.cvtColor(image_draw, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR for OpenCV

    # Correcting the start and end points
    start_point = (int(bbox[2]*width/224), int(bbox[3]*height/224))
    end_point = (int(bbox[0]*width/224),int(bbox[1]*height/224))
    color = (255, 0, 0)  # BGR format for a blue box
    thickness = 2
    image_draw = cv2.rectangle(image_draw, start_point, end_point, color, thickness)

    # Convert back to RGB for displaying in Streamlit
    image_draw = cv2.cvtColor(image_draw, cv2.COLOR_BGR2RGB)

    dict2 = {3:'기쁨이',
         4:'상처',
         0:'분노',
         2:'당황이',
         1:'불안이',
         6:'중립이',
         5:'슬픔이'}
    
    # Convert back to PIL Image and show it
    final_image = Image.fromarray(image_draw)
    st.image(final_image, caption=f"Detected Emotion: {dict2[emotion]}", use_column_width=True)

    diary_entry = st.text_input("오늘의 간단한 일기를 적어주세요!")
    if diary_entry:
        # Analyze the diary entry if needed and prepare for the next step
        
        # Show the detected emotion and recommend a song
        st.write(f"당신의 오늘의 감정은 {dict2[emotion]}네요. 알맞는 노래를 추천해드릴게요!")
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
            st.write(f"추천 노래는 \n {selected_song.iloc[0]['가수']}의 {selected_song.iloc[0]['제목']} 입니다.")
            song_url = f"https://www.melon.com/song/detail.htm?songId={selected_song.iloc[0]['ID']}"
            link_html = f"<a href='{song_url}' target='_blank'><button style='margin: 10px; padding: 5px; border: none; color: white; background-color: #009688;'>노래 듣기</button></a>"
            st.markdown(link_html, unsafe_allow_html=True)
        
            
        # If "아니오" is clicked
        elif no_button:
            # Placeholder for the code to execute when "아니오" is clicked
            # Example: Load a DataFrame and display or perform some operation
            song_dataframe = pd.read_csv("../data/updated_dataset.csv")
            selected_song = random_song(emotion,song_dataframe)
            st.write(f"추천 노래는 {selected_song.iloc[0]['가수']}의 {selected_song.iloc[0]['제목']} 입니다.")
            song_url = f"https://www.melon.com/song/detail.htm?songId={selected_song.iloc[0]['ID']}"
            link_html = f"<a href='{song_url}' target='_blank'><button style='margin: 10px; padding: 5px; border: none; color: white; background-color: #009688;'>노래 듣기</button></a>"
            st.markdown(link_html, unsafe_allow_html=True)
        