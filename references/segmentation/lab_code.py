import torch
from PIL import Image, ImageOps
import torchvision.transforms as T
import numpy as np
import cv2
import time

# Define the helper function
def decode_segmap(image, nc=3):
    label_colors = np.array([(0, 0, 0),  # 0=background
                             #1=floor, 2=blockage
                             (255, 255, 255), (100, 100, 100),
                             ])

    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)

    for l in range(0, nc):
        idx = image == l
        r[idx] = label_colors[l, 0]
        g[idx] = label_colors[l, 1]
        b[idx] = label_colors[l, 2]

    rgb = np.stack([r, g, b], axis=2)
    return rgb

# 모델 로드
model = torch.load('.\model.pth', weights_only=False)
model.eval()

# 비디오 경로 및 캡처 객체 정의
video_path = 'INPUT YOUR VIDEO PATH'
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Cannot open video file")
    exit()

# 비디오 속성 가져오기
fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 출력 비디오 작성을 위한 코덱 및 VideoWriter 객체 정의
output_path = 'INPUT YOUR RESULT VIDEO PATH'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out_video= cv2.VideoWriter(output_path, fourcc, fps, (1280, 480))
output_mask_path = 'INPUT YOUR RESULT VIDEO PATH'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
mask_video= cv2.VideoWriter(output_mask_path, fourcc, fps, (640, 480))


# 각 프레임 처리
while cap.isOpened():
    start_time = time.time()
    ret, frame = cap.read()
    
    if not ret:
        break
    
    # OpenCV 이미지(BGR)를 PIL 이미지(RGB)로 변환
    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    # 이미지 전처리
    pil_img = ImageOps.exif_transpose(pil_img)
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
    ])
    img = transform(pil_img).unsqueeze(0).to('cuda')
    
    # 모델 출력 가져오기
    with torch.no_grad():
        out = model(img)['out']

    om = torch.argmax(out.squeeze(), dim=0).detach().cpu().numpy()
    
    # 세그멘테이션 맵 디코딩
    rgb = decode_segmap(om)
    
    # RGB를 BGR로 변환하여 OpenCV에 저장
    output_frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    
    output_frame = cv2.resize(output_frame, (640, 480))
    frame = cv2.resize(frame, (640, 480))   
    #output_frame = cv2.resize(output_frame, (640, 320))
    #frame = cv2.resize(frame, (640,320))   
    end_time = time.time()
    
    # 원본 및 세그멘테이션 프레임 나란히 연결
    combined_frame = np.hstack((frame, output_frame))
    
    # 프레임 표시
    cv2.imshow('Original and Segmentation', combined_frame)
    print("image_size : (", frame_width, frame_height, ")", ', one_frame_time : ', end_time - start_time)
    print("combined width and height : ", combined_frame.shape)
    # 키 입력 대기, 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    # 출력 비디오 파일에 작성
    #out_video.write(combined_frame)
    #mask_video.write(output_frame)

# 비디오 캡처 및 기록기 해제
cap.release()
out_video.release()
mask_video.release()
cv2.destroyAllWindows()
