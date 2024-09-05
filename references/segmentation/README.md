# 수행한 작업
1. Labelme 라이브러리를 이용하여 라벨링을 직접 수행하여 새로운 데이터셋을 생성
2. Labelme 데이터셋을 COCO 데이터셋 형태로 변환하여 학습을 진행할 수 있도록 클래스 수 등의 코드를 수정
3. 데이터 증식 기법을 추가하여 V2가 아닌 V1에서도 여러 증식 기법을 사용할 수 있도록 함.
4. 훈련과 검증 모두 loss값과 acc값을 출력할수 있도록 직접 계산하거나 데이터를 추출
5. 생성된 checkpoint.pth를 TensorRT에서 사용할 수 있도록 onnx파일로 변환.

   
![image](https://github.com/user-attachments/assets/b5c54e0c-3dbd-46cb-9223-2e701464fead)
![image](https://github.com/user-attachments/assets/32cf7cae-8754-4026-a403-5b00051f9313)
![image](https://github.com/user-attachments/assets/eed2058a-3112-4ac1-a927-35115c006acf)
![image](https://github.com/user-attachments/assets/29f788ce-d692-4d81-8fa3-53f407d9765b)


V1으로 ElasticTransform을 구현한 이미지

![image](https://github.com/user-attachments/assets/5f887edf-820c-4f43-a462-8548b13fdaec)
