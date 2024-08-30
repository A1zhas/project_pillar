import os
import torch
import requests
import yaml
from PIL import Image
from ultralytics import YOLO

#os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
device='cuda' if torch.cuda.is_available() else 'gpu'

def main():
    yaml_file_path = "C:\\Users\\Aizhas\\Desktop\\ml_backend\\projects\\project opory\\pivot\\data.yaml"
    with open(yaml_file_path, 'r') as file:
        dataset_info = yaml.safe_load(file)

    print("Путь к обучающим данным:", dataset_info['train'])
    print("Путь к валидационным данным:", dataset_info['val'])
    print("Количество классов:", dataset_info['nc'])
    print("Имена классов:", dataset_info['names'])
    print(device)

    # Загрузка предварительно обученной модели (рекомендуется для обучения)
    model = YOLO('yolov8s.pt')

    # Обучение модели
    results = model.train(
        data=yaml_file_path,
        epochs=1,
        imgsz=640,
        device='cuda' if torch.cuda.is_available() else 'gpu',
    )

    # Сохранение обученной модели
    model.export(format='onnx')


if __name__ == '__main__':
    main()