import torch
import yaml
from ultralytics import YOLO
from ultralytics import settings

# # View all settings
print(settings)

# Проверка доступности CUDA
print(f"CUDA доступен: {torch.cuda.is_available()}")

# Вывод количества доступных GPU
print(f"Количество доступных GPU: {torch.cuda.device_count()}")

# Путь к файлу YAML с информацией о датасете
yaml_file_path = "pivot/data.yaml"

# Чтение информации о датасете из файла YAML
with open(yaml_file_path, 'r') as file:
    dataset_info = yaml.safe_load(file)

# Извлечение путей к обучающим, валидационным и тестовым данным
train_data_path = dataset_info['train']
val_data_path = dataset_info['val']
# test_data_path = dataset_info['test']
num_classes = dataset_info['nc']

# Список имен классов
class_names = dataset_info['names']

# Печать информации о датасете
print("Путь к обучающим данным:", train_data_path)
print("Путь к валидационным данным:", val_data_path)
# print("Путь к тестовым данным:", test_data_path)
print("Количество классов:", num_classes)
print("Имена классов:", class_names)

# Загрузка предварительно обученной модели (рекомендуется для обучения)
model = YOLO('yolov8s.pt')

# Обучение модели
results = model.train(
    data=yaml_file_path,
    device=0,
    epochs=200,
    imgsz=640,
)

# Сохранение обученной модели
model.export(format='onnx')
