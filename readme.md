# DraftVision

## About
Библиотека разрабатывается для быстрого прототипирования (возможности быстро попробовать SOTA решения на доступных данных) и не имеет цели стать вторым HuggingFace. Предполагается поддерживать небольшой пул моделей, не раздувая библиотеку до уровня `MMdetection`
## Available tasks

### Detection

### Segmentation

### Pose Estimation

# TODO list
## ML
1. Накопление градиента
2. Добавить аугментации MixUp, AugMix, CopyPast и др. (Добавлены частично для детекции)
3. AMP
4. Multi GPU training
5. Продвинутое логгирование: ClearML & TensorBoard (Частично, только TB)
6. Dfine seg
7. Dfine for pose estimation 
8. Deploy для моделей (метод deploy, который вызывается перед конвертацией в ONNX)


## Core
1. Поменять dataclass на BaseModel (pydantic). Это позволит явно задавть формат ожидаемых входных параметров. Но я не уверен насколько это критично
2. Тесты на ядро