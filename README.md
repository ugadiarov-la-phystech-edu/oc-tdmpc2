# Объектно-центрированное обучение с подкреплением с прогнозирующими моделями
[Группа в Telegram](https://t.me/+_sI-L8H5gWwxMmZi)
Основано на [оригинальном репозитории TD-MPC2](https://github.com/nicklashansen/tdmpc2)
## 1 Введение
### 1.1 Объектно центрированное обучение
Человеческое восприятие объектно-ориентировано. При восприятии сцены люди выделяют ключевые компоненты, такие как объекты, их границы и пространство вокруг них. Ожидается, что выделение из неструктурированных данных (изображений) представлений, соответствующих отдельным сущностям, улучшит обобщающую способность нейронных сетей. Например, полученные представления объектов можно будет повторно использовать для предсказания свойств новых комбинаций объектов, отсутствующих в обучающей выборке. Самые эффективные методы объектно-центрированного обучения без учителя используют модуль Slot Attention, который группирует признаки, соответствующие одному объекту на изображении, в отдельные векторы фиксированной длины — слоты.
### 1.2 Обучение с подкреплением с прогнозирующими моделями
Методы семейства TD-MPC направлены на поиск оптимальной стратегии, объединяя планирования и обучения с подкреплением. В процессе обучения создаётся модель, прогнозирующая динамику среды и вознаграждение, что позволяет выбрать такую последовательность действий, которая приведёт к максимальной отдаче с точки зрения текущей модели среды. Из-за вычислительной сложности планирования оптимизируется достаточно короткая последовательность действий, при этом модель полезности действий, обученная с использованием метода временных различий (TD), используется для прогнозирования отдачи, которую получит агент при выполнение действий за горизонтом планирования.
## 2 Описание задачи
### 2.1 Object-Centric TD-MPC2
Предлагается реализовать объектно-центрированный вариант TD-MPC2, где модель среды предсказывает динамику представлений отдельных объектов. В качестве кодировщика предлагается использовать предобученную модель [Dinosaur](https://arxiv.org/abs/2209.14860), которая извлекает из изображения множество объектных представлений — слотов. В оригинальном TD-MPC2 модели динамики, вознаграждения, полезности и стратегии представляют собой многослойные персептроны. Для работы со слотовым представлением эти компоненты предлагается реализовать с помощью графовых нейронных сетей (*можете предложить свою архитектуру*).
### 2.2 Описание сред
Реализованный алгоритм будет тестироваться в симуляционных средах для задач робототехники с непрерывным пространством действий и плотным вознаграждением.
#### 2.2.1 Задача Block Lifting в среде Robosuite
Красный куб размещён на столе. Управляя манипулятором, агент должен поднять куб над столом. Более подробное описание среды и визуализация задачи в [документации](https://robosuite.ai/docs/modules/environments.html#block-lifting).
#### *2.2.2\* Задача PushCube-v1 в среде ManiSkill (опционально)*
Синий куб размещён на столе. Управляя манипулятором, агент должен сдвинуть синий куб в целевую область на столе. Более подробное описание среды в [документации](https://maniskill.readthedocs.io/en/latest/tasks/table_top_gripper/index.html#pushcube-v1).
### 2.3 Оценка решения
Оценка решения задачи будет производиться путём сравнения эффективности обучения разработанного алгоритма и оригинального TD-MPC2.
## 3 Реализация
Для визуализации логов рекомендуется использовать сервис [wandb](https://wandb.ai/site).
### 3.1 Baseline (TD-MPC2)
Базовый алгоритм для сравнения - оригинальный TD-MPC2.

Запуск TD-MPC2 для задачи Block Lifting в среде Robosuite:
```
export WANDB_API_KEY=${your_wandb_api_key}
python tdmpc2/train.py task=lift obs=rgb obs_size=64 time_limit=125 model_size=48 batch_size=512 seed=0 eval_episodes=30 eval_freq=50000 wandb_project=tdmpc2_robosuite_lift wandb_entity=${your_entity_name} wandb_group_name=monolithic wandb_run_name=monolithic disable_wandb=false
```
*Запуск TD-MPC2 для задачи PushCube-v1 в среде ManiSkill (опционально):*
```
export WANDB_API_KEY=${your_wandb_api_key}
python tdmpc2/train.py task=push-cube obs=rgb obs_size=64 time_limit=100 model_size=48 batch_size=512 seed=0 eval_episodes=30 eval_freq=50000 wandb_project=tdmpc2_maniskill_push-cube wandb_entity=${your_entity_name} wandb_group_name=monolithic wandb_run_name=monolithic disable_wandb=false
```
**Визуализация метрик обучения TD-MPC2 в [Block Lifting](https://wandb.ai/ula_elfray/tdmpc2_robosuite-lift) и [PushCube-v1](https://wandb.ai/ula_elfray/tdmpc2_maniskill-push-cube).** 
### 3.2 Object-centric TD-MPC2
#### 3.2.1 Слотовый кодировщик: Dinosaur
Для извлечения неупорядоченного множества слотов из изображения применяется модель [Dinosaur](https://arxiv.org/abs/2209.14860): $\text{Dinosaur}(\text{image}) \rightarrow \boldsymbol{z} = \{z ^ 1, \dots, z ^ K \}$, где $z ^ i$ - слот $i$, $K$ - количество слотов.
Код Dinosaur расположен в `tdmpc2/ocr/tools.py`.
Веса предобученных моделей: [Block Lifting (Robosuite)](https://drive.google.com/file/d/1dj5Oq-iP-wDTC7StOOsNHULxONDQZ8LQ/view?usp=drive_link) и [PushCube-v1 (ManiSkill)](https://drive.google.com/file/d/1v0gjazJPzBLyWDgBmwtDBnBVNXbVokf7/view?usp=sharing).
Модель используется с замороженными весами в обёртке `SlotExtractorWrapper` в `tdmpc2/envs/wrappers/slots.py` для преобразования изображения в множество слотов.
Пример выделения объектов с предобученной моделью Dinosaur в среде Robosuite:
</br><img src="assets/robosuite_dinosaur.gif" width="100%"></br>
#### 3.2.2 Объектно-центрированная модель среды на основе графовых нейронных сетей
Код объектно-центрированной модели `OCWorldModel` с небольшими изменениями повторяет код оригинальной модели TD-MPC2 `WorldModel` (`tdmpc2/common/world_model.py`), но зависит от классов `OCDynamicsModel`, `OCRewardModel`, `OCPolicy`, **которые необходимо реализовать**.
В отличие от `WorldModel`, на вход `OCWorldModel` подаётся факторизованное состояние среды в виде множества слотов $\boldsymbol{z}_t = (z_t ^ 1, \dots, z_t ^ K)$.
Эти множества мы можем рассматривать как полные графы и обрабатывать их с помощью графовых нейронных сетей (Graph Neural Network - GNN).
GNN состоит из двух MLP: модели вершины $\texttt{node}$ и модели ребра $\texttt{edge}$.
Модель ребра для пары вершин и действия рассчитывает эмбеддинг ребра $\texttt{edge} (z_t ^ i, z_t ^ j, a_t))$.
Модель вершины принимает на вход слот, действие и сумму эмбеддингов входящих в вершину рёбер, и выдаёт эмбеддинг вершины.
Класс GNN реализован в `tdmpc2/common/layers.py`.
##### 3.2.2.1 OCDynamicsModel:
Предсказывает значение слотов при совершении действия $a$. 

Вход модели: множество слотов $\boldsymbol{z}_t$ для текущего шага $t$ и действие $a_t$.

Выход модели: предсказанный множество слотов $\hat{\boldsymbol{z}}_{t+1}$ для следующего шага.

Выход модели вершины $\texttt{node}$ интерпретируется как предсказание значение эмбеддинга для следующего шага.

$\hat{z} ^ i = \texttt{node} (z_t ^ i, a_t, \sum_{j \neq i} \texttt{edge} (z_t ^ i, z_t ^ j, a_t))$
##### 3.2.2.2 OCRewardModel:
Предсказывает вознаграждение при совершении действия $a$.

Вход модели: множество слотов $\boldsymbol{z}_t$ для текущего шага $t$ и действие $a_t$.

Выход модели: один вектор $\text{embed}$ (эмбеддинг графа), на основании которого рассчитывается вознаграждение.

Для получения единого вектора применяется операция READOUT, инвариантная относительно перестановки вершин в графе, например, усреднение эмбеддингов вершин, их сумма, min, max и т.д.
mean-READOUT: $\text{embed} = MLP \[ \sum_{i = 1} \texttt{node} (z_t ^ i, a_t, \sum_{j \neq i} \texttt{edge} (z_t ^ i, z_t ^ j, a_t)) / K\]$
##### 3.2.2.3 OCPolicy:
Предсказывает $\hat{\mu}$ и $\hat{\sigma}$ для распределения действий, моделируемого нормальным распределением.

Вход модели: множество слотов $\boldsymbol{z}_t$ для текущего шага $t$ .

Выход модели: один вектор $\text{embed}$, на основании которого рассчитывается мат. ожидание и дисперсия распределения действий.

Архитектура повторяет `OCRewardModel`, но не зависит от действия.

$\text{embed} = MLP \[ \sum_{i = 1} \texttt{node} (z_t ^ i, \sum_{j \neq i} \texttt{edge} (z_t ^ i, z_t ^ j)) / K\]$
##### 3.2.2.4 Функция полезности действий Q:
Предсказывает полезность действия $a$.

Вход модели: множество слотов $\boldsymbol{z}_t$ для текущего шага $t$ и действие $a_t$.

Выход модели: один вектор $\text{embed}$, на основании которого рассчитывается полезность.

Архитектура полностью повторяет `OCRewardModel`, но в TD-MPC2 используется ансамбль из нескольких моделей для предсказания полезности.
### 3.2.3 Запуск
Дополнительные параметры, связанные с использованием Dinosaur: `n_slots` - количество извлекаемых слотов, `slot_dim` - размерность слота, `slot_extractor_checkpoint_path` - путь к чекпоинту модели Dinosaur,
`dino_model_name` - тип модели DINO, `num_patches` - количество используемых патчей в моделе DINO, `input_feature_dim` - размерность входного вектора признаков,
`features` - размерности SlotAttention.

Параметры, с которыми следует запускать объектно-центрированные варианты TD-MPC2: 
Запуск Object-Centric TD-MPC2 для задачи Block Lifting в среде Robosuite:
```
export WANDB_API_KEY=${your_wandb_api_key}
python tdmpc2/train.py task=lift obs=slots obs_size=224 time_limit=125 model_size=48 batch_size=512 seed=0 eval_episodes=30 eval_freq=50000 n_slots=5 slot_dim=64 slot_extractor_checkpoint_path=${path/to/dinosaur/checkpoint} wandb_project=tdmpc2_robosuite_lift wandb_entity=${your_entity_name} wandb_group_name=slots wandb_run_name=slots disable_wandb=false
```
*Запуск Object-Cenric TD-MPC2 для задачи PushCube-v1 в среде ManiSkill (опционально):*
```
export WANDB_API_KEY=${your_wandb_api_key}
python tdmpc2/train.py task=push-cube obs=slots obs_size=224 time_limit=100 model_size=48 batch_size=512 seed=0 eval_episodes=30 eval_freq=50000 dino_model_name=vit_small_patch8_224_dino n_slots=4 slot_dim=128 input_feature_dim=384 num_patches=784 features=\[1024,1024,1024\] slot_extractor_checkpoint_path=${path/to/dinosaur/checkpoint} wandb_project=tdmpc2_maniskill_push-cube wandb_entity=${your_entity_name} wandb_group_name=slots wandb_run_name=slots disable_wandb=false
```
### 3.2.3 Отладка
В TD-MPC2 перед началом обучения происходит первичное заполнение буфера прецендентов.
Для ускорения отладки кода, чтобы не ждать долго заполнения буфера, запуск можно осуществлять со следующими параметрами: `buffer_size=1000 eval_episodes=1 time_limit=20`.
## 4 Окружение
### 4.1 Docker
Сбор docker-образа:
```
cd docker && docker build . -t <user>/tdmpc2:1.0.0
```
### 4.2 Conda
Локальная установка зависимостей через `conda`:
```
conda env create -f docker/environment.yaml
```
