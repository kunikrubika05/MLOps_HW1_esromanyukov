# Скоринг фродовых транзакций

## Запуск

Нужны Docker с Compose v2 и работающим Docker Engine. Модель и пример данных включены в репозиторий.

Из корня клонированного репозитория:

```sh
docker compose up -d --build
```

Откройте http://localhost:8501. Загрузите `data/demo.csv` или `test.csv`, выберите количество транзакций и нажмите «Отправить». Результаты обновляются кнопкой «Посмотреть результаты» в соседней вкладке. Прогресс скоринга обновляется автоматически.

## Проверка

```sh
docker compose exec interface python -m tests.integration
docker compose exec interface python -m unittest discover -s tests
```

Первая команда проверяет полный поток и оставляет тестовые записи в базе. Вторая проверяет модель и интерфейс.

Логи: `docker compose logs scorer storage`. Остановка: `docker compose down`. Сброс данных: `docker compose down -v`.

## Ссылки

[Модель и препроцессинг](https://github.com/NikitaMalykhin/mts25_mlops_hw2_real_time_fraud_detection/tree/41540d91f0cbde0106671d3eb8270403508fade4) · [Данные](https://www.kaggle.com/competitions/teta-ml-1-2025)

Повторная подготовка артефактов с зависимостями из `requirements.txt`:

```sh
python -m scripts.prepare_artifacts --train /path/to/train.csv --model /path/to/my_catboost.cbm
```
