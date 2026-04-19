# dl-2025
classification

* модель классифицирует wine dataset на 3 класса;
* test_acc  0.9629629850387573     
* test_loss 0.06445620954036713    
* лучший checkpoint выбран по val_loss
* https://colab.research.google.com/drive/1Y99KD3-zsJak0hWo0TtvABrM0CZ2_m0p?authuser=3#scrollTo=J_Jau1EqMxmG

# Multi-Branch MLP для Wine Quality Classification

## 📌 Описание
Реализация multi-branch нейронной сети для мультиклассовой классификации качества вина (Wine Quality Dataset, 6 классов: оценки 3-8).

## 🎯 Результаты

| Метрика | Значение |
|---------|----------|
| **F1 score (macro)** | **0.5354** ✅ |
| Accuracy | 0.5719 |
| Лучший F1 за обучение | 0.5354 |

**✅ Требование задания выполнено: F1 ≥ 40%**

## 🏗️ Архитектура модели

### Три параллельные ветки:
| Ветка | Преобразование | Описание |
|-------|----------------|----------|
| **Bottleneck Branch** | dim → dim//4 → dim | Сужение размерности |
| **Inverted Bottleneck Branch** | dim → dim*4 → dim | Расширение размерности |
| **Regular Branch** | dim → hidden_dim → dim | Обычный residual блок |

### Объединение: Concatenation

## 🔧 Гиперпараметры

| Параметр | Значение |
|----------|----------|
| hidden_dim | 256 |
| num_blocks | 6 |
| Learning rate | 1e-3 |
| Optimizer | AdamW |
| Batch size | 64 |
| Dropout | 0.1 |
| Combine mode | concat |
| Max epochs | 200 |
| Early stopping patience | 30 |

## 📊 Балансировка классов (SMOTE)

Исходный дисбаланс (классы 3 и 8 имели 8 и 15 примеров) был исправлен с помощью SMOTE:

| Класс (оценка) | Исходно | После SMOTE |
|----------------|---------|-------------|
| 3 | 8 | 300 |
| 4 | 42 | 300 |
| 5 | 545 | 545 |
| 6 | 510 | 510 |
| 7 | 159 | 159 |
| 8 | 15 | 300 |

## 📈 Метрики по классам

| Класс (оценка) | F1 score | Поддержка |
|----------------|----------|-----------|
| 3 | 0.0000* | 2 |
| 4 | 0.1633 | 11 |
| 5 | 0.6107 | 136 |
| 6 | 0.5517 | 128 |
| 7 | 0.5542 | 40 |
| 8 | 0.2857 | 3 |

> *Класс 3 не предсказался на валидации из-за крайне малого числа примеров (2)

## 🛠️ Технологии

- Python 3.8+
- PyTorch / PyTorch Lightning
- scikit-learn
- imbalanced-learn (SMOTE)
- matplotlib / seaborn

## 📁 Структура репозитория

