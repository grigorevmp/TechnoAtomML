"""
Для второй части ничего из приложения не потребуется.

Постановка задачи:Необходимо реализовать функции accuracy_score, precision_score, recall_score, lift_score, f1_score
(названия функций должны совпадать).
Каждая функция должна иметь 3 обязательных параметра def precision_score(y_true, y_predict, percent=None).
Добавлять свои параметры нельзя.

Нельзя использовать готовые реализации этих метрик
Если percent=None то метрика должна быть рассчитана по порогу вероятности >=0.5
Если параметр percent принимает значения от 1 до 100 то метрика должна быть рассчитана на соответствующем ТОПе
1 - верхний 1% выборки
100 - вся выборка
y_predict - имеет размерность (N_rows; N_classes)
Ожидаемый результат: файл *.py c реализованными функциями
"""

import numpy as np


def createClassification(y_p, y_true, percent):
    a = y_p[:, -1]
    threshold = (100 - percent) / 100
    quantile = np.percentile(a=a, q=(100 - percent))

    elements_in_quantile = [_a for _a in enumerate(a) if _a[1] >= quantile]

    top_y = [y_true[index[0]] for index in elements_in_quantile]
    binary_y_p = (element[1] > threshold for element in elements_in_quantile)

    return getClassification(top_y, binary_y_p)


def getClassification(y_t, y_p):
    """
    :param y_t: true y
    :param y_p: predicted y
    :return:
    """
    tp, fp, tn, fn = 0, 0, 0, 0
    for _t, _p in zip(y_t, y_p):
        if _t == _p == 1:
            tp += 1
        elif _t == 1 and _p == 0:
            fn += 1
        elif _t == _p == 0:
            tn += 1
        elif _t == 0 and _p == 1:
            fp += 1
    return tp, fp, tn, fn


DEFAULT_PERCENT = 50


def accuracy_score(y_true, y_pred, percent=DEFAULT_PERCENT):
    tp, fp, tn, fn = createClassification(y_pred, y_true, percent)
    return (tp + tn) / (tp + tn + fp + fn)


def precision_score(y_true, y_pred, percent=DEFAULT_PERCENT):
    tp, fp, tn, fn = createClassification(y_pred, y_true, percent)
    return tp / (tp + fp)


def recall_score(y_true, y_pred, percent=DEFAULT_PERCENT):
    tp, fp, tn, fn = createClassification(y_pred, y_true, percent)
    return tp / (tp + fn)


def lift_score(y_true, y_pred, percent=DEFAULT_PERCENT):
    tp, fp, tn, fn = createClassification(y_pred, y_true, percent)
    return (tp / (tp + fp)) / ((tp + fn) / (tp + tn + fp + fn))


def f1_score(y_true, y_pred, percent=DEFAULT_PERCENT):
    recall = recall_score(y_true, y_pred, percent)
    precision = precision_score(y_true, y_pred, percent)
    return 2 * (precision * recall) / (precision + recall)


if __name__ == '__main__':
    file = np.loadtxt('HW2_labels.txt', delimiter=',')
    y_predict, y_true = file[:, :2], file[:, -1]
    print('Accuracy - ', accuracy_score(y_true, y_predict))
    print('Precision - ', precision_score(y_true, y_predict))
    print('Recall - ', recall_score(y_true, y_predict))
    print('Lift - ', lift_score(y_true, y_predict))
    print('F1 - ', f1_score(y_true, y_predict))

    print('Accuracy - ', accuracy_score(y_true, y_predict, 70))
    print('Precision - ', precision_score(y_true, y_predict, 70))
    print('Recall - ', recall_score(y_true, y_predict, 70))
    print('Lift - ', lift_score(y_true, y_predict, 70))
    print('F1 - ', f1_score(y_true, y_predict, 70))
