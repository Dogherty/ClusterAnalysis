# Кластеризація даних за допомогою MeanShift і KMeans
Цей проєкт є прикладом кластеризації даних з використанням алгоритмів `MeanShift` і `KMeans` з бібліотеки `scikit-learn`. Тут ми створюємо і візуалізуємо штучні дані, а потім застосовуємо обидва алгоритми до цих даних для кластеризації.
## Опис

Цей код реалізує алгоритм кластеризації методом зсуву середнього (`MeanShift`) і метод к-середніх (`KMeans`) для набору даних, створеного з використанням функції `make_blobs` з бібліотеки `scikit-learn`. Він досліджує структуру даних і знаходить оптимальну кількість кластерів, використовуючи метрику силуету (`silhouette_score`), а потім візуалізує результати кластеризації з використанням графіків.

Це може бути корисно для аналізу даних, виявлення груп або патернів у даних, таких як сегментація користувачів, аналіз ринку, визначення груп товарів та інших структур у даних.

- <strong>Генерація даних:</strong> Ми починаємо з генерації штучних даних за допомогою функції make_blobs з бібліотеки `scikit-learn`. Ця функція створює кластери точок даних із заданими центрами і стандартним відхиленням.

- <strong>Кластеризація за допомогою `MeanShift`::</strong> Ми застосовуємо алгоритм `MeanShift` до наших даних для кластеризації. Алгоритм автоматично визначає кількість кластерів.

- <strong>Візуалізація вихідних даних і центрів кластерів `MeanShift`:</strong> Ми візуалізуємо вихідні дані та центри кластерів, визначені алгоритмом `MeanShift`.

- <strong>Визначення оптимальної кількості кластерів за допомогою `KMeans`:</strong> Ми використовуємо алгоритм `KMeans` для визначення оптимальної кількості кластерів на основі метрики `silhouette_score`.

- <strong>Кластеризація за допомогою `KMeans`:</strong> Ми застосовуємо алгоритм `KMeans` з оптимальною кількістю кластерів до наших даних.

- <strong>Візуалізація кластеризованих даних з областями кластеризації:</strong> Ми візуалізуємо кластеризовані дані та центри кластерів за допомогою `KMeans`.

## Скриншоти
  
<table><tr>
  <td colspan="2">
  <img src="https://lh3.googleusercontent.com/pw/ABLVV86psi2qDCF6Xcz5NrS3PYADfha_czBXSm_7xiyBGNGeNg4IZfdizxsgnMZlqn6SU7iniSdBQE5Xux-mpnND77eNhRyH_caT7TqUTyzqZwWHfoY5m6WQZdsidEZnoZVnrLfL4LikRaQTW4imhfzI_huO_DHn1uTpmQhD3sCtXCGH_nh5m1AspZIEisgmQrb3H0BJLnp_XNQ1CcvdlEnyyfbtaXel86zXIDmPfAbJHi7YtwZ6b0fDN_u_hJFcOWSaAsrw5nXaidPG9HDLENmDc36jN0UPjRmxETRyjeVpLAc_UkyhzIXSOr492LKeJnA9HqAYmq1IzaEQ-JbieSrKhv-AxOe2CSslab2yMitAj08VssEjCM9uIwYwOvWe9-fpu7nNo9t8-JnL58omzmTGoTDKfmW9ZCpTcwBXIioV5YqUQIWIGTZLEVVHjjijaUkeMQATUzY83up0uzMcLgisLU8u7OS5exI10WodbKKYJnYfQb424UJERcY4QFIeOTGxcD1qgSupykI58EgAu23s4xXWF3VescVxbrAQcreRlJ5UB2TLd-KgfSvd7yDZ58O3dqu65oL4Duqx2ITPsNRDYztiF_THSv83dkns4-l_gpOH0MR34c4442cmuoCjJJjMKgi0b2Vo7XPeppz3MuGGwke2abbYQoz1uqfFsJeFIaNLQQWJFPhHjJnFM1IuodTcrVqlQL_xzLo9JwXa5ND4KgNKD-ocZOHPzhPSzORWZiS61Pa3qbkr22VBOGEMY25y4MFCN0SU49gDi4w-EB179pPxUrpsv00xffbsabR6Td5xPLsyWckinuVtN5X6xhodhbMsSrzwhA-vaBElw1_zN9p6eCoLydOw4qWeWxKR7IYGfCtycRGvEozkqIDgp-zJ2HoCuGS7wCUuiKd_2tSJtORbW5s=w950-h531-s-no?authuser=0" alt="Image" width="900" height="500">

</td>
</tr>
<tr>
<td>
  <img src="https://lh3.googleusercontent.com/pw/ABLVV86VLxGrPwhkGYlwZ_YKlTXdPKG8kcS70fWk8XvZf0DyXk47Pemd199OEj-TYHdXD958JRt3_7h2OcU1pgUFPCzq4qp4bsOMxA5RbC7qVvFjqXBwo0V9N7Z6jSU_q1xTJoypuTkJxnBqouf3ouaXemCDlrd6JB2qrHY-xVJQiAKN4WGS9xAqsNT-oQ5yvAQpnV2ndjitercW04AZuL4LsQTK8N5wB11dK0LNtE_kj-CtOjlqKcrdfmcqOAXDrhplBy0wi8HbwvlmMVr6DQgGZZvj1ZTXLvPjIt-183p5KnyoHtFRy2M0qfPjY3cUPlDswPdGs-l2rbVEFKcz7kdq5i1HpS5jIMgMzMJhyEPu1j3GgFCi5DJUUiB6CjltpJZ85VnvINrnNanXgwDu_ZxIV-bQOIjlMnbCYq05R1XwAN_2jRyZs02E-FWRps0FubCA8yaLwOWrhMWmrNauEhpN4rcucM3d6WBcgRVHkh-LY9APNpedt3nskKu2E6RlGRjNYq77uCKY7N9HlDPf0qNIUxQiODvu1pvXuCsohZhzjtYtJINA_nrZD-BdfYXCiWc9vTiOqORtZv7ULSECsybxJtVHBIwGa63vilszcaSW-hhyWxZiroqj00nyOdAjeMAAyzixEudjM-KjuE8doiFcTaQnlO7vdhRkyS9gGDSTogr3Sic1rQbMLLXzmeN93e6Dks0WsT4dBqo8MN6zHmATXmvEmgrw-paJH6I_bKornIWDpWxR1hg1k7fvZ-9Dqeh-mIOqPoUtoAPF4-oCZZy0O9NwIi1pHczRo5jP4V_96lY-dBcG1fWaWwPtZPMzvb6f6YH0tSK5830S38pObLn4xwyktWkGkWShvW7vZExDTKoajqNB1KxZskZWvpUoAtAOuE1hkl1mgo2E4qRec3HY3y1B5QI=w598-h672-s-no?authuser=0" alt="Image" width="450" height="495">
</td>
  <td><img src="https://lh3.googleusercontent.com/pw/ABLVV84NAcu1pEkSsYF3uH08iLpJ1ENUMWuCm48Q2UhTEVeGYlbE14lJsGl7DcW_jEItUGwVbzPPcwSBksQfK_iVUXopwuCc7AYK7dic6ueTtPXp-RBXFp2wDQ0mRiQd71w7Na5FMG-Ky1RJliBNeUlqlf1sT1C4r8qRpckSOreXw4t_zq-IQ8DZgrT0zghzDtA35_YfUSU2s1twSXXj-AkUK0_hW7cz-s08oaO41CMs48z9cz9-m88jj15kxrhGtCyxFcBkRCiKJYYtwtmYHsBYOs3psJOep4rj9rxqgnq1g2Q_JG5Gx05Friu1xZqnXKe4eCC_VxbHvgu2i87_yKGKEt4g3UFELhvHCFH_Jl5OuY5e3wSnkYNz5f4P6-O4Q2zeSg-tdFmOkZSUCZivX60VF2tnoT-LOIBM_8Wl6Gi6VanjPINJl6IsHBWn3i2exGso8OnlCAroS7QUB3Au_gTBC2VtFNv9CS1lTbsNPNlQ3nRETdesy4TFNekSLNGXwt8Mslegt8Um1Nqk3kEnpIQ2X90LDOtU2dxQ2icBA0-xgSdVF7VIxFSBi9m2Q66xTGMGA6CuhwysRqUcfXYCu9sIZP1FHO67oxGfw8dbRd8fNHyhxBpe4mFAyZt1X9m4m-avLiHw8LMLEDjkbrqtnfnkmNu96l0UZ1GQgcofErjCYKf6RhFrWbC9_qmhPnQVHFCW1n0IMII1w0hdgSaZq6rGearJFVh2qLF-W06_nF8J_CWvNmCOoQ19beRnt-QFITtTcaJBWspKyhNv4NNzhY3AaE2WhFlR6zGDkgmw68xr6hiG1QHCbT-uhI68B6J3gVwCey1EflEGKupZqDNUf3X7YIpXxp1tmHcXvpG40JnvzR-FPHpa-0BEi32FGcQa6G8YWZ1eca1EJTFoSYyNjXk_0cWSTR4=w599-h671-s-no?authuser=0" alt="Image" width="450" height="495"></td>
</tr>
</table>

## Вимоги

Для успішного розгортання і запуску проєкту вам знадобляться такі бібліотеки, перелічені у файлі `requirements.txt`:

- matplotlib
- scikit-learn

## Установка

1. Клонуйте репозіторій проекту на ваш локальний комп'ютер:
    ```bash
    git clone https://github.com/Dogherty/ClusterAnalysis.git
      
2. Встановіть залежності Python, зазначені в requirements.txt:
     ```bash
       pip install -r requirements.txt
