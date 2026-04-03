import os
import numpy as np  # Импорт библиотеки NumPy для работы с массивами и математическими функциями

try:
    import matplotlib.pyplot as plt  # Импорт библиотеки Matplotlib для построения графиков
    from mpl_toolkits.mplot3d import Axes3D  # Импорт модуля для 3D графиков
    from matplotlib.animation import FuncAnimation  # Импорт модуля для создания анимаций
    MATPLOTLIB_AVAILABLE = True
except Exception:
    plt = None
    Axes3D = None
    FuncAnimation = None
    MATPLOTLIB_AVAILABLE = False

try:
    from vispy import app, scene
    VISPY_AVAILABLE = True
except Exception:
    app = None
    scene = None
    VISPY_AVAILABLE = False

# Исходные данные моего варианта
mu = 398600.5  # Гравитационный параметр Земли в км³/с²
t0 = 10 + 29 / 60 + 32.94 / 3600  # Начальное время наблюдения в часах (перевод из часов, минут, секунд в десятичные часы)
FIRST_COSMIC_VELOCITY = 7.9  # 1-я космическая скорость, км/с
SECOND_COSMIC_VELOCITY = 11.2  # 2-я космическая скорость, км/с
EPSILON = 1e-10  # Порог для сравнения с нулем

# Координаты и скорости спутника в заданный момент времени
x, y, z = 770.42, 4708.93, 6699.08  # Координаты спутника в геоцентрической системе (км)
x_dot, y_dot, z_dot = 6.7609253, -0.9665345, 1.3605900  # Компоненты вектора скорости спутника (км/с)


def get_speed_classification(speed, first_cosmic, second_cosmic):
    """
    Классификация траектории по текущей скорости относительно 1-й и 2-й космических
    """
    if speed < first_cosmic:
        return "Скорость меньше 1-й космической: спутник не выходит на орбиту и падает обратно на Землю."
    if speed < second_cosmic:
        return "Скорость от 1-й до 2-й космической: спутник остается на орбите вокруг Земли."
    return "Скорость больше или равна 2-й космической: спутник покидает орбиту Земли."


def get_flight_direction(x_dot, y_dot, z_dot):
    """
    Направление полета по вектору скорости
    """
    speed = np.sqrt(x_dot ** 2 + y_dot ** 2 + z_dot ** 2)
    if abs(speed) < EPSILON:
        return "Направление не определено (нулевая скорость)."

    dir_x, dir_y, dir_z = x_dot / speed, y_dot / speed, z_dot / speed
    return f"Направление полета (единичный вектор): ({dir_x:.4f}, {dir_y:.4f}, {dir_z:.4f})"


def get_velocity_with_user_speed(x_dot, y_dot, z_dot):
    """
    Интерфейс ввода скорости взлета: пользователь может задать новый модуль скорости
    """
    current_speed = np.sqrt(x_dot ** 2 + y_dot ** 2 + z_dot ** 2)
    print(f"\nТекущий модуль скорости запуска: {current_speed:.6f} км/с")
    user_input = input("Введите скорость взлета спутника (км/с) или нажмите Enter, чтобы оставить текущее значение: ").strip()

    if not user_input:
        return x_dot, y_dot, z_dot, current_speed

    try:
        new_speed = float(user_input)
        if new_speed <= 0:
            print("Введено неположительное значение, используется исходная скорость.")
            return x_dot, y_dot, z_dot, current_speed
    except ValueError:
        print("Некорректный ввод, используется исходная скорость.")
        return x_dot, y_dot, z_dot, current_speed

    scale = new_speed / current_speed
    return x_dot * scale, y_dot * scale, z_dot * scale, new_speed


def calculate_orbit_parameters(x, y, z, x_dot, y_dot, z_dot, mu):
    """
    Расчет всех параметров орбиты по заданным координатам и скоростям
    """
    # 1. Модули векторов положения и скорости
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)  # Расстояние от центра Земли до спутника
    r_dot = np.sqrt(x_dot ** 2 + y_dot ** 2 + z_dot ** 2)  # Модуль вектора скорости

    # 2. Константа энергии движения (удельная механическая энергия)
    h_energy = r_dot ** 2 - 2 * mu / r  # Интеграл энергии

    # 3. Вектор момента количества движения (интеграл площадей)
    c_x = y * z_dot - z * y_dot  # X-компонента векторного произведения r × v
    c_y = z * x_dot - x * z_dot  # Y-компонента векторного произведения r × v
    c_z = x * y_dot - y * x_dot  # Z-компонента векторного произведения r × v
    c = np.sqrt(c_x ** 2 + c_y ** 2 + c_z ** 2)  # Модуль вектора момента количества движения

    # 4. Вектор Лапласа (вектор эксцентриситета)
    lambda_x = -(mu / r) * x + c_z * y_dot - c_y * z_dot  # X-компонента вектора Лапласа
    lambda_y = -(mu / r) * y + c_x * z_dot - c_z * x_dot  # Y-компонента вектора Лапласа
    lambda_z = -(mu / r) * z + c_y * x_dot - c_x * y_dot  # Z-компонента вектора Лапласа
    lambda_val = np.sqrt(lambda_x ** 2 + lambda_y ** 2 + lambda_z ** 2)  # Модуль вектора Лапласа

    # 5. Элементы формы и размеров орбиты
    e = lambda_val / mu  # Эксцентриситет орбиты (мера вытянутости)
    p = c ** 2 / mu  # Фокальный параметр (параметр орбиты)
    a = p / (1 - e ** 2) if e != 1 else p  # Большая полуось (для эллипса a>0)
    b = a * np.sqrt(1 - e ** 2) if e < 1 else a * np.sqrt(e ** 2 - 1)  # Малая полуось
    r_pi = a * (1 - e)  # Расстояние перигея (ближайшая точка к Земле)
    r_a = a * (1 + e)  # Расстояние апогея (самая дальняя точка от Земли)

    # 6. Среднее движение и период обращения
    n = np.sqrt(mu / a ** 3)  # Среднее движение в радианах в секунду
    n_deg = n * 180 / np.pi  # Среднее движение в градусах в секунду
    P = 2 * np.pi / n  # Период обращения спутника в секундах

    # 7. Элементы ориентировки орбиты в пространстве
    i = np.arccos(c_z / c)  # Наклонение орбиты (угол между плоскостью орбиты и экватором)

    # Долгота восходящего узла (угол от точки весеннего равноденствия до восходящего узла)
    Omega = np.arctan2(c_x, -c_y)  # Используем арктангенс с двумя аргументами для правильного квадранта
    if Omega < 0:  # Корректируем угол, если он отрицательный
        Omega += 2 * np.pi  # Приводим к диапазону [0, 2π)

    # Аргумент перицентра (угол от восходящего узла до перицентра)
    denominator = c_x * lambda_y - c_y * lambda_x  # Знаменатель для вычисления аргумента перицентра
    if abs(denominator) > 1e-10:  # Проверка на ненулевой знаменатель
        w = np.arctan2(c * lambda_z, denominator)  # Вычисление аргумента перицентра
    else:
        w = 0  # Если знаменатель нулевой, устанавливаем 0
    if w < 0:  # Корректируем угол, если он отрицательный
        w += 2 * np.pi  # Приводим к диапазону [0, 2π)

    # 8. Элементы положения спутника на орбите
    # Аргумент широты (угол от восходящего узла до радиус-вектора спутника)
    denominator_u = y * c_x - x * c_y  # Знаменатель для вычисления аргумента широты
    if abs(denominator_u) > 1e-10:  # Проверка на ненулевой знаменатель
        u = np.arctan2(z * c, denominator_u)  # Вычисление аргумента широты
    else:
        u = 0  # Если знаменатель нулевой, устанавливаем 0
    if u < 0:  # Корректируем угол, если он отрицательный
        u += 2 * np.pi  # Приводим к диапазону [0, 2π)

    # Истинная аномалия (угол от перицентра до радиус-вектора спутника)
    dot_product = x * lambda_x + y * lambda_y + z * lambda_z  # Скалярное произведение r·λ
    v = np.arctan2(c * r * r_dot, dot_product)  # Вычисление истинной аномалии
    if v < 0:  # Корректируем угол, если он отрицательный
        v += 2 * np.pi  # Приводим к диапазону [0, 2π)

    # Эксцентрическая аномалия (вспомогательный угол для вычисления положения)
    E = 2 * np.arctan(np.sqrt((1 - e) / (1 + e)) * np.tan(v / 2))  # Формула связи с истинной аномалией
    if E < 0:  # Корректируем угол, если он отрицательный
        E += 2 * np.pi  # Приводим к диапазону [0, 2π)

    # Средняя аномалия (линейно зависящая от времени)
    M = E - e * np.sin(E)  # Уравнение Кеплера для эллиптической орбиты
    M_deg = M * 180 / np.pi  # Перевод средней аномалии в градусы

    # Время полета от перицентра
    delta_t = M / n if n != 0 else 0  # Время с момента прохождения перицентра
    t_pi = t0 * 3600 - delta_t  # Абсолютное время прохождения перицентра (в секундах)

    # Контроль вычислений (проверка ортогональности векторов)
    c_dot_r = c_x * x + c_y * y + c_z * z  # Скалярное произведение c·r (должно быть 0)
    c_dot_r_dot = c_x * x_dot + c_y * y_dot + c_z * z_dot  # Скалярное произведение c·v (должно быть 0)
    c_dot_lambda = c_x * lambda_x + c_y * lambda_y + c_z * lambda_z  # Скалярное произведение c·λ (должно быть 0)
    lambda_check = c ** 2 * h_energy + mu ** 2  # Проверка тождества λ² = c²·h + μ²

    # Возвращение всех вычисленных параметров в виде словаря
    return {
        'r': r, 'r_dot': r_dot, 'h_energy': h_energy,  # Базовые величины
        'c_x': c_x, 'c_y': c_y, 'c_z': c_z, 'c': c,  # Вектор момента количества движения
        'lambda_x': lambda_x, 'lambda_y': lambda_y, 'lambda_z': lambda_z, 'lambda': lambda_val,  # Вектор Лапласа
        'e': e, 'p': p, 'a': a, 'b': b, 'r_pi': r_pi, 'r_a': r_a, 'n': n_deg, 'n_rad': n, 'P': P,  # Геометрические и временные параметры
        'i': np.degrees(i), 'Omega': np.degrees(Omega), 'w': np.degrees(w),  # Углы ориентировки в градусах
        'u': np.degrees(u), 'v': np.degrees(v), 'E': np.degrees(E), 'M': M_deg, 'M_rad': M,  # Углы положения в градусах
        'delta_t': delta_t, 't_pi': t_pi,  # Временные параметры
        'control': {  # Результаты контрольных вычислений
            'c_dot_r': c_dot_r, 'c_dot_r_dot': c_dot_r_dot,
            'c_dot_lambda': c_dot_lambda, 'lambda_check': lambda_check,
            'a_check': -mu / h_energy  # Альтернативный расчет большой полуоси
        }
    }


def select_render_backend():
    """
    Выбор backend отрисовки:
    - auto: VisPy (GPU) при наличии, иначе Matplotlib (CPU)
    - vispy/gpu: принудительно VisPy
    - matplotlib/cpu: принудительно Matplotlib
    """
    requested = os.getenv("ORBIT_RENDER_BACKEND", "auto").strip().lower()

    if requested in {"vispy", "gpu"}:
        if VISPY_AVAILABLE:
            return "vispy"
        print("VisPy недоступен, используется Matplotlib.")
        return "matplotlib"

    if requested in {"matplotlib", "cpu"}:
        return "matplotlib"

    if VISPY_AVAILABLE:
        return "vispy"
    return "matplotlib"


def get_orbit_points_3d(params, points_count=2048):
    """
    Генерация точек орбиты в 3D для рендеринга
    """
    a, e = params['a'], params['e']
    Omega_rad = np.radians(params['Omega'])
    i_rad = np.radians(params['i'])
    w_rad = np.radians(params['w'])

    rotation_matrix = np.array([
        [np.cos(w_rad) * np.cos(Omega_rad) - np.sin(w_rad) * np.cos(i_rad) * np.sin(Omega_rad),
         -np.sin(w_rad) * np.cos(Omega_rad) - np.cos(w_rad) * np.cos(i_rad) * np.sin(Omega_rad),
         np.sin(i_rad) * np.sin(Omega_rad)],
        [np.cos(w_rad) * np.sin(Omega_rad) + np.sin(w_rad) * np.cos(i_rad) * np.cos(Omega_rad),
         -np.sin(w_rad) * np.sin(Omega_rad) + np.cos(w_rad) * np.cos(i_rad) * np.cos(Omega_rad),
         -np.sin(i_rad) * np.cos(Omega_rad)],
        [np.sin(w_rad) * np.sin(i_rad),
         np.cos(w_rad) * np.sin(i_rad),
         np.cos(i_rad)]
    ])

    theta_full = np.linspace(0, 2 * np.pi, points_count)
    r_full = a * (1 - e ** 2) / (1 + e * np.cos(theta_full))
    x_orb_plane = r_full * np.cos(theta_full)
    y_orb_plane = r_full * np.sin(theta_full)
    z_orb_plane = np.zeros_like(x_orb_plane)

    orbit_points = np.vstack([x_orb_plane, y_orb_plane, z_orb_plane])
    orbit_3d = (rotation_matrix @ orbit_points).T
    return orbit_3d


def plot_flat_orbit(params):
    """
    Построение плоского чертежа орбиты с анимацией
    """
    # Извлечение параметров из словаря
    a, e, i, Omega, w, v, b, p = params['a'], params['e'], params['i'], params['Omega'], params['w'], params['v'], \
        params['b'], params['p']

    # Построение точек для полной орбиты
    theta_full = np.linspace(0, 2 * np.pi, 200)  # Массив углов от 0 до 2π
    r_full = a * (1 - e ** 2) / (1 + e * np.cos(theta_full))  # Радиус для каждого угла (уравнение конического сечения в полярных координатах)
    x_orb_full = r_full * np.cos(theta_full)  # X-координаты точек орбиты
    y_orb_full = r_full * np.sin(theta_full)  # Y-координаты точек орбиты

    # Создание фигуры и осей для 2D графика
    fig, ax = plt.subplots(figsize=(16, 12))  # Создание фигуры размером 16x12 дюймов

    # Орбита
    ax.plot(x_orb_full, y_orb_full, 'b-', linewidth=2, label='Орбита')  # Синяя линия орбиты

    # Центральное тело (Земля)
    ax.plot(0, 0, 'yo', markersize=20, label='Земля')  # Желтый круг в центре

    # Перигей и апогей
    perigee_x = a * (1 - e)  # X-координата перигея
    apogee_x = -a * (1 + e)  # X-координата апогея
    ax.plot(perigee_x, 0, 'ro', markersize=8, label='Перигей')  # Красная точка перигея
    ax.plot(apogee_x, 0, 'go', markersize=8, label='Апогей')  # Зеленая точка апогея

    # Обозначения радиусов-векторов
    ax.plot([0, perigee_x], [0, 0], 'r-', alpha=0.5, linewidth=1)  # Линия от центра до перигея
    ax.text(perigee_x / 2, -a * 0.1, 'r_π', fontsize=10, color='black', ha='center', fontweight='bold')  # Подпись для перигея

    ax.plot([0, apogee_x], [0, 0], 'g-', alpha=0.5, linewidth=1)  # Линия от центра до апогея
    ax.text(apogee_x / 2, a * 0.1, 'r_a', fontsize=10, color='green', ha='center', fontweight='bold')  # Подпись для апогея

    # Фокусы эллипса
    c_focus = a * e  # Расстояние от центра до фокуса
    ax.plot(c_focus, 0, 'k+', markersize=12, label='Фокус F')  # Черный крест для фокуса F
    ax.plot(-c_focus, 0, 'k+', markersize=12, label='Фокус F\'')  # Черный крест для фокуса F'

    # Оси координат
    axis_length = max(a * (1 + e), params['r_a']) * 1.2  # Длина осей с запасом
    ax.arrow(0, 0, axis_length, 0, head_width=axis_length * 0.03, head_length=axis_length * 0.05,
             fc='red', ec='red', linewidth=2, label='Ось X')  # Красная стрелка для оси X
    ax.arrow(0, 0, 0, axis_length, head_width=axis_length * 0.03, head_length=axis_length * 0.05,
             fc='green', ec='green', linewidth=2, label='Ось Y')  # Зеленая стрелка для оси Y

    # Подписи осей координат
    ax.text(axis_length * 1.05, 0, 'X', fontsize=14, color='red', fontweight='bold')  # Подпись оси X
    ax.text(0, axis_length * 1.05, 'Y', fontsize=14, color='green', fontweight='bold')  # Подпись оси Y

    # Гринвич на оси X
    ax.plot(axis_length * 0.8, 0, 'b*', markersize=15, label='Гринвич (Г)')  # Синяя звезда для Гринвича
    ax.text(axis_length * 0.8, -axis_length * 0.05, 'Г', fontsize=12, color='blue',
            fontweight='bold', ha='center')  # Подпись Гринвича

    # Большая и малая полуоси из центра эллипса
    center_x = -c_focus  # X-координата центра эллипса
    ax.plot(center_x, 0, 'kx', markersize=10, label='Центр орбиты O')  # Черный крестик для центра орбиты

    # Большая полуось из центра
    ax.plot([center_x - a, center_x + a], [0, 0], 'r--', alpha=0.7, linewidth=2, label='Большая полуось (a)')  # Красная пунктирная линия
    # Малая полуось из центра
    ax.plot([center_x, center_x], [-b, b], 'g--', alpha=0.7, linewidth=2, label='Малая полуось (b)')  # Зеленая пунктирная линия

    # Подписи полуосей из центра
    ax.text(center_x + a / 2, -axis_length * 0.05, 'a', fontsize=11, color='red',
            ha='center', va='top', fontweight='bold', bbox=dict(boxstyle="round,pad=0.3", facecolor='red', alpha=0.1))  # Подпись большой полуоси
    ax.text(center_x + axis_length * 0.05, b / 2, 'b', fontsize=11, color='green',
            ha='left', va='center', fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='green', alpha=0.1))  # Подпись малой полуоси

    # Линия узлов (направление восходящего узла) - проходит через всю орбиту
    Omega_rad = np.radians(Omega)  # Перевод долготы узла в радианы

    # Точки пересечения линии узлов с орбитой
    theta_asc = Omega_rad  # Угол для восходящего узла
    theta_desc = Omega_rad + np.pi  # Угол для нисходящего узла (на противоположной стороне)

    # Координаты узлов на орбите
    r_asc = a * (1 - e ** 2) / (1 + e * np.cos(theta_asc - np.radians(w)))  # Радиус для восходящего узла
    r_desc = a * (1 - e ** 2) / (1 + e * np.cos(theta_desc - np.radians(w)))  # Радиус для нисходящего узла

    x_asc = r_asc * np.cos(theta_asc)  # X-координата восходящего узла
    y_asc = r_asc * np.sin(theta_asc)  # Y-координата восходящего узла
    x_desc = r_desc * np.cos(theta_desc)  # X-координата нисходящего узла
    y_desc = r_desc * np.sin(theta_desc)  # Y-координата нисходящего узла

    # Линия узлов через всю орбиту
    node_length = max(r_asc, r_desc) * 1.5  # Длина линии узлов с запасом
    x_node_line = node_length * np.cos(Omega_rad)  # X-координата конца линии узлов
    y_node_line = node_length * np.sin(Omega_rad)  # Y-координата конца линии узлов

    ax.plot([-x_node_line, x_node_line], [-y_node_line, y_node_line], 'c-', alpha=0.8, linewidth=3, label='Линия узлов')  # Голубая линия узлов

    # Узлы на орбите
    ax.plot([x_asc], [y_asc], 'co', markersize=8, label='Восходящий узел ☊')  # Голубой круг для восходящего узла
    ax.plot([x_desc], [y_desc], 'co', markersize=8, label='Нисходящий узел')  # Голубой круг для нисходящего узла

    # Угол от Гринвича до линии узлов (Ω)
    arc_Omega = np.linspace(0, Omega_rad, 30)  # Массив углов для дуги
    r_arc_Omega = axis_length * 0.3  # Радиус дуги
    x_arc_Omega = r_arc_Omega * np.cos(arc_Omega)  # X-координаты дуги
    y_arc_Omega = r_arc_Omega * np.sin(arc_Omega)  # Y-координаты дуги
    ax.plot(x_arc_Omega, y_arc_Omega, 'c-', linewidth=2)  # Рисование дуги
    ax.text(r_arc_Omega * np.cos(Omega_rad / 2), r_arc_Omega * np.sin(Omega_rad / 2),
            'Ω', fontsize=11, color='darkblue', fontweight='bold')  # Подпись угла Ω

    # Линия апсид (большая ось)
    w_rad = np.radians(w)  # Перевод аргумента перигея в радианы
    apsides_length = axis_length * 0.6  # Длина линии апсид
    x_apsides = apsides_length * np.cos(Omega_rad + w_rad)  # X-координата конца линии апсид
    y_apsides = apsides_length * np.sin(Omega_rad + w_rad)  # Y-координата конца линии апсид
    ax.plot([0, x_apsides], [0, y_apsides], 'y-', alpha=0.7, linewidth=2, label='Линия апсид')  # Желтая линия апсид

    # Аргумент перигея (ω) - от линии узлов до линии апсид
    arc_w = np.linspace(Omega_rad, Omega_rad + w_rad, 30)  # Массив углов для дуги ω
    r_arc_w = axis_length * 0.25  # Радиус дуги
    x_arc_w = r_arc_w * np.cos(arc_w)  # X-координаты дуги
    y_arc_w = r_arc_w * np.sin(arc_w)  # Y-координаты дуги
    ax.plot(x_arc_w, y_arc_w, 'y-', linewidth=2)  # Рисование дуги
    ax.text(r_arc_w * np.cos(Omega_rad + w_rad / 2), r_arc_w * np.sin(Omega_rad + w_rad / 2),
            'ω', fontsize=11, color='darkorange', fontweight='bold')  # Подпись угла ω

    # Инициализация анимированных элементов
    satellite, = ax.plot([], [], 'mo', markersize=10, label='Спутник S')  # Фиолетовая точка спутника
    trail, = ax.plot([], [], 'r-', alpha=0.5, linewidth=1)  # Красная линия траектории
    velocity_arrow = ax.quiver([], [], [], [], color='red', scale=1e5, width=0.005)  # Стрелка вектора скорости
    radius_line, = ax.plot([], [], 'm-', alpha=0.5, linewidth=1)  # Линия радиуса-вектора

    # Аргумент широты (u) - от линии узлов до спутника (анимированный)
    arc_u_line, = ax.plot([], [], 'm-', linewidth=2)  # Линия дуги аргумента широты
    arc_u_text = ax.text(0, 0, 'u', fontsize=10, color='magenta', fontweight='bold')  # Текст "u"

    # Истинная аномалия (ν) - от линии апсид до спутника (анимированный)
    arc_v_line, = ax.plot([], [], 'r-', linewidth=2)  # Линия дуги истинной аномалии
    arc_v_text = ax.text(0, 0, 'ν', fontsize=10, color='red', fontweight='bold')  # Текст "ν"

    # Траектория (хранилище для координат)
    trail_x, trail_y = [], []  # Пустые списки для хранения координат траектории

    def animate(frame):
        # Функция анимации, вызывается для каждого кадра
        # Текущее положение спутника
        theta_current = frame * 2 * np.pi / len(theta_full)  # Текущий угол на орбите
        r_current = a * (1 - e ** 2) / (1 + e * np.cos(theta_current))  # Текущий радиус
        x_current = r_current * np.cos(theta_current)  # Текущая X-координата
        y_current = r_current * np.sin(theta_current)  # Текущая Y-координата

        # Обновление положения спутника
        satellite.set_data([x_current], [y_current])  # Установка новых координат спутника

        # Обновление траектории
        trail_x.append(x_current)  # Добавление X-координаты в историю
        trail_y.append(y_current)  # Добавление Y-координаты в историю
        trail.set_data(trail_x, trail_y)  # Обновление линии траектории

        # Обновление вектора скорости
        if frame < len(theta_full) - 1:  # Если не последний кадр
            dx = x_orb_full[frame + 1] - x_orb_full[frame]  # Разность X-координат между кадрами
            dy = y_orb_full[frame + 1] - y_orb_full[frame]  # Разность Y-координат между кадрами
        else:  # Если последний кадр
            dx = x_orb_full[0] - x_orb_full[frame]  # Разность между началом и концом
            dy = y_orb_full[0] - y_orb_full[frame]  # Разность между началом и концом

        scale = axis_length * 0.1 / np.sqrt(dx ** 2 + dy ** 2)  # Масштабирование вектора
        dx *= scale  # Масштабирование X-компоненты
        dy *= scale  # Масштабирование Y-компоненты

        velocity_arrow.set_offsets([[x_current, y_current]])  # Установка начала стрелки
        velocity_arrow.set_UVC(dx, dy)  # Установка направления стрелки

        # Обновление линии радиуса-вектора
        radius_line.set_data([0, x_current], [0, y_current])  # Линия от центра к спутнику

        # Обновление аргумента широты
        u_current = theta_current - Omega_rad  # Текущий аргумент широты
        if u_current < 0:  # Корректировка угла
            u_current += 2 * np.pi  # Приведение к [0, 2π)

        arc_u = np.linspace(Omega_rad, Omega_rad + u_current, 30)  # Углы для дуги
        r_arc_u = axis_length * 0.2  # Радиус дуги
        x_arc_u = r_arc_u * np.cos(arc_u)  # X-координаты дуги
        y_arc_u = r_arc_u * np.sin(arc_u)  # Y-координаты дуги
        arc_u_line.set_data(x_arc_u, y_arc_u)  # Обновление линии дуги
        arc_u_text.set_position((r_arc_u * np.cos(Omega_rad + u_current / 2),
                                 r_arc_u * np.sin(Omega_rad + u_current / 2)))  # Позиция текста "u"

        # Обновление истинной аномалии
        v_current = theta_current - Omega_rad - w_rad  # Текущая истинная аномалия
        if v_current < 0:  # Корректировка угла
            v_current += 2 * np.pi  # Приведение к [0, 2π)

        arc_v = np.linspace(Omega_rad + w_rad, Omega_rad + w_rad + v_current, 30)  # Углы для дуги
        r_arc_v = axis_length * 0.15  # Радиус дуги
        x_arc_v = r_arc_v * np.cos(arc_v)  # X-координаты дуги
        y_arc_v = r_arc_v * np.sin(arc_v)  # Y-координаты дуги
        arc_v_line.set_data(x_arc_v, y_arc_v)  # Обновление линии дуги
        arc_v_text.set_position((r_arc_v * np.cos(Omega_rad + w_rad + v_current / 2),
                                 r_arc_v * np.sin(Omega_rad + w_rad + v_current / 2)))  # Позиция текста "ν"

        # Возврат всех обновленных объектов для анимации
        return satellite, trail, velocity_arrow, radius_line, arc_u_line, arc_v_line, arc_u_text, arc_v_text

    # Настройки оформления плоского графика
    ax.set_xlabel('X (км)', fontsize=12, fontweight='bold')  # Подпись оси X
    ax.set_ylabel('Y (км)', fontsize=12, fontweight='bold')  # Подпись оси Y
    ax.set_title('ПЛОСКИЙ ЧЕРТЕЖ ОРБИТЫ С АНИМАЦИЕЙ ДВИЖЕНИЯ СПУТНИКА', fontsize=16, fontweight='bold', pad=20)  # Заголовок
    ax.grid(True, alpha=0.3)  # Включение сетки с прозрачностью
    ax.axis('equal')  # Равные масштабы по осям
    ax.legend(loc='upper left', bbox_to_anchor=(0, 1))  # Легенда в верхнем левом углу

    # Таблица с элементами орбиты
    textstr = '\n'.join((  # Многострочная строка с параметрами
        '=== ОСНОВНЫЕ ПАРАМЕТРЫ ОРБИТЫ ===',
        f'Большая полуось: a = {a:.2f} км',
        f'Малая полуось: b = {b:.2f} км',
        f'Эксцентриситет: e = {e:.6f}',
        f'Фокальный параметр: p = {params["p"]:.2f} км',
        f'Перигей: r_π = {params["r_pi"]:.2f} км',
        f'Апогей: r_a = {params["r_a"]:.2f} км',
        '',
        '=== ВРЕМЕННЫЕ ПАРАМЕТРЫ ===',
        f'Среднее движение: n = {params["n"]:.6e} °/с',
        f'Период: P = {params["P"] / 3600:.4f} ч',
        f'Момент перицентра: t_π = {params["t_pi"]:.1f} с',
        '',
        '=== ОРИЕНТАЦИЯ ОРБИТЫ ===',
        f'Наклонение: i = {i:.2f}°',
        f'Долгота узла: Ω = {Omega:.2f}°',
        f'Аргумент перигея: ω = {w:.2f}°',
        '',
        '=== ПОЛОЖЕНИЕ СПУТНИКА ===',
        f'Аргумент широты: u = {params["u"]:.2f}°',
        f'Истинная аномалия: ν = {v:.2f}°',
        f'Средняя аномалия: M = {params["M"]:.2f}°',
        f'Эксцентрическая аномалия: E = {params["E"]:.2f}°'
    ))

    # Восходящий узел ☊ (дополнительное отображение)
    ax.plot([x_asc], [y_asc], 'co', markersize=8, label='Восходящий узел ☊')  # Голубой круг
    ax.text(x_asc, y_asc, '  ☊', fontsize=12, color='cyan', fontweight='bold', va='center')  # Символ восходящего узла

    props = dict(boxstyle='round', facecolor='lightblue', alpha=0.9)  # Стиль рамки для текста
    ax.text(0.8, 0.98, textstr, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=props, fontfamily='monospace')  # Текст с параметрами

    # Создание анимации
    anim = FuncAnimation(fig, animate, frames=len(theta_full), interval=50, blit=True, repeat=True)  # Анимация с 50 мс между кадрами

    return fig, ax, anim  # Возврат фигуры, осей и анимации


def plot_spatial_orbit_gpu(params):
    """
    Быстрая 3D отрисовка орбиты на GPU с использованием VisPy/OpenGL
    """
    if not VISPY_AVAILABLE:
        raise RuntimeError("VisPy не установлен: GPU-отрисовка недоступна.")

    orbit_3d = get_orbit_points_3d(params, points_count=4096)
    max_range = float(np.max(np.abs(orbit_3d)) * 1.25)
    earth_radius = 6371.0

    canvas = scene.SceneCanvas(keys='interactive', size=(1600, 1000), bgcolor='#04070f', show=True)
    view = canvas.central_widget.add_view()
    view.camera = scene.cameras.TurntableCamera(fov=45, azimuth=45, elevation=25, distance=max(max_range * 2.2, 25000))

    scene.visuals.XYZAxis(parent=view.scene)
    scene.visuals.Sphere(radius=earth_radius, rows=48, cols=96,
                         color=(0.15, 0.35, 1.0, 0.35), edge_color=None, parent=view.scene)
    scene.visuals.Line(pos=orbit_3d, color=(0.25, 0.75, 1.0, 1.0), width=2.0, parent=view.scene, method='gl')

    trail = scene.visuals.Line(pos=orbit_3d[:1], color=(1.0, 0.35, 0.35, 0.9), width=2.5, parent=view.scene, method='gl')
    satellite = scene.visuals.Markers(parent=view.scene)
    satellite.set_data(orbit_3d[:1], face_color=(1.0, 0.1, 1.0, 1.0), size=12)

    direction_len = max_range * 0.06
    velocity_line = scene.visuals.Line(
        pos=np.array([orbit_3d[0], orbit_3d[0] + np.array([direction_len, 0.0, 0.0])]),
        color=(1.0, 0.2, 0.2, 1.0),
        width=3.0,
        parent=view.scene,
        method='gl'
    )

    frame_idx = 0

    def on_timer(event):
        nonlocal frame_idx
        idx = frame_idx
        next_idx = (idx + 1) % len(orbit_3d)

        current = orbit_3d[idx]
        nxt = orbit_3d[next_idx]
        direction = nxt - current
        norm = np.linalg.norm(direction)
        if norm > EPSILON:
            direction = direction / norm * direction_len

        satellite.set_data(np.array([current]), face_color=(1.0, 0.1, 1.0, 1.0), size=12)
        trail.set_data(orbit_3d[:idx + 1])
        velocity_line.set_data(np.array([current, current + direction]))
        frame_idx = next_idx

    canvas.orbit_timer = app.Timer(interval=1 / 120, connect=on_timer, start=True)
    app.run()


def plot_spatial_orbit_with_animation(params, x, y, z, x_dot, y_dot, z_dot):
    """
    Построение пространственного чертежа орбиты с АНИМАЦИЕЙ движения спутника
    """
    # Извлечение параметров орбиты
    a, e, i, Omega, w, v, b, p = params['a'], params['e'], params['i'], params['Omega'], params['w'], params['v'], \
        params['b'], params['p']

    # Перевод углов в радианы
    Omega_rad = np.radians(Omega)  # Долгота восходящего узла
    i_rad = np.radians(i)  # Наклонение
    w_rad = np.radians(w)  # Аргумент перигея
    u_rad = np.radians(params['u'])  # Аргумент широты
    v_rad = np.radians(v)  # Истинная аномалия

    # Матрица поворота для преобразования из орбитальной плоскости в инерциальную систему
    R = np.array([
        [np.cos(w_rad) * np.cos(Omega_rad) - np.sin(w_rad) * np.cos(i_rad) * np.sin(Omega_rad),
         -np.sin(w_rad) * np.cos(Omega_rad) - np.cos(w_rad) * np.cos(i_rad) * np.sin(Omega_rad),
         np.sin(i_rad) * np.sin(Omega_rad)],
        [np.cos(w_rad) * np.sin(Omega_rad) + np.sin(w_rad) * np.cos(i_rad) * np.cos(Omega_rad),
         -np.sin(w_rad) * np.sin(Omega_rad) + np.cos(w_rad) * np.cos(i_rad) * np.cos(Omega_rad),
         -np.sin(i_rad) * np.cos(Omega_rad)],
        [np.sin(w_rad) * np.sin(i_rad),
         np.cos(w_rad) * np.sin(i_rad),
         np.cos(i_rad)]
    ])

    # Генерация полной орбиты для анимации
    theta_full = np.linspace(0, 2 * np.pi, 200)  # Полный набор углов
    r_full = a * (1 - e ** 2) / (1 + e * np.cos(theta_full))  # Радиусы для всех углов

    # Координаты в орбитальной плоскости
    x_orb_plane = r_full * np.cos(theta_full)  # X в плоскости орбиты
    y_orb_plane = r_full * np.sin(theta_full)  # Y в плоскости орбиты
    z_orb_plane = np.zeros_like(x_orb_plane)  # Z=0 в плоскости орбиты

    # Преобразование в 3D координаты
    orbit_points = np.vstack([x_orb_plane, y_orb_plane, z_orb_plane])  # Объединение в одну матрицу 3xN
    orbit_3d = (R @ orbit_points).T  # Умножение матрицы поворота на координаты и транспонирование

    # Разделение на координаты для отрисовки
    x_3d_full = orbit_3d[:, 0]  # X-координаты в 3D
    y_3d_full = orbit_3d[:, 1]  # Y-координаты в 3D
    z_3d_full = orbit_3d[:, 2]  # Z-координаты в 3D

    # Создание фигуры для 3D графика
    fig = plt.figure(figsize=(20, 14))  # Большая фигура 20x14 дюймов
    ax = fig.add_subplot(111, projection='3d')  # 3D оси

    # Орбита (статическая часть)
    ax.plot(x_3d_full, y_3d_full, z_3d_full, 'b-', linewidth=2, label='Орбита', alpha=0.8)  # Синяя линия орбиты

    # Земля (сфера)
    u_earth = np.linspace(0, 2 * np.pi, 100)  # Углы для параметризации сферы
    v_earth = np.linspace(0, np.pi, 50)  # Углы для параметризации сферы
    x_earth = 6371 * np.outer(np.cos(u_earth), np.sin(v_earth))  # X-координаты Земли (радиус 6371 км)
    y_earth = 6371 * np.outer(np.sin(u_earth), np.sin(v_earth))  # Y-координаты Земли
    z_earth = 6371 * np.outer(np.ones(np.size(u_earth)), np.cos(v_earth))  # Z-координаты Земли
    ax.plot_surface(x_earth, y_earth, z_earth, color='lightblue', alpha=0.3, label='Земля')  # Поверхность Земли

    # Плоскость экватора (плоскость XY)
    max_range = max(max(abs(x_3d_full)), max(abs(y_3d_full)), max(abs(z_3d_full))) * 1.2  # Максимальный диапазон для осей
    xx, yy = np.meshgrid(np.linspace(-max_range, max_range, 20), np.linspace(-max_range, max_range, 20))  # Сетка для плоскости
    zz = np.zeros_like(xx)  # Z=0 для плоскости экватора
    ax.plot_surface(xx, yy, zz, color='yellow', alpha=0.2, label='Плоскость экватора')  # Желтая полупрозрачная плоскость

    # Гринвичский меридиан (плоскость XZ, где Y=0)
    xx_gr, zz_gr = np.meshgrid(np.linspace(-max_range, max_range, 20), np.linspace(-max_range, max_range, 20))  # Сетка
    yy_gr = np.zeros_like(xx_gr)  # Y=0 для меридиана
    ax.plot_surface(xx_gr, yy_gr, zz_gr, color='orange', alpha=0.15, label='Гринвичский меридиан')  # Оранжевая плоскость

    # Гринвич на оси X
    ax.plot([max_range * 0.8], [0], [0], 'b*', markersize=15, label='Гринвич (Г)')  # Синяя звезда
    ax.text(max_range * 0.8, 0, max_range * 0.05, 'Г', fontsize=14, color='blue', fontweight='bold')  # Подпись

    # Координатные оси
    # Ось X (красная)
    ax.quiver(0, 0, 0, max_range, 0, 0, color='red', linewidth=3, arrow_length_ratio=0.1, label='Ось X')  # Красная стрелка
    # Ось Y (зеленая)
    ax.quiver(0, 0, 0, 0, max_range, 0, color='green', linewidth=3, arrow_length_ratio=0.1, label='Ось Y')  # Зеленая стрелка
    # Ось Z (синяя)
    ax.quiver(0, 0, 0, 0, 0, max_range, color='blue', linewidth=3, arrow_length_ratio=0.1, label='Ось Z')  # Синяя стрелка

    # Подписи осей
    ax.text(max_range, 0, 0, 'X', fontsize=14, color='red', fontweight='bold')  # X
    ax.text(0, max_range, 0, 'Y', fontsize=14, color='green', fontweight='bold')  # Y
    ax.text(0, 0, max_range, 'Z', fontsize=14, color='blue', fontweight='bold')  # Z

    # Параметры ориентировки
    # Линия узлов - проходит через всю орбиту
    # Точки пересечения линии узлов с орбитой
    theta_asc = Omega_rad  # Восходящий узел
    theta_desc = Omega_rad + np.pi  # Нисходящий узел

    # Координаты узлов в орбитальной плоскости
    r_asc = a * (1 - e ** 2) / (1 + e * np.cos(theta_asc - w_rad))  # Радиус восходящего узла
    r_desc = a * (1 - e ** 2) / (1 + e * np.cos(theta_desc - w_rad))  # Радиус нисходящего узла

    x_asc_orb = r_asc * np.cos(theta_asc)  # X восходящего узла в плоскости орбиты
    y_asc_orb = r_asc * np.sin(theta_asc)  # Y восходящего узла в плоскости орбиты
    x_desc_orb = r_desc * np.cos(theta_desc)  # X нисходящего узла в плоскости орбиты
    y_desc_orb = r_desc * np.sin(theta_desc)  # Y нисходящего узла в плоскости орбиты

    # Преобразование в 3D
    asc_point = np.array([[x_asc_orb, y_asc_orb, 0]])  # Точка восходящего узла
    desc_point = np.array([[x_desc_orb, y_desc_orb, 0]])  # Точка нисходящего узла

    asc_3d = (R @ asc_point.T).T[0]  # Преобразование в инерциальную систему
    desc_3d = (R @ desc_point.T).T[0]  # Преобразование в инерциальную систему

    # Линия узлов через всю орбиту
    node_line_length = max(r_asc, r_desc) * 1.5  # Длина линии узлов
    x_nodes_line = node_line_length * np.cos(Omega_rad)  # X в плоскости орбиты
    y_nodes_line = node_line_length * np.sin(Omega_rad)  # Y в плоскости орбиты

    node_line_points = np.array([[-x_nodes_line, -y_nodes_line, 0], [x_nodes_line, y_nodes_line, 0]])  # Две точки линии
    node_line_3d = (R @ node_line_points.T).T  # Преобразование в 3D

    ax.plot(node_line_3d[:, 0], node_line_3d[:, 1], node_line_3d[:, 2],
            'c-', alpha=0.8, linewidth=3, label='Линия узлов')  # Голубая линия узлов

    # Точки узлов
    ax.plot([asc_3d[0]], [asc_3d[1]], [asc_3d[2]], 'co', markersize=8, label='Восходящий узел ☊')  # Восходящий узел
    ax.plot([desc_3d[0]], [desc_3d[1]], [desc_3d[2]], 'co', markersize=8, label='Нисходящий узел')  # Нисходящий узел

    # Подпись долготы восходящего узла
    arc_Omega = np.linspace(0, Omega_rad, 30)  # Углы для дуги Ω
    r_arc_Omega = max_range * 0.4  # Радиус дуги
    x_arc_Omega = r_arc_Omega * np.cos(arc_Omega)  # X дуги
    y_arc_Omega = r_arc_Omega * np.sin(arc_Omega)  # Y дуги
    ax.plot(x_arc_Omega, y_arc_Omega, 0, 'c-', linewidth=2)  # Дуга Ω
    ax.text(r_arc_Omega * np.cos(Omega_rad / 2), r_arc_Omega * np.sin(Omega_rad / 2), max_range * 0.05,
            'Ω', fontsize=12, color='darkblue', fontweight='bold')  # Подпись Ω

    # Угол наклонения
    incline_length = max_range * 0.6  # Длина линии наклонения
    z_incline = incline_length * np.sin(i_rad)  # Z-компонента
    xy_incline = incline_length * np.cos(i_rad)  # Проекция на плоскость XY
    x_incline = xy_incline * np.cos(Omega_rad)  # X-компонента
    y_incline = xy_incline * np.sin(Omega_rad)  # Y-компонента
    ax.plot([0, x_incline], [0, y_incline], [0, z_incline], 'm-', alpha=0.8, linewidth=3, label='Наклонение')  # Линия наклонения

    # Угол наклонения (дуга)
    theta_i = np.linspace(0, i_rad, 30)  # Углы для дуги i
    arc_radius = max_range * 0.3  # Радиус дуги
    x_arc_i = arc_radius * np.cos(theta_i) * np.cos(Omega_rad)  # X дуги
    y_arc_i = arc_radius * np.cos(theta_i) * np.sin(Omega_rad)  # Y дуги
    z_arc_i = arc_radius * np.sin(theta_i)  # Z дуги
    ax.plot(x_arc_i, y_arc_i, z_arc_i, 'm-', linewidth=3)  # Дуга i
    ax.text(x_arc_i[15], y_arc_i[15], z_arc_i[15], 'i',
            fontsize=12, color='purple', fontweight='bold')  # Подпись i

    # Центр орбиты и фокусы
    center_offset = a * e  # Смещение центра для эллиптической орбиты
    center_point = np.array([[-center_offset, 0, 0]])  # Точка центра в плоскости орбиты
    center_3d = (R @ center_point.T).T  # Преобразование в 3D
    ax.plot([center_3d[0, 0]], [center_3d[0, 1]], [center_3d[0, 2]], 'kx', markersize=10, label='Центр орбиты O')  # Центр

    # Фокусы
    focus1_point = np.array([[0, 0, 0]])  # Главный фокус (Земля)
    focus1_3d = (R @ focus1_point.T).T  # Преобразование в 3D
    ax.plot([focus1_3d[0, 0]], [focus1_3d[0, 1]], [focus1_3d[0, 2]], 'k+', markersize=12, label='Фокус F')  # Фокус F

    focus2_point = np.array([[-2 * center_offset, 0, 0]])  # Второй фокус
    focus2_3d = (R @ focus2_point.T).T  # Преобразование в 3D
    ax.plot([focus2_3d[0, 0]], [focus2_3d[0, 1]], [focus2_3d[0, 2]], 'k+', markersize=12, label='Фокус F\'')  # Фокус F'

    # Большая полуось в 3D
    major_axis_points = np.array([[-a, 0, 0], [a, 0, 0]])  # Концы большой оси в плоскости орбиты
    major_axis_rotated = (R @ major_axis_points.T).T  # Преобразование в 3D
    ax.plot(major_axis_rotated[:, 0], major_axis_rotated[:, 1], major_axis_rotated[:, 2],
            'r--', linewidth=2, alpha=0.7, label='Большая полуось (a)')  # Красная пунктирная линия

    # Малая полуось в 3D
    minor_axis_points = np.array([[0, -b, 0], [0, b, 0]])  # Концы малой оси в плоскости орбиты
    minor_axis_rotated = (R @ minor_axis_points.T).T  # Преобразование в 3D
    ax.plot(minor_axis_rotated[:, 0], minor_axis_rotated[:, 1], minor_axis_rotated[:, 2],
            'g--', linewidth=2, alpha=0.7, label='Малая полуось (b)')  # Зеленая пунктирная линия

    # Фокальный параметр в 3D
    p_point = np.array([[0, p / (1 + e), 0]])  # Точка на орбите для фокального параметра
    p_point_3d = (R @ p_point.T).T  # Преобразование в 3D
    ax.plot([focus1_3d[0, 0], p_point_3d[0, 0]],
            [focus1_3d[0, 1], p_point_3d[0, 1]],
            [focus1_3d[0, 2], p_point_3d[0, 2]],
            'm-', alpha=0.7, linewidth=2, label='Фокальный параметр (p)')  # Линия фокального параметра

    # Линия апсид в 3D
    apsides_length = max_range * 0.6  # Длина линии апсид
    apsides_point = np.array([[apsides_length * np.cos(w_rad), apsides_length * np.sin(w_rad), 0]])  # Точка в плоскости орбиты
    apsides_3d = (R @ apsides_point.T).T  # Преобразование в 3D
    ax.plot([0, apsides_3d[0, 0]], [0, apsides_3d[0, 1]], [0, apsides_3d[0, 2]],
            'y-', alpha=0.7, linewidth=2, label='Линия апсид')  # Желтая линия апсид

    # Аргумент перигея в 3D
    arc_w_points = []  # Список для точек дуги ω
    for angle in np.linspace(0, w_rad, 30):  # 30 точек для дуги
        arc_point = np.array([[max_range * 0.25 * np.cos(angle), max_range * 0.25 * np.sin(angle), 0]])  # Точка в плоскости орбиты
        arc_w_points.append((R @ arc_point.T).T[0])  # Преобразование и добавление
    arc_w_points = np.array(arc_w_points)  # Преобразование в массив NumPy
    ax.plot(arc_w_points[:, 0], arc_w_points[:, 1], arc_w_points[:, 2], 'y-', linewidth=2)  # Дуга ω
    mid_point = arc_w_points[15]  # Средняя точка дуги
    ax.text(mid_point[0], mid_point[1], mid_point[2], 'ω',
            fontsize=11, color='darkorange', fontweight='bold')  # Подпись ω

    # АНИМИРОВАННЫЕ ЭЛЕМЕНТЫ
    # Спутник
    satellite, = ax.plot([], [], [], 'mo', markersize=10, label='Спутник S')  # Фиолетовая точка

    # Траектория спутника
    trail, = ax.plot([], [], [], 'r-', alpha=0.5, linewidth=1)  # Красная линия траектории

    # Вектор скорости (инициализация)
    velocity_arrow = None  # Будет создаваться динамически

    # Радиус-вектор
    radius_line, = ax.plot([], [], [], 'm-', alpha=0.6, linewidth=2, label='Радиус-вектор')  # Линия от центра к спутнику

    # Аргумент широты (анимированный)
    arc_u_line, = ax.plot([], [], [], 'm-', linewidth=2)  # Линия дуги u
    arc_u_text = ax.text(0, 0, 0, 'u', fontsize=10, color='magenta', fontweight='bold')  # Текст "u"

    # Истинная аномалия (анимированный)
    arc_v_line, = ax.plot([], [], [], 'r-', linewidth=2)  # Линия дуги ν
    arc_v_text = ax.text(0, 0, 0, 'ν', fontsize=10, color='red', fontweight='bold')  # Текст "ν"

    # Хранилище для траектории
    trail_x, trail_y, trail_z = [], [], []  # Списки для хранения координат траектории

    def update_3d(frame):
        nonlocal velocity_arrow  # Объявление нелокальной переменной для изменения velocity_arrow

        # Текущее положение в орбитальной плоскости
        theta_current = frame * 2 * np.pi / len(theta_full)  # Текущий угол
        r_current = a * (1 - e ** 2) / (1 + e * np.cos(theta_current))  # Текущий радиус

        # Координаты в орбитальной плоскости
        x_orb_current = r_current * np.cos(theta_current)  # X в плоскости орбиты
        y_orb_current = r_current * np.sin(theta_current)  # Y в плоскости орбиты
        z_orb_current = 0  # Z=0 в плоскости орбиты

        # Преобразование в 3D
        current_point = np.array([[x_orb_current, y_orb_current, z_orb_current]])  # Текущая точка
        current_3d = (R @ current_point.T).T[0]  # Преобразование в инерциальную систему

        x_current, y_current, z_current = current_3d  # Распаковка координат

        # Обновление положения спутника
        satellite.set_data([x_current], [y_current])  # Установка 2D координат
        satellite.set_3d_properties([z_current])  # Установка Z-координаты

        # Обновление траектории
        trail_x.append(x_current)  # Добавление X в историю
        trail_y.append(y_current)  # Добавление Y в историю
        trail_z.append(z_current)  # Добавление Z в историю
        trail.set_data(trail_x, trail_y)  # Обновление 2D координат линии
        trail.set_3d_properties(trail_z)  # Обновление Z-координат линии

        # Вычисление вектора скорости (касательного к орбите)
        if frame < len(theta_full) - 1:  # Если не последний кадр
            dx = x_3d_full[frame + 1] - x_3d_full[frame]  # Разность X между кадрами
            dy = y_3d_full[frame + 1] - y_3d_full[frame]  # Разность Y между кадрами
            dz = z_3d_full[frame + 1] - z_3d_full[frame]  # Разность Z между кадрами
        else:  # Если последний кадр
            dx = x_3d_full[0] - x_3d_full[frame]  # Разность между началом и концом
            dy = y_3d_full[0] - y_3d_full[frame]  # Разность между началом и концом
            dz = z_3d_full[0] - z_3d_full[frame]  # Разность между началом и концом

        # Масштабирование вектора скорости
        scale = max_range * 0.05 / np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)  # Коэффициент масштабирования
        dx *= scale  # Масштабирование X-компоненты
        dy *= scale  # Масштабирование Y-компоненты
        dz *= scale  # Масштабирование Z-компоненты

        # Удаление старого вектора скорости, если он существует
        if velocity_arrow is not None:
            velocity_arrow.remove()  # Удаление предыдущей стрелки

        # Создание нового вектора скорости
        velocity_arrow = ax.quiver(x_current, y_current, z_current,
                                   dx, dy, dz,
                                   color='red', arrow_length_ratio=0.2,
                                   linewidth=2)  # Новая стрелка скорости

        # Обновление радиуса-вектора
        radius_line.set_data([0, x_current], [0, y_current])  # 2D координаты линии
        radius_line.set_3d_properties([0, z_current])  # Z-координаты линии

        # Обновление аргумента широты
        u_current = theta_current - Omega_rad  # Текущий аргумент широты
        if u_current < 0:  # Корректировка угла
            u_current += 2 * np.pi  # Приведение к [0, 2π)

        # Генерация дуги для аргумента широты
        arc_u_angles = np.linspace(Omega_rad, Omega_rad + u_current, 30)  # Углы для дуги
        r_arc_u = max_range * 0.2  # Радиус дуги
        x_arc_u_plane = r_arc_u * np.cos(arc_u_angles)  # X в плоскости орбиты
        y_arc_u_plane = r_arc_u * np.sin(arc_u_angles)  # Y в плоскости орбиты
        z_arc_u_plane = np.zeros_like(x_arc_u_plane)  # Z=0 в плоскости орбиты

        arc_u_points = np.vstack([x_arc_u_plane, y_arc_u_plane, z_arc_u_plane])  # Объединение координат
        arc_u_3d = (R @ arc_u_points).T  # Преобразование в 3D

        arc_u_line.set_data(arc_u_3d[:, 0], arc_u_3d[:, 1])  # Обновление 2D координат дуги
        arc_u_line.set_3d_properties(arc_u_3d[:, 2])  # Обновление Z-координат дуги

        # Обновление позиции текста аргумента широты
        mid_u_idx = len(arc_u_angles) // 2  # Индекс средней точки
        arc_u_text.set_position((arc_u_3d[mid_u_idx, 0], arc_u_3d[mid_u_idx, 1]))  # Установка 2D позиции
        arc_u_text.set_3d_properties(arc_u_3d[mid_u_idx, 2], 'z')  # Установка Z-позиции

        # Обновление истинной аномалии
        v_current = theta_current - Omega_rad - w_rad  # Текущая истинная аномалия
        if v_current < 0:  # Корректировка угла
            v_current += 2 * np.pi  # Приведение к [0, 2π)

        # Генерация дуги для истинной аномалии
        arc_v_angles = np.linspace(Omega_rad + w_rad, Omega_rad + w_rad + v_current, 30)  # Углы для дуги
        r_arc_v = max_range * 0.15  # Радиус дуги
        x_arc_v_plane = r_arc_v * np.cos(arc_v_angles)  # X в плоскости орбиты
        y_arc_v_plane = r_arc_v * np.sin(arc_v_angles)  # Y в плоскости орбиты
        z_arc_v_plane = np.zeros_like(x_arc_v_plane)  # Z=0 в плоскости орбиты

        arc_v_points = np.vstack([x_arc_v_plane, y_arc_v_plane, z_arc_v_plane])  # Объединение координат
        arc_v_3d = (R @ arc_v_points).T  # Преобразование в 3D

        arc_v_line.set_data(arc_v_3d[:, 0], arc_v_3d[:, 1])  # Обновление 2D координат дуги
        arc_v_line.set_3d_properties(arc_v_3d[:, 2])  # Обновление Z-координат дуги

        # Обновление позиции текста истинной аномалии
        mid_v_idx = len(arc_v_angles) // 2  # Индекс средней точки
        arc_v_text.set_position((arc_v_3d[mid_v_idx, 0], arc_v_3d[mid_v_idx, 1]))  # Установка 2D позиции
        arc_v_text.set_3d_properties(arc_v_3d[mid_v_idx, 2], 'z')  # Установка Z-позиции

        # Возврат всех обновленных объектов для анимации
        return [satellite, trail, radius_line, arc_u_line, arc_v_line, arc_u_text, arc_v_text]

    # Настройки 3D графика
    ax.set_xlabel('X (км)', fontsize=12, fontweight='bold')  # Подпись оси X
    ax.set_ylabel('Y (км)', fontsize=12, fontweight='bold')  # Подпись оси Y
    ax.set_zlabel('Z (км)', fontsize=12, fontweight='bold')  # Подпись оси Z
    ax.set_title('ПРОСТРАНСТВЕННЫЙ ЧЕРТЕЖ ОРБИТЫ С АНИМАЦИЕЙ ДВИЖЕНИЯ СПУТНИКА',
                 fontsize=16, fontweight='bold', pad=20)  # Заголовок

    # Таблица с элементами орбиты
    textstr = '\n'.join((  # Многострочная строка
        '=== ОСНОВНЫЕ ПАРАМЕТРЫ ОРБИТЫ ===',
        f'Большая полуось: a = {a:.2f} км',
        f'Малая полуось: b = {b:.2f} км',
        f'Эксцентриситет: e = {e:.6f}',
        f'Фокальный параметр: p = {params["p"]:.2f} км',
        f'Перигей: r_π = {params["r_pi"]:.2f} км',
        f'Апогей: r_a = {params["r_a"]:.2f} км',
        '',
        '=== ВРЕМЕННЫЕ ПАРАМЕТРЫ ===',
        f'Среднее движение: n = {params["n"]:.6e} °/с',
        f'Период: P = {params["P"] / 3600:.4f} ч',
        f'Момент перицентра: t_π = {params["t_pi"]:.1f} с',
        '',
        '=== ОРИЕНТАЦИЯ ОРБИТЫ ===',
        f'Наклонение: i = {i:.2f}°',
        f'Долгота узла: Ω = {Omega:.2f}°',
        f'Аргумент перигея: ω = {w:.2f}°',
        '',
        '=== ПОЛОЖЕНИЕ СПУТНИКА ===',
        f'Аргумент широты: u = {params["u"]:.2f}°',
        f'Истинная аномалия: ν = {v:.2f}°',
        f'Средняя аномалия: M = {params["M"]:.2f}°',
        f'Эксцентрическая аномалия: E = {params["E"]:.2f}°'
    ))

    # Восходящий узел ☊
    ax.plot([asc_3d[0]], [asc_3d[1]], [asc_3d[2]], 'co', markersize=8, label='Восходящий узел ☊')  # Голубой круг
    # Подпись восходящего узла
    ax.text(asc_3d[0], asc_3d[1], asc_3d[2] + max_range * 0.05, '☊',
            fontsize=14, color='cyan', fontweight='bold', ha='center', va='center')  # Символ ☊

    # Текстовая таблица с параметрами
    ax.text2D(0.8, 0.98, textstr, transform=ax.transAxes, fontsize=9,
              verticalalignment='top',
              bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.9),
              fontfamily='monospace')  # Текст в 2D координатах фигуры

    ax.legend(loc='upper left', bbox_to_anchor=(0, 1))  # Легенда в верхнем левом углу

    # Установка равного масштаба по осям
    ax.set_xlim([-max_range, max_range])  # Ограничение по X
    ax.set_ylim([-max_range, max_range])  # Ограничение по Y
    ax.set_zlim([-max_range, max_range])  # Ограничение по Z

    # Создание анимации
    anim_3d = FuncAnimation(fig, update_3d, frames=len(theta_full),
                            interval=50, blit=True, repeat=True)  # Анимация с 50 мс между кадрами

    return fig, ax, anim_3d  # Возврат фигуры, осей и анимации


# Основная программа
if __name__ == "__main__":
    # Интерфейс изменения скорости взлета спутника
    x_dot, y_dot, z_dot, selected_speed = get_velocity_with_user_speed(x_dot, y_dot, z_dot)

    # Фиксированные 1-я и 2-я космические скорости (км/с)
    first_cosmic = FIRST_COSMIC_VELOCITY
    second_cosmic = SECOND_COSMIC_VELOCITY
    trajectory_type = get_speed_classification(selected_speed, first_cosmic, second_cosmic)
    flight_direction = get_flight_direction(x_dot, y_dot, z_dot) if selected_speed >= second_cosmic else None

    # Расчет параметров орбиты
    params = calculate_orbit_parameters(x, y, z, x_dot, y_dot, z_dot, mu)  # Вызов функции расчета

    # Вывод результатов в консоль
    print("=" * 80)  # Разделительная линия
    print("РЕЗУЛЬТАТЫ РАСЧЕТА ПАРАМЕТРОВ ОРБИТЫ МОЕГО ВАРИАНТА")  # Заголовок
    print("=" * 80)  # Разделительная линия

    print("\n1. ИСХОДНЫЕ ДАННЫЕ:")  # Раздел 1
    print(f"   Координаты: r = ({x}, {y}, {z}) км")  # Координаты
    print(f"   Скорости: r' = ({x_dot}, {y_dot}, {z_dot}) км/с")  # Скорости
    print(f"   Скорость взлета |v| = {selected_speed:.6f} км/с")  # Скорость взлета
    print(f"   1-я космическая v1 = {first_cosmic:.6f} км/с")  # Первая космическая
    print(f"   2-я космическая v2 = {second_cosmic:.6f} км/с")  # Вторая космическая
    print(f"   Тип траектории: {trajectory_type}")  # Классификация траектории
    if flight_direction:
        print(f"   {flight_direction}")  # Направление полета при скорости >= 2-й космической
    print(f"   Время t₀ = {t0:.4f} ч")  # Время

    print("\n2. ПЕРВЫЕ ИНТЕГРАЛЫ И КОНСТАНТЫ:")  # Раздел 2
    print(f"   r = {params['r']:.4f} км")  # Радиус
    print(f"   r' = {params['r_dot']:.4f} км/с")  # Скорость
    print(f"   h = {params['h_energy']:.4f} км²/с²")  # Удельная энергия
    print(f"   c = ({params['c_x']:.4f}, {params['c_y']:.4f}, {params['c_z']:.4f})")  # Вектор момента
    print(f"   |c| = {params['c']:.4f}")  # Модуль момента
    print(f"   λ = ({params['lambda_x']:.4f}, {params['lambda_y']:.4f}, {params['lambda_z']:.4f})")  # Вектор Лапласа
    print(f"   |λ| = {params['lambda']:.4f}")  # Модуль вектора Лапласа

    print("\n3. РАЗМЕРЫ, ФОРМЫ И ДВИЖЕНИЕ:")  # Раздел 3
    print(f"   Эксцентриситет e = {params['e']:.6f}")  # Эксцентриситет
    print(f"   Фокальный параметр p = {params['p']:.4f} км")  # Фокальный параметр
    print(f"   Большая полуось a = {params['a']:.4f} км")  # Большая полуось
    print(f"   Малая полуось b = {params['b']:.4f} км")  # Малая полуось
    print(f"   Перигей r_π = {params['r_pi']:.4f} км")  # Перигей
    print(f"   Апогей r_a = {params['r_a']:.4f} км")  # Апогей
    print(f"   Среднее движение n = {params['n']:.8f} °/с")  # Среднее движение
    print(f"   Период P = {params['P']:.4f} с = {params['P'] / 3600:.4f} ч")  # Период

    print("\n4. ОРИЕНТАЦИЯ ОРБИТЫ:")  # Раздел 4
    print(f"   Наклонение i = {params['i']:.4f}°")  # Наклонение
    print(f"   Долгота восх. узла Ω = {params['Omega']:.4f}°")  # Долгота узла
    print(f"   Аргумент перицентра ω = {params['w']:.4f}°")  # Аргумент перицентра

    print("\n5. ПОЛОЖЕНИЕ СПУТНИКА:")  # Раздел 5
    print(f"   Аргумент широты u = {params['u']:.4f}°")  # Аргумент широты
    print(f"   Истинная аномалия v = {params['v']:.4f}°")  # Истинная аномалия
    print(f"   Эксцентрическая аномалия E = {params['E']:.4f}°")  # Эксцентрическая аномалия
    print(f"   Средняя аномалия M = {params['M']:.4f}°")  # Средняя аномалия
    print(f"   Время от перицентра Δt = {params['delta_t']:.4f} с")  # Время от перицентра
    print(f"   Момент перицентра t_π = {params['t_pi']:.4f} с")  # Время перицентра

    print("\n6. КОНТРОЛЬ ВЫЧИСЛЕНИЙ:")  # Раздел 6
    print(f"   c·r = {params['control']['c_dot_r']:.2e} (должно быть ≈ 0)")  # Проверка ортогональности c и r
    print(f"   c·r' = {params['control']['c_dot_r_dot']:.2e} (должно быть ≈ 0)")  # Проверка ортогональности c и v
    print(f"   c·λ = {params['control']['c_dot_lambda']:.2e} (должно быть ≈ 0)")  # Проверка ортогональности c и λ
    print(f"   λ² = {params['lambda'] ** 2:.4f}")  # Квадрат модуля λ
    print(f"   c²·h + μ² = {params['control']['lambda_check']:.4f}")  # Проверка тождества
    print(f"   a = {params['a']:.4f}")  # Большая полуось
    print(f"   -μ/h = {params['control']['a_check']:.4f}")  # Альтернативный расчет большой полуоси

    # Построение графиков
    backend = select_render_backend()
    print(f"\nВыбран backend отрисовки: {backend}")

    if backend == "vispy":
        print("Запуск GPU-отрисовки (VisPy/OpenGL)...")
        plot_spatial_orbit_gpu(params)
    else:
        if not MATPLOTLIB_AVAILABLE:
            raise RuntimeError("Matplotlib недоступен. Установите matplotlib или используйте VisPy.")
        print("\nСтроится плоский чертеж орбиты с анимацией...")  # Сообщение
        fig_flat, ax_flat, anim_flat = plot_flat_orbit(params)  # Построение 2D графика

        print("Строится пространственный чертеж орбиты с анимацией...")  # Сообщение
        fig_spatial, ax_spatial, anim_spatial = plot_spatial_orbit_with_animation(params, x, y, z, x_dot, y_dot, z_dot)  # Построение 3D графика

        plt.tight_layout()  # Автоматическая регулировка расположения элементов
        plt.show()  # Отображение графиков
