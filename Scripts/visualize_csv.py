import pandas as pd
import matplotlib.pyplot as plt
import os

# === Wczytanie danych ===
basic_path = r"C:\Users\Marcin\Desktop\Studia\IoT\Data\Telefon_na_rece"
for directory in os.listdir(basic_path):
    file_name = os.path.join(basic_path, directory, "acce.txt")
    with open(file_name, 'r') as file:
        lines = file.readlines()

    # Filtrowanie linii zawierających dane (pomija nagłówki, opisy)
    data = [line.strip().split() for line in lines if line.strip() and line[0].isdigit()]

    # Konwersja do DataFrame
    df = pd.DataFrame(data, columns=['timestamp', 'x', 'y', 'z'])
    df = df.astype({'timestamp': 'int64', 'x': 'float', 'y': 'float', 'z': 'float'})

    # === Reset indeksu i ustawienie jako numer odczytu ===
    df.reset_index(drop=True, inplace=True)
    df.index = range(1, len(df)+1)

    # === Rysowanie wykresów ===
    fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    axs[0].plot(df.index, df['x'], marker='o', markersize=3, linestyle='-', color='blue')
    axs[0].set_ylabel('X')
    axs[0].set_title(directory)

    axs[1].plot(df.index, df['y'], marker='o', markersize=3, linestyle='-', color='green')
    axs[1].set_ylabel('Y')
    axs[1].set_title("na rece")

    axs[2].plot(df.index, df['z'], marker='o', markersize=3, linestyle='-', color='red')
    axs[2].set_ylabel('Z')
    axs[2].set_title(len(data))
    axs[2].set_xlabel("x")

    plt.tight_layout()
    plt.show()
