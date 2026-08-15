import matplotlib.pyplot as plt
import matplotlib.dates as mdates # Wichtig für die Formatierung der X-Achse
import numpy as np
from datetime import datetime
from cleaning_utils import extract_important_features


from method import mahalanobis_distances 

def plot_distances(distances, time_strings):
    # 1. Konvertiere die Distanzen in ein NumPy-Array für die Mathematik
    distances = np.array(distances)

    # 2. Konvertiere die Zeit-Strings in datetime-Objekte
    # Wir schneiden mit [:19] die Zeitzone (+02) ab, um das Parsen zu vereinfachen
    # Aus "2026-08-07 09:37:00+02" wird "2026-08-07 09:37:00"
    time_objs = [datetime.strptime(t[:19], "%Y-%m-%d %H:%M:%S") for t in time_strings]
    time = np.array(time_objs) # Auch als NumPy-Array für das Boolean-Indexing

    # Schwellenwert berechnen
    mean = np.mean(distances)
    threshold = mean + 3 * np.std(distances)
    outliers = distances > threshold

    plt.figure(figsize=(12, 6))
    
    # Plotten
    plt.plot(time, distances, color='steelblue', linewidth=1.5, label='Mahalanobis distance')
    plt.scatter(time[outliers], distances[outliers], facecolors='none', edgecolors='red', s=100, label='Outlier')
    plt.axhline(threshold, color='darkred', linestyle='--', linewidth=1.5, label='Threshold')

    # Werte über die Outliers schreiben
    for i, d in enumerate(distances):
        if outliers[i]:
            plt.text(time[i], d + (max(distances)*0.02), f'{d:.2f}', fontsize=8, ha='center')

    # 3. X-Achse hübsch formatieren (z. B. nur Stunden und Minuten anzeigen)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    plt.gcf().autofmt_xdate() # Rotiert die Labels leicht, damit sie nicht überlappen

    plt.title('Mahalanobis Distance Over Time')
    plt.xlabel('Time (HH:MM)')
    plt.ylabel('Mahalanobis Distance')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    features, time_stamps = extract_important_features('./Test_Data/data-1786192670480.csv')
    
    # Falls mahalanobis_distances in method.py noch Tupel zurückgibt, 
    # musst du hier nur die Distanzen extrahieren!
    # Angenommen, es gibt nur die reine Distanz-Liste zurück:
    distances = mahalanobis_distances(features) 
    
    plot_distances(distances, time_stamps)