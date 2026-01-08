import os
import datetime

# Remplace par ton dossier
folder = "group_psd"

# Liste tous les fichiers du dossier
for filename in os.listdir(folder):
    filepath = os.path.join(folder, filename)
    if os.path.isfile(filepath):
        # Récupère la date de dernière modification
        mod_time = os.path.getmtime(filepath)
        # Convertit en format lisible
        readable_time = datetime.datetime.fromtimestamp(mod_time)
        print(f"{filename} - Dernière modification : {readable_time}")


# # Répertoire courant
# cwd = os.getcwd()
# print("Répertoire courant :", cwd)

# # Liste tous les fichiers .npy dans ce répertoire
# npy_files = [f for f in os.listdir(cwd) if f.endswith(".npy")]
# print("Fichiers .npy trouvés :", npy_files)
