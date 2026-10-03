import sys
import json

if len(sys.argv) < 2:
  print("Error: Por favor, introduce el nombre del archivo.")
  print("Uso: python3 retrieve_paths.py nombre_del_archivo.txt")
  sys.exit(1)

nombre_archivo = sys.argv[1]

try:
  with open(nombre_archivo, "r", encoding="utf-8") as archivo:
    data = json.load(archivo)
    print(len(data["rag_questions"]))
    
    for pregunta in data["rag_questions"]:
        for source in pregunta["sources"]:
            longitud = source["last_character_index"] - source["first_character_index"]
            print(f"File Path: {source['file_path']}, Length: {longitud}")

except FileNotFoundError:
  print(f"Error: El archivo '{nombre_archivo}' no existe.")
