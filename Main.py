
# Importerer nødvendige biblioteker og moduler.
import numpy as np
import pandas as pd
from telegram_bot import Bot
from DataPreprocessing import PreProcessing
from ModelsEmplimentation import ModelEmplimentation
from sklearn.model_selection import train_test_split


# Data Preprocessing prosesser
print("\n\nSteg 1: PreProcessing starter.")
# Starter med trinnet for data forbehandling.
merged_data = PreProcessing().Merge()
print("Done!")

# Filtrer ut data som er før 2023 og fjerner rader med manglende verdier.
merged_datasett_under2023 = merged_data[merged_data["År"]< 2023].dropna()
merged_datasett_under2023.sort_index(inplace = True)

# Deler datasettet inn i trenings- og testsdata. 
X = merged_datasett_under2023.drop("Trafikkmengde", axis = 1)
y = merged_datasett_under2023["Trafikkmengde"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.99, shuffle = False)
print(f"\n\nX_train shape: {X_train.shape}.\ny_train shape: {y_train.shape}")


print("\n\nSteg 2 & 3: Modelimplimentasjon og evaluering.")
# Går videre til trinn 2 og 3: implementasjon og evaluering av modellene.
model = ModelEmplimentation()
message1 = model.evaluate_all(X_train, y_train, save_model=True)
model.evaluate_final_best_model(X_test, y_test) # Evaluerer den beste modellen på testsettet.
message2 = model.predict_2023_values(merged_data) # Predikerer trafikkmengden for 2023 og lagrer det i eget csv.fil (predictions.csv) ved hjelp av den beste modellen som er lagret i en pickle fil (model.pkl).

# If you want to activate telegram messages function, make it to True. OPS! Make sure that you put your telegram token/chat_id info in telegram_bot.py.  
do_you_want_telegram_messages = False

# Status oppdatere meg på telegram!
if do_you_want_telegram_messages:
    Bot((message1, message2)).send_message()