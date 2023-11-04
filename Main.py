# Import necessary libraries and modules.
import numpy as np
import pandas as pd
from telegram_bot import Bot
from DataPreprocessing import PreProcessing
from ModelsEmplimentation import ModelImplementation
from sklearn.model_selection import train_test_split


# Data Preprocessing steps
print("\n\nStep 1: PreProcessing begins.")

# Starting with the data preprocessing step.
merged_data = PreProcessing().Merge()
print("Done!")

# Filter out data that's before 2023 and remove rows with missing values.
merged_dataset_under2023 = merged_data[merged_data["År"]< 2023].dropna()
merged_dataset_under2023.sort_index(inplace = True)

# Splitting the dataset into training and testing data.
X = merged_dataset_under2023.drop("Trafikkmengde", axis = 1)
y = merged_dataset_under2023["Trafikkmengde"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, shuffle = False)
print(f"\n\nX_train shape: {X_train.shape}.\ny_train shape: {y_train.shape}")


print("\n\nSteps 2 & 3: Model implementation and evaluation.")

# Move on to steps 2 and 3: implementation and evaluation of the models.
model = ModelImplementation()
message1 = model.evaluate_all(X_train, y_train, save_model=True)
model.evaluate_final_best_model(X_test, y_test)  # Evaluate the best model on the test set.
message2 = model.predict_2023_values(merged_data)  # Predict traffic volume for 2023 and save it in its own csv file (predictions.csv) using the best model saved in a pickle file (model.pkl).

# If you want to activate the telegram messages function, set it to True. NOTE: Make sure that you have entered your telegram token/chat_id info in telegram_bot.py.  
do_you_want_telegram_messages = False

# Update me on Telegram!
if do_you_want_telegram_messages:
    Bot((message1, message2)).send_message()
