# About this project

This project "Bike Traffic Prediction", is focused on forecasting the number of cyclists who will cross the Nygårdsbroen in 2023. The core data for these predictions is sourced from two Norwegian institutions: Statens vegvesen (The Norwegian Public Roads Administration) and the Geofysisk institutt (Geophysical Institute).

### Data Sources:

Traffic Data from Statens vegvesen: This dataset comprises several columns including "Dato" (Date), "Fra tidspunkt" (From Time), "Felt" (Lane), and "Trafikkmengde" (Traffic Volume). The main objective is to forecast the "Trafikkmengde" values for entries where the "Felt" is marked as "Totalt" (Total). The data for 2023 is notably absent and hence requires prediction.

Weather Data from Geofysisk institutt: Weather data is partitioned into separate files for each year. It encompasses columns like "Dato" (Date), "Tid" (Time), and other weather-related columns. The predictions can incorporate any relevant weather data, given that it has been recorded no later than the specified "Fra tidspunkt" (From Time).

### Project Components:

Data Preparation & Exploratory Data Analysis (EDA): Initial stages involve cleaning, preprocessing, and understanding the data's underlying patterns.
Modeling & Prediction: Using the cleaned data to build and train predictive models.

2023 Bike Traffic Prediction: The ultimate goal is to estimate the bike traffic volume for 2023 based on the historical data and potential influencing factors like weather.
Web Interface Creation: To enhance user interaction, there's a plan to design a website that will likely present the findings and perhaps offer real-time predictions or insights.


## Setup and Requirements

1. This project uses **Python 3.11**.

2. Create a virtual environment by running the following command in the terminal:

    ```bash
    python3 -m venv venv
    ```

    Then, activate the virtual environment:

    ```bash
    source venv/bin/activate
    ```

3. Install the necessary packages by running the following commands:

    ```bash
    pip install numpy
    pip isntall pandas 
    pip install Plotly 
    pip install scikit-learn 
    pip install flask 
    pip install waitress 
    pip install requests
    ```
    or just this:
   ```bash
   pip install numpy pandas Plotly scikit-learn flask waitress requests
   ```

5. Ensure you have the dataset in the project folder, under the "data" directory. The dataset should be the same as the one available at UIB with the same filename.

## File Structure and Explanation

The project directory contains the following files and their purpose:

- **ModelEmplimentation.py**: This file includes the implementation of the models, encompassing the process of running cross-validation on training data, selecting the best model, and evaluating the best model on test data.

- **DataExploration.ipynb**: This is a Jupyter Notebook file that provides exploration and plotting of training data.

- **DataPreprocessing.py**: This file manages the data preparation, feature engineering and data preprocessing process.

- **DataVisualize.py**: This file contains code for data visualization.

- **app.py**: This is the main program for the web application.

- **predictions.csv**: This is a CSV file containing the 2023 data prediction.

- **model.pkl**: This is a Pickle file that holds the finally selected model in a pipline with Standatdscaller and KNNimputer.

- **templates**: This directory contains the HTML code for the web application.

- **log.txt**: All code output in the terminal will be saved in this file.

- **telegram_bot.py**: Training the models and running cross-validation can take som time. The idea here is to be able to run the code and, if desired, receive updated info, such as cross-validation output, as a message on Telegram to keep track while you're out and doing your life thing.

- **Main.py**: This is the main program to be executed to generate the final results. Here, all classes and codes are combined. To generate the final results, install all necessary packages and run the file by typing "python Main.py" in the terminal. Note that it might take some time to run this file, as the models are trained, cross-validation, and evaluations are performed.

To launch the website and test the the final chosed model in live test and to predict new data, use the following command: 
```bash
python app.py
```
Then search and go to:
```bash
localhost:8080
```
