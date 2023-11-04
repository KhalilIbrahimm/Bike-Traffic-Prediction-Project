import os
import numpy as np
import pandas as pd

# The PreProcessing class contains functions for preprocessing raw datasets before they're used in machine learning experiments.
class PreProcessing:
    def __init__(self):
        pass

    def find_outliers(self, df):
        """
        Find and print the number of outliers in a DataFrame.

        Args:
        df (pd.DataFrame): Data to be examined for outliers.

        Returns:
        None
        """
        df = pd.to_numeric(df)
        q1 = df.quantile(0.25)
        q3 = df.quantile(0.75)
        iqr = q3-q1
        outliers = df[((df < (q1-1.5*iqr)) | (df > (q3+1.5*iqr)))]
        print("number of outliers: " + str(len(outliers)))

    def TrafikkDataPreprocess(self):
        """
        Preprocess traffic data by performing various data cleaning operations.

        Returns:
        pd.DataFrame: Preprocessed traffic data.
        """
        df = pd.read_csv("data/trafikkdata.csv", sep=r';|\|', engine="python")
        pd.set_option('display.max_columns', None)  # To display all columns
        pd.set_option('display.max_rows', None)
        
        # Filter the "Felt" column for the "Totalt" value, as that's what we'll be predicting later and it's important to us.
        df = df[df["Felt"] == "Totalt"]
        
        # Combine "Dato" and "Fra tidspunkt" into the same column.
        df["Fra"] = df["Dato"] + " " + df["Fra tidspunkt"]
        df["Til"] = df["Dato"] + " " + df["Til tidspunkt"]
        
        # Drop unnecessary columns from the dataset. The rationale is explained in the report.
        df = df.drop(["Felt"], axis=1)
        df = df.drop(df.loc[:, "Til":"Til tidspunkt"], axis=1)
        df = df.drop(df.loc[:, "Dekningsgrad (%)":], axis=1)
        df = df.drop(df.loc[:, "Trafikkregistreringspunkt":"Vegreferanse"], axis=1)
        
        # Change data types for the "Fra" and "Til" columns to later merge the entire dataset with the Florida datasets.
        df["Fra"] = pd.to_datetime(df["Fra"], format='%Y-%m-%d %H:%M')
        
        # Drop "-" values in "Trafikkmengde"
        df[df["Trafikkmengde"] == "-"] = np.nan
        
        # Make the "Fra" column an index column
        df = df.set_index("Fra")

        return df

    def FloridaDataPreprocess(self):
        """
        Preprocess Florida weather data by performing various data cleaning operations.

        Returns:
        pd.DataFrame: Preprocessed Florida weather data.
        """
        liste = []

        # Find all files starting with "Florida" in the "data" folder using the os library.
        filer = [fil for fil in os.listdir("data") if fil.startswith("Florida")]
    
        for fil in filer:
            data = pd.read_csv(f"data/{fil}")
            data["Dato"] = pd.to_datetime(data["Dato"] + " " + data["Tid"], format="%Y-%m-%d %H:%M")
            data = data.drop(["Tid"], axis=1)
            liste.append(data)
            
        concat_columns = pd.concat(liste)
        concat_columns = concat_columns.replace(9999.99, np.nan)
        
        # Drop the "Relativ luftfuktighet" column, as it doesn't have a significant impact.
        concat_columns = concat_columns.drop(["Relativ luftfuktighet"], axis=1)
        concat_columns["Dato"] = pd.to_datetime(concat_columns["Dato"])
        concat_columns = concat_columns.set_index("Dato")
        concat_columns["Globalstraling"] = concat_columns["Globalstraling"].clip(lower=0, upper=1000)
        concat_columns_resampled = concat_columns.resample("1H").mean()
        concat_columns_resampled["Solskinstid"] = concat_columns["Solskinstid"].resample("H").sum()

        return concat_columns_resampled

    def Merge(self):
        """
        Merges traffic data and Florida weather data into a single dataset and performs feature engineering.

        Returns:
        pd.DataFrame: Merged dataset with performed changes.
        """
        trafikk_data_df = self.TrafikkDataPreprocess()
        florida_datasett = self.FloridaDataPreprocess()
        merged_datasett = trafikk_data_df.merge(florida_datasett, left_index=True, right_index=True, how="inner")
        
        # Remove duplicates in the "Fra" column due to the daylight saving time shift.
        merged_datasett = merged_datasett.reset_index()
        merged_datasett = merged_datasett.rename(columns={"index":"Fra"})
        merged_datasett = merged_datasett.drop_duplicates(["Fra"])

        # Feature Engineering
        merged_datasett = merged_datasett.set_index("Fra")
        merged_datasett["Dager"] = merged_datasett.index.dayofweek
        merged_datasett["Tid"] = merged_datasett.index.hour
        merged_datasett["Måned"] = merged_datasett.index.month
        merged_datasett["År"] = merged_datasett.index.year

        # Drop the "Fra" (or index) column as we no longer need to keep it.
        merged_datasett = merged_datasett.reset_index().drop(["Fra"], axis=1)
        
        # Convert merged_datasett types to float.
        merged_datasett = merged_datasett.astype(float)
        
        print("\nOVERVIEW OF NaN VALUES:\n", merged_datasett.isna().sum())
        
        return merged_datasett

