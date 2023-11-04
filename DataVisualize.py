
import plotly.express as px
import pandas as pd


# DataPloter-klassen inneholder funksjoner for å lage datagrafikk og visualisere datasettet.
class DataPloter:
    def __init__(self):
        pass

    def plot_trafic_by_week_day(self, df, label):
        """
        Plott trafikkmengde fordelt etter ukedag for hvert år.

        Args:
        df (pd.DataFrame): Datasettet med trafikkdata.
        label (str): Navnet på kolonnen som inneholder trafikkmengden.

        Returns:
        None
        """

        df = pd.concat([df, label], axis=1)
        
        ## Trafikkmengde over dager hvert år for seg
        år = [2015,2016,2017,2018,2019,2020]
        dag_navn = ["Mandag", "Tirsdag", "Onsdag", "Torsdag", "Fredag", "Lørdag", "Søndag"]
        for årstall in år:
            merged_data_under2023= df[df["År"]<=årstall]
            dag_navn_navnlig = merged_data_under2023["Dager"].astype(int).apply(lambda day_num: dag_navn[day_num])  # Mapper nummer til dagens navn
            fig = px.histogram(merged_data_under2023, x=dag_navn_navnlig, y="Trafikkmengde", title = f"{årstall} trafikkmengde!", color = dag_navn_navnlig, histfunc = "avg").update_layout(xaxis_title = "Dag", yaxis_title = "Trafikkmengde") 
            #histfunc = "avg", den finner gjennomsnitten av trafikkmengde isteden for å bruke .mean
            fig.show()


    def plot_trafic_by_year(self, df, label):
        """
        Plott trafikkmengde fordelt etter år.

        Args:
        df (pd.DataFrame): Datasettet med trafikkdata.
        label (str): Navnet på kolonnen som inneholder trafikkmengden.

        Returns:
        None
        """

        fig = px.histogram(df, x="År", y=label, title = "I hvilket år er det mest antall syklister? ", color = "År").update_layout(xaxis_title = "År", yaxis_title = "Trafikkmengde")
        fig.show()
        
    def plot_trafic_by_month(self, df, label):
        """
        Plott trafikkmengde fordelt etter måned.

        Args:
        df (pd.DataFrame): Datasettet med trafikkdata.
        label (str): Navnet på kolonnen som inneholder trafikkmengden.

        Returns:
        None
        """

        måned_navn_list = [
        "Januar", "Februar", "Mars", "April", "Mai", "Juni",
        "Juli", "August", "September", "Oktober", "November", "Desember"
    ]
        df = pd.concat([df, label], axis=1)
        
        måned_num_series = df["Måned"].astype(int)
        måned_navnlig = måned_num_series.apply(lambda måned_num: måned_navn_list[måned_num - 1])
        
        fig = px.histogram(df, x=måned_navnlig, y="Trafikkmengde", title="I hvilke måned er det mest antall syklister? \nTrafikkmengde over måneder!", color = måned_navnlig, histfunc="avg").update_layout(xaxis_title = "Måned", yaxis_title = "Trafikkmengde")
        
        fig.show()

    def plot_trafic_by_time(self, df, label):
        """
        Plott trafikkmengde fordelt etter tidspunkt på døgnet.

        Args:
        df (pd.DataFrame): Datasettet med trafikkdata.
        label (str): Navnet på kolonnen som inneholder trafikkmengden.

        Returns:
        None
        """

        ## Trafikkmengde over tidspunkt
        fig = px.histogram(df, x ="Tid", y =label, histfunc="avg").update_layout(xaxis_title = "Tidspunkt", yaxis_title="Trafikkmengde",  title = "I hvilke tidspunkt er det mest antall syklister?\nTrafikkmengde over tidspunkt. ")
        fig.show()

        
