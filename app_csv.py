import streamlit as st

import os
import json
import pandas as pd
import plotly.graph_objects as go
import re
from typing import List

import unidecode
from bing_image_urls import bing_image_urls
from streamlit_searchbox import st_searchbox
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity



# Define function to filter players by position
def filter_players_by_position(df: pd.DataFrame, positions: list[str]) -> pd.DataFrame:
    if positions:
        filtered_df = df[df['Pos'].isin(positions)]
        return filtered_df.head(20)  # Return the top 20 players matching the selected positions
    else:
        return pd.DataFrame()

# Streamlit app for filtering players by position
def filter_by_position_page():
    st.title('Filter Players by Position')

    # Sidebar for filtering positions
    st.sidebar.markdown('**Filter by Position:**')
    positions = st.sidebar.multiselect('Select Positions:', df_player['Pos'].unique().tolist())

    # Filter players based on selected positions
    filtered_players = filter_players_by_position(df_player, positions)

    # Display filtered players
    if not filtered_players.empty:
        st.subheader('Filtered Players')
        st.dataframe(filtered_players)
    else:
        st.write('No players found for the selected positions.')


#taking data from csv file
df_player = pd.read_csv('football-player-stats-2023.csv')


def similar_players_page():
    #Search Engine
    press = False
    choice = None

    #streamlit session state useful for page reloading
    if 'expanded' not in st.session_state:
        st.session_state.expanded = True

    if 'choice' not in st.session_state:
        st.session_state.choice = None


    df_player = pd.read_csv('football-player-stats-2023.csv')


    def remove_accents(text: str) -> str:
        return unidecode.unidecode(text)

    def search_csv(searchterm: str) -> List[str]:
        if searchterm:
            normalized_searchterm = remove_accents(searchterm.lower())
            df_player['NormalizedPlayer'] = df_player['Player'].apply(lambda x: remove_accents(x.lower()))
            filtered_df = df_player[df_player['NormalizedPlayer'].str.contains(normalized_searchterm, case=False, na=False)]
            suggestions = filtered_df['Player'].tolist()
            return suggestions
        else:
            return []

    selected_value = st_searchbox(
    search_csv,
    key="csv_searchbox",
    placeholder="🔍 Search a Football Player - CSV version"
    )

    st.session_state.choice = selected_value
    choice = st.session_state.choice
    if choice:
    
        # Extract column names 
        columns_to_process = list(df_player.columns)
        df_player_norm = df_player.copy()
        custom_mapping = {
            'GK': 1,
            'DF,FW': 4,
            'MF,FW': 8,
            'DF': 2,
            'DF,MF': 3,
            'MF,DF': 5,
            'MF': 6,
            'FW,DF': 7,
            'FW,MF': 9,
            'FW': 10
        }


        df_player_norm['Pos'] = df_player_norm['Pos'].map(custom_mapping)

        selected_features = ['Pos', 'Age', 'Int',
            'Clr', 'KP', 'PPA', 'CrsPA', 'PrgP', 'Playing Time MP',
            'Performance Gls', 'Performance Ast', 'Performance G+A',
            'Performance G-PK', 'Performance Fls', 'Performance Fld',
            'Performance Crs', 'Performance Recov', 'Expected xG', 'Expected npxG', 'Expected xAG',
            'Expected xA', 'Expected A-xAG', 'Expected G-xG', 'Expected np:G-xG',
            'Progression PrgC', 'Progression PrgP', 'Progression PrgR',
            'Tackles Tkl', 'Tackles TklW', 'Tackles Def 3rd', 'Tackles Mid 3rd',
            'Tackles Att 3rd', 'Challenges Att', 'Challenges Tkl%',
            'Challenges Lost', 'Blocks Blocks', 'Blocks Sh', 'Blocks Pass',
            'Standard Sh', 'Standard SoT', 'Standard SoT%', 'Standard Sh/90', 'Standard Dist', 'Standard FK',
            'Performance GA', 'Performance SoTA', 'Performance Saves',
            'Performance Save%', 'Performance CS', 'Performance CS%',
            'Penalty Kicks PKatt', 'Penalty Kicks Save%', 'SCA SCA',
            'GCA GCA', 
            'Aerial Duels Won', 'Aerial Duels Lost', 'Aerial Duels Won%',
            'Total Cmp', 'Total Att', 'Total Cmp', 'Total TotDist',
            'Total PrgDist', '1/3'
        ]



        #Cosine Similarity
        scaler = MinMaxScaler()
        df_player_norm[selected_features] = scaler.fit_transform(df_player_norm[selected_features])

        similarity = cosine_similarity(df_player_norm[selected_features])

        index_player = df_player.loc[df_player['Player'] == choice, 'Rk'].values[0]

        # Calculate similarity scores and sorting them in descending order
        similarity_score = list(enumerate(similarity[index_player]))
        similar_players = sorted(similarity_score, key=lambda x: x[1], reverse=True)

        # similar players here
        similar_players_data = []

        # Loop 
        for player in similar_players[1:11]:  
            index = player[0]
            player_records = df_player[df_player['Rk'] == index]
            if not player_records.empty:
                player_data = player_records.iloc[0]  # Get the first row (there should be only one)
                similar_players_data.append(player_data)

        similar_players_df = pd.DataFrame(similar_players_data)

        # Analytics 
        url_player = bing_image_urls(choice+ " "+df_player.loc[df_player['Player'] == choice, 'Squad'].iloc[0]+" 2023", limit=1, )[0]

        with st.expander("Features of The Player selected - The data considered for analysis pertains to the period of 2022 - 2023.", expanded=True):

            col1, col2, col3 = st.columns(3)

            with col1:
                st.subheader(choice)
                st.image(url_player, width=356)

            with col2:
                st.caption("📄 Information of Player")
                col_1, col_2, col_3 = st.columns(3)

                with col_1:
                    st.metric("Nation", df_player.loc[df_player['Player'] == choice, 'Nation'].iloc[0], None)
                    st.metric("Position", df_player.loc[df_player['Player'] == choice, 'Pos'].iloc[0], None)

                with col_2:
                    st.metric("Born", df_player.loc[df_player['Player'] == choice, 'Born'].iloc[0], None)
                    st.metric("Match Played", df_player.loc[df_player['Player'] == choice, 'Playing Time MP'].iloc[0], None, help="In 2022/2023")

                with col_3:
                    st.metric("Age", df_player.loc[df_player['Player'] == choice, 'Age'].iloc[0], None)

                st.metric(f"🏆 League: {df_player.loc[df_player['Player'] == choice, 'Comp'].iloc[0]}", df_player.loc[df_player['Player'] == choice, 'Squad'].iloc[0], None)

            with col3:
                st.caption("⚽ Information target of Player")
                
                if df_player.loc[df_player['Player'] == choice, 'Pos'].iloc[0] == "GK":
                        col_1, col_2 = st.columns(2)

                        with col_1:
                            st.metric("Saves", df_player.loc[df_player['Player'] == choice, 'Performance Saves'].iloc[0], None, help="Total number of saves made by the goalkeeper.")
                            st.metric("Clean Sheet", df_player.loc[df_player['Player'] == choice, 'Performance CS'].iloc[0], None, help="Total number of clean sheets (matches without conceding goals) by the goalkeeper.")

                        with col_2:
                            st.metric("Goals Against", df_player.loc[df_player['Player'] == choice, 'Performance GA'].iloc[0], None, help="Total number of goals conceded by the goalkeeper.")
                            st.metric("ShoTA", df_player.loc[df_player['Player'] == choice, 'Performance SoTA'].iloc[0], None, help="Total number of shots on target faced by the goalkeeper.")

                
                if df_player.loc[df_player['Player'] == choice, 'Pos'].iloc[0] == "DF" or df_player.loc[df_player['Player'] == choice, 'Pos'].iloc[0] == "DF,MF" or df_player.loc[df_player['Player'] == choice, 'Pos'].iloc[0] == "DF,FW":
                    col_1, col_2, col_3 = st.columns(3)

                    with col_1:
                        st.metric("Assist", df_player.loc[df_player['Player'] == choice, 'Performance Ast'].iloc[0], None, help="Total number of assists provided by the defender.")
                        st.metric("Goals", df_player.loc[df_player['Player'] == choice, 'Performance Gls'].iloc[0], None, help="Total number of goals scored by the defender.")

                    with col_2:
                        st.metric("Aerial Duel", df_player.loc[df_player['Player'] == choice, 'Aerial Duels Won'].iloc[0], None, help="Percentage of aerial duels won by the defender.")
                        st.metric("Tackle", df_player.loc[df_player['Player'] == choice, 'Tackles TklW'].iloc[0], None, help="Total number of successful tackles made by the defender in 2022/2023.")

                    with col_3:
                        st.metric("Interception", df_player.loc[df_player['Player'] == choice, 'Int'].iloc[0], None, help="Total number of interceptions made by the defender.")
                        st.metric("Key Passage", df_player.loc[df_player['Player'] == choice, 'KP'].iloc[0], None, help="Total number of key passes made by the defender.")

                
                if df_player.loc[df_player['Player'] == choice, 'Pos'].iloc[0] == "MF" or df_player.loc[df_player['Player'] == choice, 'Pos'].iloc[0] == "MF,DF" or df_player.loc[df_player['Player'] == choice, 'Pos'].iloc[0] == "MF,FW":
                    col_1, col_2, col_3 = st.columns(3)

                    with col_1:
                        st.metric("Assist", df_player.loc[df_player['Player'] == choice, 'Performance Ast'].iloc[0], None, help="Total number of assists provided by the player.")
                        st.metric("Goals", df_player.loc[df_player['Player'] == choice, 'Performance Gls'].iloc[0], None, help="Total number of goals scored by the player.")
                        st.metric("Aerial Duel", df_player.loc[df_player['Player'] == choice, 'Aerial Duels Won'].iloc[0], None, help="Percentage of aerial duels won by the player.")

                    with col_2:
                        st.metric("GCA", df_player.loc[df_player['Player'] == choice, 'GCA GCA'].iloc[0], None, help="Total number of goal-creating actions by the player.")
                        st.metric("Progressive PrgP", df_player.loc[df_player['Player'] == choice, 'Progression PrgP'].iloc[0], None, help="Total number of progressive passes by the player.")

                    with col_3:
                        st.metric("SCA", df_player.loc[df_player['Player'] == choice, 'SCA SCA'].iloc[0], None, help="Total number of shot-creating actions by the player.")
                        st.metric("Key Passage", df_player.loc[df_player['Player'] == choice, 'KP'].iloc[0], None, help="Total number of key passes by the player.")

                
                if df_player.loc[df_player['Player'] == choice, 'Pos'].iloc[0] == "FW" or df_player.loc[df_player['Player'] == choice, 'Pos'].iloc[0] == "FW,MF" or df_player.loc[df_player['Player'] == choice, 'Pos'].iloc[0] == "FW,DF":
                    col_1, col_2, col_3 = st.columns(3) 

                    with col_1:
                        st.metric("Assist", df_player.loc[df_player['Player'] == choice, 'Performance Ast'].iloc[0], None, help="Total number of assists provided by the player.")
                        st.metric("Goals", df_player.loc[df_player['Player'] == choice, 'Performance Gls'].iloc[0], None, help="Total number of goals scored by the player.")
                        st.metric("Aerial Duel", df_player.loc[df_player['Player'] == choice, 'Aerial Duels Won'].iloc[0], None, help="Percentage of aerial duels won by the player.")

                    with col_2:
                        st.metric("SCA", df_player.loc[df_player['Player'] == choice, 'SCA SCA'].iloc[0], None, help="Total number of shot-creating actions by the player.")
                        st.metric("xG", df_player.loc[df_player['Player'] == choice, 'Expected xG'].iloc[0], None, help="Expected goals (xG) by the player.")
                        st.metric("xAG", df_player.loc[df_player['Player'] == choice, 'Expected xAG'].iloc[0], None, help="Expected assists (xAG) by the player.")

                    with col_3:
                        st.metric("GCA", df_player.loc[df_player['Player'] == choice, 'GCA GCA'].iloc[0], None, help="Total number of goal-creating actions by the player.")
                        st.metric("Key Passage", df_player.loc[df_player['Player'] == choice, 'KP'].iloc[0], None, help="Total number of key passes by the player.")

                                
                        
        # Radar and Rank
        col1, col2 = st.columns([1.2, 2])

        with col1:
            #Similar Players Component 
            st.subheader(f'Similar Players to {choice}')
            st.caption("This ranking list is determined through the application of a model based on **Cosine Similarity**. It should be noted that, being a ranking, the result obtained is inherently subjective.")
            selected_columns = ["Player", "Nation", "Squad", "Pos", "Age"]
            st.dataframe(similar_players_df[selected_columns], hide_index=True, use_container_width=True)

        with col2:
            #Radar Analytics 
            categories = ['Performance Gls', 'Performance Ast', 'KP', 'GCA GCA','Aerial Duels Won', 'Int', 'Tackles TklW', 'Performance Saves', 'Performance CS', 'Performance GA','Performance SoTA']
            selected_players = similar_players_df.head(10)

            fig = go.Figure()

            for index, player_row in selected_players.iterrows():
                player_name = player_row['Player']
                values = [player_row[col] for col in categories]
                
                fig.add_trace(go.Scatterpolar(
                    r=values,
                    theta=categories,
                    fill='toself',
                    name=player_name
                ))

            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                    )
                ),
                showlegend=True,  
                legend=dict(
                    orientation="v", 
                    yanchor="top",  
                    y=1,  
                    xanchor="left",  
                    x=1.02,  
                ),
                width=750,  
                height=520  
            )

            st.plotly_chart(fig, use_container_width=True)
 
    
def main():
    # Set page configuration
    st.set_page_config(page_title="Player Recommendation System", page_icon="⚽", layout="wide")

    # Display title and description
    st.markdown("<h1 style='text-align: center;'>⚽🔍 Player Recommendation System</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size: 18px;'>Welcome to the Player Recommendation System.</p>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size: 18px;'>--------------Choose your option--------------</p>", unsafe_allow_html=True)

    # Add buttons for navigation centered horizontally
    col1, col2 = st.columns(2)  # Create two columns

    with col1:
        # Center-align the button in the column
        st.write("")  # Placeholder for layout
        st.write("")  # Placeholder for layout
        st.write("")  # Placeholder for layout
        st.markdown(
            "<style>div[data-testid='stHorizontalBlock'] div:first-child { display: flex; justify-content: center; }</style>",
            unsafe_allow_html=True,
        )
        if st.button("Filter by Positions", key="filter_button"):
            st.session_state.page = "filter_positions" 

    with col2:
        # Center-align the button in the column
        st.write("")  # Placeholder for layout
        st.write("")  # Placeholder for layout
        st.write("")  # Placeholder for layout
        st.markdown(
            "<style>div[data-testid='stHorizontalBlock'] div:first-child { display: flex; justify-content: center; }</style>",
            unsafe_allow_html=True,
        )
        if st.button("Similar Players", key="similar_button"):
            st.session_state.page = "similar_players" 

    # Render appropriate page based on session state
    if "page" in st.session_state:
        if st.session_state.page == "filter_positions":
            filter_by_position_page()  # Redirect to filter by positions page
        elif st.session_state.page == "similar_players":
            similar_players_page()  # Redirect to similar players page


    

if __name__ == "__main__":
    main()
