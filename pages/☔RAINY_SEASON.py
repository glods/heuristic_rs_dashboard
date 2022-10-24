import streamlit as st
import pandas as pd
import functions
import harvest_rainy

import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
import seaborn as sns
import geopandas as gpd
import fiona
import os
import gompertz_eval__rain_with_s1

st.set_page_config(layout="wide", page_icon="☔",)
if st.checkbox('HARVEST ESTIMATES '):
    st.markdown("# GRID 2917  VISUALISATION : HARVEST ")
    # st.write("Harvest ")
    harvest_rainy.harvest()
elif st.checkbox('FLOODING ESTIMATES 💦'):


    fiona.drvsupport.supported_drivers['geojson'] = 'rw' # enable KML support which is disabled by default
    fiona.drvsupport.supported_drivers['GeoJSON'] = 'rw'
    ##============================= INPUT DATA=============================
    #for grid level
    datagridlevel = 'data/rain/dag_grid2917.csv'
        # '/home/glorie/Documents/DASHBOARD/streamlit/GRID2917/data/dag_grid2917.csv'

    # gpd.io.file.fiona.drvsupport.supported_drivers['KML'] = 'rw'
    # gpd.io.file.fiona.drvsupport.supported_drivers ['GeoJSON'] = 'rw'

    #How to increase the width of web page
    # st.set_page_config(layout="wide")
    # st.set_page_config(layout="wide", page_icon="☔")

    st.title('GRID 2917  VISUALISATION : FLOODING 💦')

    # DATE_COLUMN = 'date'
    DATE_COLUMN = 'flooding_date'
    i='2019'
    DATA_URL =  'data/DATA_TO_SHARE/GRID2917-20220922T122839Z-001/GRID2917/Dagana/DRY_HOT_SEASON/'+i+'/flooding/flooding_Dagana'+i+'.csv'
    #'/home/glorie/Documents/DASHBOARD/streamlit/GRID2917/data/dag_grid2917.csv'
    @st.cache
    def load_data(nrows):
        data = pd.read_csv(DATA_URL, nrows=nrows)
        # data = pd.read_csv(DATA_URL)
        data = data.drop(columns='Unnamed: 0')
        lowercase = lambda x: str(x).lower()
        # data = data[data['flooding_date'] != '0']
        data.rename(lowercase, axis='columns', inplace=True)
        # data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN])
        return data #.set_index("year")
    data_load_state = st.text('Loading data...')
    data = load_data(10000)
    data_load_state.text("Done!")
    data = data.rename(columns={'planted_area':'flooded_area'})

    if st.checkbox('Show spatial points'):
        st.map(data)

    if st.checkbox('Show raw data'):
         st.subheader('Sample data in '+ str(i))
         st.write(data)
    #========================================================
    data = data[data['flooding_date'] != '0']
    data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN])



    # Some number in the range 0-23
    # hour_to_filter = st.slider('Month', 0, 5, 2)
    # hour_to_filter = st.radio(label="Month :", options=[1, 2, 3, 4, 5], index=2)
    # filtered_data = data[data[DATE_COLUMN].dt.month == hour_to_filter]

    # st.subheader('Flooding areas during the month %s' % hour_to_filter)
    # st.map(filtered_data)
    #================================================




    datav = pd.read_csv(DATA_URL)

    season ='RAINY_SEASON'
    y = ['2019' ,'2020', '2021', '2022']

    datahist  = gpd.GeoDataFrame()
    datahist1  = gpd.GeoDataFrame()
    for i in y :
        datav = pd.read_csv('data/DATA_TO_SHARE/GRID2917-20220922T122839Z-001/GRID2917/Dagana/'+season+'/'+i+'/flooding/flooding_Dagana'+i+'.csv' )
        datav =datav[['ID','flooding_date', 'aoi_area', 'total_flooding_area', 'latitude', 'longitude', 'geometry' ]]
        datav = datav[datav['flooding_date']!='0' ]
        datatmp =gpd.GeoDataFrame()
        # datatmp[i] = pd.to_datetime( datav.flooding_date ).dt.dayofyear

        # cold = [c for c in datav.columns if i in c]
        # doy_col = list(datav[cold] .sum(axis=0).index )
        #
        # datatmp [ doy_col] = [list(datav[cold] .sum(axis=0).values)]

        datatmp['ID'] = datav ['ID']
        datatmp ['year'] =  i
        datatmp [ 'DOY'] = pd.to_datetime( datav.flooding_date ).dt.dayofyear
        datatmp['total_flooding_area'] = datav['total_flooding_area']
        datahist1 = datahist1.append(datatmp)
        #
        datahist['ID'] = datav['ID']
        datahist[i] = pd.to_datetime(datav.flooding_date).dt.dayofyear
        datageo = datahist.copy()

        datageo['latitude'] =pd.to_numeric( datav['latitude'] )
        datageo['longitude'] =pd.to_numeric( datav['longitude'] )
        datageo['geometry'] = datav['geometry']
        # datageo = gpd.GeoDataFrame(datageo,geometry=datav['geometry'] , crs="EPSG:4326")
         # datatmp[i]    # datahist.append( datatmp )


    # if st.checkbox('Show raw data - DATA Geo'):
    #      st.subheader('Raw data')
    #      st.write(datageo)

    # if st.checkbox('Show raw data - DATA hist'):
    #      st.subheader('Raw data')
    #      st.write(datahist)
    # if st.checkbox('Show raw data - DATA hist 1'):
    #      st.subheader('Raw data')
    #      st.write(datahist1)
    # st.map(datageo)
    # datahist = datav.copy()
    # datahist = datahist[['flooding_date', 'aoi_area', 'year']]
    # datahist = datahist[datahist['flooding_date'] != '0']
    # datahist ['DOY'] = pd.to_datetime(datav.flooding_date ).dt.dayofyear


    ################# Histogram Logic ########################
    # if st.checkbox('Show raw data-hist'):
    #     st.subheader('Raw data')
    #     st.write(datahist)
    #--------------------------------------------------------

    # df =datahist1
    # # st.write(datahist1)
    # bins= st.sidebar.slider('Bins2', 10, 200, 100)
    # fig = px.histogram (df,  x="DOY", color="year", nbins=bins,
    #                      # or violin, rug
    #                     hover_data=df.columns, )
    # #
    # st.plotly_chart(fig, use_container_width=False)
    #
    # fig = px.box (df, y="DOY", color="year",
    #                     # or violin, rug
    #                    hover_data=df.columns,notched=True)
    #
    # st.plotly_chart(fig, use_container_width=False)
    #--------------------------------------------------------

    #####
    # create two columns for charts

    #####


    # sns.histplot(data=datahist1, x="DOY",  hue="year")
    # plt.show()
    #--------------------------------------------------------

    # fig = sns.displot(df, x='DOY', hue="year",  kde=True, alpha=0.5)
    # st.pyplot(fig)
    # #--------------------------------------------------------


    # st.plotly_chart(fig, use_container_width=True)

    # st.write(df)


    # Create distplot with custom bin_size

    # fig = ff.create_distplot(
    #           hist_data, group_labels,bin_size= bins)
    # st.plotly_chart(fig, use_container_width=False)




    #----------------------------
    # fig = go.Figure()
    # for val in hist_data:
    # # fig.add_trace(go.Histogram(x=hist_data[0]))
    #     fig.add_trace(go.Histogram(x=val ))
    # # Overlay both histograms
    # fig.update_layout(barmode='overlay')
    # # Reduce opacity to see both histograms
    # fig.update_traces(opacity=0.35)
    # st.plotly_chart(fig)
    # -------------------------------------


    # st.sidebar.markdown("### Histogram: Explore Distribution of planting timing over years : ")
    # st.sidebar.markdown('Choose which visualizations you want to see 👇')

    all_vizuals = [
                    'Summary statistics',
                   'Distribution of Numerical Columns', 'Box Plots', 'geospatial distribution', 'GRID-LEVEL ANALYSIS',
        'COMPARISON' ,  'GOMPERTZ-EVALUATION'
                   ]
    # functions.sidebar_space(3)
    vizuals = st.sidebar.multiselect("Choose which visualizations you want to see 👇", all_vizuals)

    num_columns = datahist.select_dtypes(exclude='object').columns
    cat_columns = datahist.select_dtypes(include='object').columns
    # bins = st.sidebar.radio(label="Bins :", options=[10,20,30,40,50,70], index=4)



    if 'Summary statistics' in vizuals:
        st.write('______________________________________________________________________________________________')
        st.write('______________________________________________________________________________________________')

        #---------------------

        df = datahist
        # st.write(datahist1)
        # Group data together
        col =  [ c  for c in df.columns if 'ID' not in c]
        # coltmp = [int(c) for c in col ]
        # coltmp.sort()
        # col = [str(c) for c in coltmp ]
        df =datahist[col]
        hist_data =  [ df[c].dropna() for c in col if 'ID' not in c]
        group_labels = df.columns


        fig_col1, fig_col2 = st.columns(2)
        with fig_col1:

            #--------------------------
            datau = pd.read_csv('data/comparison/data_prediction.csv')
            period = 'RAINY'
            datau = datau[datau['period'] == period]

            date_ = 'date'
            x = 'DOY'
            y = 'flooded_area'
            col = 'year'
            title = ''
            datau[x] = pd.to_datetime(datau[date_]).dt.dayofyear
            datau[col] = datau[col].astype(str)
            datau[y] = pd.to_numeric(datau[y])
            datau = datau.dropna()
            datau = datau[datau['data_source'] == 'RS']

            chartu = functions.get_chart_ts(datau, x, y, col, 'Evolution of flooded areas over years')

            st.altair_chart(
                chartu,
                use_container_width=True
            )

            #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            fig = sns.displot(datahist1, x='DOY', hue="year", kind='kde', alpha=0.5)
            st.pyplot(fig)





        with fig_col2:
        # --------------------------------------------------
        # --------------------------------------------------
            fig2 = px.box(datahist1, y="DOY", color="year",
                          # or violin, rug
                          hover_data=datahist1.columns, notched=True)
            st.plotly_chart(fig2, use_container_width=True)

        # -------------------------------------------------------
            bins = st.slider('slide-bins', 5, 30, 10)
            fig = ff.create_distplot(
                hist_data, group_labels, bin_size=bins)
            st.write(fig)

        #--------------------------------------------------
        # --------------------------------------------------
        #     fig2 = px.box(datahist1, y="total_flooding_area", color="year",
        #                   # or violin, rug
        #                   hover_data=datahist1.columns, notched=True)
        #     st.plotly_chart(fig2, use_container_width=True)    #     fig2 = px.box(datahist1, y="total_flooding_area", color="year",
        #                   # or violin, rug
        #                   hover_data=datahist1.columns, notched=True)
        #     st.plotly_chart(fig2, use_container_width=True)




    if 'Distribution of Numerical Columns' in vizuals:
        st.write('______________________________________________________________________________________________')
        st.write('______________________________________________________________________________________________')

        bins1 = st.sidebar.slider('Bins1', 5, 100, 10)
        if len(num_columns) == 0:
            st.write('There is no numerical columns in the data.')
        else:
            selected_num_cols = functions.sidebar_multiselect_container('Choose columns for Distribution plots:',
                                                                        num_columns, 'Distribution')
            # st.subheader('Distribution of numerical columns')
            st.subheader("### Histogram: flooding timing over years : ")
            i = 0

            while (i < len(selected_num_cols)):
                c1, c2 = st.columns( 2 )
                for j in [ c1, c2 ]:

                    if ( i >= len ( selected_num_cols ) ):
                        break

                    fig = px.histogram(datahist, x=selected_num_cols[i], nbins=bins1, labels={
                                            selected_num_cols[i]: "DOY",  },
                    title = selected_num_cols[i]
                  )

                    j.plotly_chart (fig, use_container_width=True)
                    i += 1



    if 'Box Plots' in vizuals:
        st.write('______________________________________________________________________________________________')
        st.write('______________________________________________________________________________________________')

        if len( num_columns ) == 0:
                st.write('There is no numerical columns in the data.')
        else:
                selected_num_cols = functions.sidebar_multiselect_container('Choose columns for Box plots:', num_columns,
                                                                            'Box')
                st.subheader('Box plots :  Flooding timing over years')
                i = 0
                while (i < len(selected_num_cols)):
                    c1, c2 = st.columns(2)
                    for j in [c1, c2]:

                        if (i >= len(selected_num_cols)):
                            break
                        fig = px.box(datahist, y=selected_num_cols[i], labels={
                         selected_num_cols[i]: "DOY",

                     },
                    title=selected_num_cols[i])

                        j.plotly_chart(fig, use_container_width=False, )
                        i += 1

    #========================================================================================================================

    if 'geospatial distribution' in vizuals:
            st.write('______________________________________________________________________________________________')
            st.write('______________________________________________________________________________________________')

        #https://plotly.com/python/mapbox-layers/
            # LOAD GEIJASON FILE
            yar='2019'
            datag ='data/DATA_TO_SHARE/GRID2917-20220922T122839Z-001/GRID2917/Dagana/DRY_HOT_SEASON/'+yar+'/flooding/flooding_Dagana'+yar+'.csv'
            # st.write('HERE')
            # st.write()

            # with open(datag) as response:
            #     geo = json.load(response)
            #     json.loads(j.read())
            import sys, json

            # struct = {}
            # try:
            geo = pd.read_csv(datag)
            # geo = gpd.GeoDataFrame(geo, crs="EPSG:4326")
            gs = gpd.GeoSeries.from_wkt(geo['geometry'])
            geo = gpd.GeoDataFrame(geo, geometry=gs, crs="EPSG:4326")
            # except:
            #     st.write('error')
            #     print( sys.exc_info() )
            # with open(datag, 'r') as j:
            #     geo = json.loads(j.read())
            ######################################################################################################
            # st.write('adat')
            # st.write(geo)
            # Add title and header


            # Geographic Map
            fig = go.Figure(
                go.Choroplethmapbox(
                    geojson=geo,
                    # locations=df_gb_canton.kan_name,
                    # featureidkey="properties.kan_name",
                    z=geo['flooding_date'],
                    colorscale="sunsetdark",
                    # zmin=0,
                    # zmax=500000,
                    # marker_opacity=0.5,
                    # marker_line_width=0,
                )
            )
            fig.update_layout(
                mapbox_style="carto-positron",
                mapbox_zoom=6.6,
                mapbox_center={"lat": 46.8, "lon": 8.2},
                width=800,
                height=600,
            )
            fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
            # st.plotly_chart(fig)
            # fig.write_image("fig1.png")
            ######################################################################################################
        # def grid_map(gdf, title='', legend_name='', on_column='doy'):
            # gdf_path = ''
            # geo = pd.read_csv(datag)
            # geo = gpd.GeoDataFrame(geo, crs="EPSG:4326")
            gs = gpd.GeoSeries.from_wkt(datageo['geometry'])
            gdfall = gpd.GeoDataFrame(datageo, geometry=gs, crs="EPSG:4326")

            gdf = gdfall[['ID','2019', 'geometry', 'latitude', 'longitude']]
            # gdf['month'] = '2019'
            # st.write(gdf)
            # gdf['month'] = pd.to_datetime(gdf['month'] * 1000 + gdf['2019'], format='%Y%j')
            # gdf['month'] = gdf['month'].dt.month
            # st.write(gdf)

            gdf = gdf.dropna()

            # gdf = datageo #gpd.read_file(gdf_path)
            legend_name = 'DOY'
            title = 'Distribution of estimated flooding events'
            # gdf_path = ''

            # gdf ['doy'] = pd.to_datetime(gdf['flooding_date']).dt.dayofyear
            # param = y
            # gdfx_ [ param] = dfx_[param]
            on_column = '2019'
            gdf['coords'] = gdf['geometry'].apply(lambda x: x.centroid.coords[:])
            gdf['coords'] = [coords[0] for coords in gdf['coords']]
            # gdf['latitude'] = [coords[1] for coords in gdf['coords']]
            # gdf['longitude'] = [coords[0] for coords in gdf['coords']]


            # fig = px.scatter_mapbox(gdf, lat="latitude", lon="longitude", hover_name="ID",
            #                         hover_data=[on_column],
            #                         color_continuous_scale=px.colors.sequential.Rainbow, zoom=6,
            #                         height=600, color=on_column, #size=oncolumn
            #
            #                         )
            fig.update_layout(mapbox_style= 'white-bg') #"open-street-map") , 'dark-bg'
            fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
            # LAYOUT VAN DE VISUALISATIE
            fig.update_layout(
                font_family="Poppins",
                font_color="#002072",
                title_font_family="Poppins",
                title_font_color="#002072",
                legend_title="Distribution of flooding events",
                legend_title_font_color="#002072",
            )
            # st.plotly_chart(fig, use_container_width=True, )


            fig = px.choropleth(gdf, locations='ID', color=on_column)


            # st.plotly_chart(fig, use_container_width=True, )


            #--------------------------------------------------------------------------
            geocol = gdfall.select_dtypes(exclude='object').columns.to_list()

            for val in ['ID','geometry', 'latitude', 'longitude']:
                if val in geocol:
                    geocol.remove(val)


            if len(num_columns) == 0:
                st.write('There is no data in the data.')
            else:
                selected_num_cols = functions.multiselect_container('Choose columns for spatial Distribution plots:',
                                                                            geocol, '+')
                # st.subheader('Distribution of numerical columns')
                st.subheader("### Spatial distribution: flooding timing over years : ")
                i = 0

                while (i < len(selected_num_cols)):
                    c1, c2 = st.columns( 2 )
                    for j in [ c1, c2 ]:

                        if ( i >= len ( selected_num_cols ) ):
                            break
                        # on_column = 'DOY-' + selected_num_cols[i]
                        # title = 'Distribution of flooding events in ' + selected_num_cols[i]
                        # gdftmp = gdfall[['ID',selected_num_cols[i], 'geometry', 'latitude', 'longitude']]
                        # gdftmp= gdftmp.dropna()
                        # gdftmp = gdftmp.rename (columns={selected_num_cols[i]:on_column})
                        on_column = 'DOY-' + selected_num_cols[i]
                        title = 'Distribution of flooding events in ' + selected_num_cols[i]
                        gdftmp = gdfall[['ID',selected_num_cols[i], 'geometry', 'latitude', 'longitude']]
                        gdftmp= gdftmp.dropna()
                        gdftmp = gdftmp.rename (columns={selected_num_cols[i]:on_column})
                        # st.write(gdftmp)

                        gdftmp['day_of_the_year'] = gdftmp [on_column].astype(int)

                        gdftmp['year'] = int(selected_num_cols[i])
                        gdftmp["combined"] = gdftmp["year"] * 1000 + gdftmp["day_of_the_year"]
                        gdftmp["date"] = pd.to_datetime( gdftmp["combined"], format="%Y%j")
                        list_month = list ( gdftmp["date"].dt.month.unique() )
                        # lust_month = [int(i) for i in list_month]
                        # st.write(list_month [0] )

                        # hour_to_filter = j.radio(label="Month:", options=list_month, index=2)
                        # gdftmp = gdftmp[gdftmp['date'].dt.month == hour_to_filter]

                        opts = list_month
                        known_variables ={}
                        j.markdown('Choose the month you want to visualize in : ' +selected_num_cols[i])
                        known_variables = { symbol  : j.checkbox(f"{symbol}{'--'}{(selected_num_cols[i])}") for symbol in opts }
                        # st.write( known_variables.values() )
                        # st.write( known_variables .keys())

                        # gdftmp = gdftmp [gdftmp ['date'].dt.month == 7]
                        # st.write(gdftmp)
                        filter = [ val for val in list( known_variables.values() ) if val is True ]


                        filter1 = [ k for k in list(known_variables .keys()) if known_variables [k] is True ]


                        # filter2 = [ str(a.split() )for a in filter1 ]

                        # st.write (filter1)

                        gdftmp = gdftmp[gdftmp['date'].dt.month.isin(filter1)]
                        fig = px.scatter_mapbox(gdftmp, lat="latitude", lon="longitude", hover_name="ID",
                                                hover_data=[on_column],
                                                color_continuous_scale=px.colors.sequential.Rainbow, zoom=8,
                                                 color=on_column, labels=title  # size=oncolumn,

                                                )
                        fig.update_layout(mapbox_style='white-bg')  # "open-street-map") , 'dark-bg'
                        fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
                        # LAYOUT VAN DE VISUALISATIE
                        fig.update_layout(
                            font_family="Poppins",
                            font_color="#002072",
                            title_font_family="Poppins",
                            title_font_color="#002072",
                            # legend_title="Distribution of flooding events",
                            legend_title_font_color="#002072",
                        )
                        j.plotly_chart(fig, use_container_width=True,title=title)

                        i += 1

    #========================================================================================================================

    if 'GRID-LEVEL ANALYSIS' in vizuals:
        st.write('______________________________________________________________________________________________')
        st.write('______________________________________________________________________________________________')

        st.subheader("## GRID - LEVEL ANALYSIS  ")
        # datagridlevel = '/home/glorie/Documents/DASHBOARD/streamlit/GRID2917/data/dag_grid2917.csv'
        data = pd.read_csv(datagridlevel)
        data = data.rename(columns={'planted_area': 'flooded_area'})
        date_ = 'date'
        x='DOY'
        y='flooded_area'
        col='year'
        title='Trend of flooded area in the grid'

        data[x] = pd.to_datetime(data[date_]).dt.dayofyear
        data[col] =data[col].astype(str)
        data[y] = pd.to_numeric(data[y])
        # st.write(data)
        ids = data['ID'].unique()
        ids = ids[1500:1530]
        num_ids = ids

        # idsz = ids[0]
        # dataz = data[data['ID'].isin([idsz])].dropna()
        # # st.write(dataz)
        # chart = functions.get_chart_ts(dataz, x, y, col, title)
        # # # annotation_layer = functions.chart_ts_update()
        # st.altair_chart(
        #       chart,
        #       use_container_width=True
        #   )

        # st.altair_chart(
        #     (chart + annotation_layer).interactive(),
        #     use_container_width=False
        # )

        if len(num_ids) == 0:
            st.write('There is no grids in the data.')
        else:
            selected_num_cols = functions.multiselect_container('Choose the grid you want to visualise:',
                                                                        num_ids, 'Grid.')
            # st.subheader('Distribution of numerical columns')
            st.subheader("### Evolution of flooded area : ")
            i = 0
            cpt =0
            while (i < len(selected_num_cols)):
                c1, c2 = st.columns( 2 )
                for j in [ c1, c2 ]:

                    if ( i >= len ( selected_num_cols ) ):
                        break

                    datagrid = data[data['ID'].isin([str(selected_num_cols[i])])].dropna()
                    # st.write(data)

                    chart = functions.get_chart_ts(datagrid, x, y, col, title+'' + str(selected_num_cols[i]))

                    j.altair_chart(
                        chart,
                        use_container_width=True
                    )

                    i += 1
                    cpt+=1

    # ========================================================================================================================
    #-----------------------------------------COMPARE RESULTS ----------------------------------------------------
    if 'COMPARISON' in vizuals:
        st.write('______________________________________________________________________________________________')
        st.write('______________________________________________________________________________________________')

        # dfsaed = pd.read_excel('/home/glorie/Documents/DASHBOARD/compararison/SAED Predictions.xlsx', index_col=0)
        st.subheader("## RESULT COMPARISON  ")
        st.markdown(
            """
            **NOTE - For GOMPERTZ approach:**
            
            - In 2019 : The prediction starts on *2019-09-01*
            - In 2020 : The prediction starts on *2020-08-31*
            - In 2021 : The prediction starts on *2021-09-02*
            - In 2022 : The prediction starts on *2022-08-15*
            """
        )
        # {2019: ['2019-09-01',
        #         '2019-09-06',
        #         '2019-09-13',
        #         '2019-09-18',
        #         '2019-09-25',
        #         '2019-09-30',
        #         '2019-10-07'],
        #  2020: ['2020-08-31',
        #         '2020-09-07',
        #         '2020-09-12',
        #         '2020-09-19',
        #         '2020-09-24',
        #         '2020-10-01',
        #         '2020-10-06'],
        #  2021: ['2021-09-02',
        #         '2021-09-07',
        #         '2021-09-14',
        #         '2021-09-19',
        #         '2021-09-26',
        #         '2021-10-01',
        #         '2021-10-08'],
        #  2022: ['2022-08-15',
        #         '2022-08-17',
        #         '2022-08-20',
        #         '2022-08-22',
        #         '2022-08-25',
        #         '2022-08-27',
        #         '2022-08-30',
        #         '2022-09-01',
        #         '2022-09-04',
        #         '2022-09-06']}
        data = pd.read_csv('data/comparison/data_prediction.csv')


        period ='RAINY'
        data = data[data['period']==period]


        # st.write(data)

        # datagridlevel = '/home/glorie/Documents/DASHBOARD/streamlit/GRID2917/data/dag_grid2917.csv'
        # data = pd.read_csv(datagridlevel)
        # data = data.rename(columns={'planted_area': 'flooded_area'})
        date_ = 'date'
        x='DOY'
        y='flooded_area'
        col='data_source'
        title=''

        data[x] = pd.to_datetime(data[date_]).dt.dayofyear
        # data = data.sort_values(by=[x], ascending=False)
        data[col] =data[col].astype(str)
        data[y] = pd.to_numeric(data[y])
        # data['year'] = data['year'].astype(str) + '_'
        # st.write(data)
        ids = data['year'].unique()
        # ids = ids[1500:1530]
        num_idsc = ids

        # idsz = ids[0]
        # dataz = data[data['ID'].isin([idsz])].dropna()
        # # st.write(dataz)
        # chart = functions.get_chart_ts(dataz, x, y, col, title)
        # # # annotation_layer = functions.chart_ts_update()
        # st.altair_chart(
        #       chart,
        #       use_container_width=True
        #   )

        # st.altair_chart(
        #     (chart + annotation_layer).interactive(),
        #     use_container_width=False
        # )

        if len(num_idsc) == 0:
            st.write('There is no grids in the data.')
        else:
            selected_num_cols = functions.multiselect_container('Choose the year you want to visualise:',
                                                                        num_idsc, 'years_')
            # st.subheader('Distribution of numerical columns')
            st.subheader("### Evolution of flooded area : ")
            i = 0
            cpt =0
            while (i < len(selected_num_cols)):
                c1, c2 = st.columns( 2 )
                for j in [ c1, c2 ]:

                    if ( i >= len ( selected_num_cols ) ):
                        break

                    datagrid = data[data['year'].isin( [  selected_num_cols[i]  ]  )].dropna()
                    # st.write(datagrid)
                    chart = functions.get_chart_ts(datagrid, x, y, col, title+' in ' + str(selected_num_cols[i]))

                    j.altair_chart(
                        chart,
                        use_container_width=True
                    )

                    i += 1
                    cpt+=1


#========================================================================================================
    if 'GOMPERTZ-EVALUATION' in vizuals:
        st.write('______________________________________________________________________________________________')
        st.write('______________________________________________________________________________________________')
        st.subheader('EVALUATION OF GOMPERTZ METHOD IN 2022')
        dataval = pd.read_csv('data/DATA_TO_SHARE/rain_2022_update/grid52/flooding_Dagana2022.csv')
        datap = pd.read_csv('data/DATA_TO_SHARE/rain_2022_update/grid52/flooding_proportion_Dagana2022.csv')
        st.write('DATA - FLOODING')
        st.write(dataval)
        st.write('DATA - PROPORTION')
        # st.write(datap)
        st.write('=========== Complete images - ')
        data_upda = pd.read_csv('data/DATA_TO_SHARE/rain_2022_update/grid52/up/flooding_proportion_Dagana2022.csv')
        st.write(data_upda)

        st.write('----------------------------')

        dataCOM = pd.read_csv('data/comparison/data_prediction.csv')
        dataCOM = dataCOM[ dataCOM['year']==2022 ]
        latext = r'''
        #### HOW IT WORKS
        With all $n$ images  available at time $t$, predict the total area that will be flooded during the  season 

        '''
        st.write(latext)
        # st.write(dataCOM)

        # st.write('PLEAS WAIT')
        # data = pd.read_csv('data/gompertz_eval/grid52/prediction_growth_ts.csv')
        #----------------------  ALL DATA
        data =pd.read_csv('data/DATA_TO_SHARE/GRID2917-20220922T122839Z-001/output_model/grid52/grid52_n/prediction_growth_ts.csv')
        gcol  =  data.columns
        gcol =gcol[1:]
        data  = data [ gcol]
        data['class'] = 'GOMPERTZ'
        #HIS
        data_hist22 = pd.read_csv('data/DATA_TO_SHARE/GRID2917-20220922T122839Z-001/output_model/grid52/grid52_n/hist/prediction_growth_ts.csv')
        data_hist22 ['class'] = 'GOMPERTZ_hist'

        #=============
        dataallim = pd.read_csv('data/DATA_TO_SHARE/GRID2917-20220922T122839Z-001/output_model/grid52/grid52v2/prediction_growth_ts.csv')
        dataallim  = dataallim [ dataallim.columns[1:] ]
        dataallim['class'] = 'GOMPERTZ'
        #HIS
        dataallim_hist22 = pd.read_csv('data/DATA_TO_SHARE/GRID2917-20220922T122839Z-001/output_model/grid52/grid52v2/hist/prediction_growth_ts.csv')
        dataallim_hist22['class'] = 'GOMPERTZ_hist'

        data_c = pd.read_csv('data/DATA_TO_SHARE/rain_2022_update/grid52/up/flooding_proportion_Dagana2022.csv')
        dataa_c = pd.read_csv('data/DATA_TO_SHARE/rain_2022_update/grid52/up/flooding_Dagana2022.csv')
        dataa_c = dataa_c[ list(data_c.filter(regex=('\d{4}-?\d{2}-?\d{2}$')).columns) ]
        # dataa_c = dataa_c.filter(regex=('\d{4}-?\d{2}-?\d{2}$'))
        # st.write(dataa_c)
        sf = dataa_c .sum(axis=0)
        # st.write(sf)
        df_ = pd.DataFrame()
        df_['time_t'] = list(sf.index)
        df_['area_t'] = list(sf.values)
        df_['class'] = 'RS'
        dataa_c = df_
        # st.write(dataa_c)




        # st.write(data)
        # data.to_csv('data/gompertz_eval/grid52/gompertz_prediction_rainy_2022.csv')
        #-------------------
        dataCOM =dataCOM.rename(columns={'date':'time_t', 'flooded_area':'area_t', 'data_source':'class'})
        dataCOM = dataCOM[['time_t', 'area_t', 'class']]
        # st.write(dataCOM.head())
        datRS = dataCOM[dataCOM ['class'] =='RS']
        datRS ['class'] ='RS'

        # data = data.append(datRS)
        # data.to_csv('data/gompertz_eval/grid52/gompertz_prediction_rainy_2022_up.csv')
        # st.write(datRS)

        datsaed = dataCOM[dataCOM ['class'] =='SAED_achievement']
        datsaed ['class'] = 'SAED_achiev'

        datsaedf = dataCOM[dataCOM ['class'] =='SAED_forecast']
        datsaedf ['class'] = 'SAED_forecast'
        # st.write(datsaed)
        # data = data.append(datfilter)



        # data = data.rename({ 'area_t':'area' , 'time_t':'time' })
        # st.write(data)

        #MAX FUNCTION
        def max_value (df, col):
            listval = list(df[col].values)
            res = [listval[0]]
            for v in listval[1:]:
                if v > res[-1] :
                    res.append(v)
                else:
                    res.append(res[-1] )
            return res

        res = max_value(data, 'area_t')
        tmpdf = pd.DataFrame()
        tmpdf['time_t'] = data['time_t']
        tmpdf ['area_t'] = res
        tmpdf ['class'] = 'GMZ_AJUST_PRED'
        # data =data.append(tmpdf)
        data = data.append([datRS, datsaed, datsaedf])
        # data = data.append([datRS])
        data ['time_t'] = pd.to_datetime(data['time_t']).dt.date

        #================
        data_hist22 = data_hist22.append([datRS, datsaed, datsaedf])
        data_hist22['time_t'] = pd.to_datetime( data_hist22['time_t']).dt.date

        # st.subheader('data_hist22')
        # st.write(data_hist22)



        #--------------
        dataallim = dataallim.append([dataa_c, datsaed, datsaedf])
        dataallim['time_t'] = pd.to_datetime(dataallim['time_t']).dt.date
        #============
        dataallim_hist22 = dataallim_hist22.append([dataa_c, datsaed, datsaedf])
        dataallim_hist22['time_t'] = pd.to_datetime(dataallim_hist22['time_t']).dt.date


        # data.to_csv("data/gompertz_eval/grid52/prediction_growth_ts_from_jul_all.csv")

        # st.write(data)


        #------------------FROM AUGUST --------------------
        # st.write('----------- LET " S START THE PREDICTION IN AUGUST -----------------')
        data_aug = pd.read_csv("data/gompertz_eval/grid52/prediction_growth_ts_from_aug.csv")
        data_aug = pd.read_csv('data/DATA_TO_SHARE/GRID2917-20220922T122839Z-001/output_model/grid52/grid52_n/from_aug/prediction_growth_ts.csv')

        data_aug = data_aug[list(data_aug.columns)[1:]]
        res = max_value(data_aug, 'area_t')
        # data_aug['max_area_t'] = res
        data_aug ['class'] = 'GOMPERTZ' #'GMZ_INIT_PRED'

        tmpdf = pd.DataFrame()
        tmpdf['time_t'] = data_aug['time_t']
        tmpdf ['area_t'] = res
        # tmpdf ['class'] =  #'GMZ_AJUST_PRED'
        # data_aug = data_aug.append(tmpdf)
        data_aug =data_aug.append([datRS, datsaed, datsaedf])

        data_aug['time_t'] = pd.to_datetime(data_aug['time_t']).dt.date

        # data_aug.to_csv("data/gompertz_eval/grid52/prediction_growth_ts_from_aug_all.csv")
        # st.write(data_aug)



        fig_col1_pred, fig_col2_pred = st.columns(2)
        with fig_col1_pred:
            #-------------------------------
            st.write('----------- THE PREDICTION STARTS IN JULY -----------------')
            st.write(data)
            #----------------------------------
            x='time_t'
            y ='area_t'
            col = 'class'
            chart = functions.get_chart_ts_up(data, x, y, col, '')
            st.altair_chart(
                chart,
                use_container_width=True
            )

            #-------------------------------
            st.write('----------- THE PREDICTION STARTS IN JULY : WITH ONLY COMPLETE IMAGE (every 5 days in principle) ')

            #----------------------------------
            x='time_t'
            y ='area_t'
            col = 'class'
            chart = functions.get_chart_ts_up(dataallim , x, y, col, '')

            st.altair_chart(
                chart,
                use_container_width=True
            )
            # st.write(dataallim)


        with fig_col2_pred:
            #-------------------------------
            # st.write('-----------  PREDICTION STARTS IN AUG-----------------')
            st.write('-----------  PREDICTION WITH  HISTORICAL DATA----------')
            # st.write(data_aug)
            # #----------------------------------
            # x='time_t'
            # y ='area_t'
            # col = 'class'
            # chart = functions.get_chart_ts(data_aug, x, y, col, '')
            #
            # st.altair_chart(
            #     chart,
            #     use_container_width=True
            # )
            st.write(data_hist22)
            x='time_t'
            y ='area_t'
            col = 'class'
            chart = functions.get_chart_ts_up(data_hist22, x, y, col, '')
            st.altair_chart(
                chart,
                use_container_width=True
            )

            st.write('')
            #-----------------------------------------
            # st.write(dataallim_hist22)
            x='time_t'
            y ='area_t'
            col = 'class'
            chart = functions.get_chart_ts_up(dataallim_hist22, x, y, col, '')
            st.altair_chart(
                chart,
                use_container_width=True
            )



        # \Delta G = \Delta\sigma \frac{a}{b}
        latext = r'''
        ## 
        ### Equation  : MAXIMUM VALUE
        
        $$ 
                        y = f(t) =  \max(a_{i})_{i=1}^{t}     
        $$ 
        where $a_t$ is the total area predicted at time $t$  
        
        '''
        st.write(latext)


        st.subheader(' GOMPERTZ  : PLOT  - WITHOUT HISTORICAL DATA')
        path = 'data/DATA_TO_SHARE/GRID2917-20220922T122839Z-001/output_model/grid52/grid52_n/df/'
        dfpath = os.listdir(path)
        sum_ = []
        dfs =pd.DataFrame()
        cp = 0
        for p in dfpath :
            f = pd.read_csv(path+p)
            f = f.filter(regex=('\d{4}-?\d{2}-?\d{2}$'))
            sf = f.sum(axis=0)
            # st.write(sf)
            df_ = pd.DataFrame()
            df_['area'] = list(sf.values)
            df_ ['time'] = list(sf.index)
            df_['class'] = p.split('.')[0]  #'df'+str(cp)
            dfs = dfs.append(df_)
            cp += 1
        dfs['time'] = pd.to_datetime(dfs['time']).dt.date
        # st.write(dfs)
        # dfs =  .filter(regex=('\d{4}-?\d{2}-?\d{2}$'))
        # st.write(dfs)
        fig_col1_v1, fig_col2_v2 = st.columns(2)

        # dfs = dfs[dfs['class'] ]== 'df'+str(cp)

        with fig_col1_v1:
            #-------------------------------
            st.write('----------- ---- -----------------')
            # st.write(dfs)
            #----------------------------------
            x='time'
            y ='area'
            col = 'class'
            chart = functions.get_chart_ts(dfs, x, y, col, '')


            st.altair_chart(
                chart,
                use_container_width=True
            )


        ids = dfs['time'].unique()
        ids = ['df_'+str(c) for c in ids]

        num_ids = ids
        # st.write(ids)
        if len(num_ids) == 0:
            st.write('There is no grids in the data.')
        else:
            selected_num_cols = functions.multiselect_container('Choose the DF you want to visualise:',
                                                                num_ids, 'Grid-')
            # st.subheader('Distribution of numerical columns')
            # st.subheader("### GOMPERTZ  : ")
            i = 0
            cpt = 0
            while (i < len(selected_num_cols)):
                c1, c2 = st.columns(2)
                for j in [c1, c2]:

                    if (i >= len(selected_num_cols)):
                        break

                    datagrid = dfs[dfs['class']==selected_num_cols[i ] ] .dropna()
                    # st.write('yyyyy')
                    # st.write(datagrid)
                    x = 'time'
                    y = 'area'
                    col = 'class'
                    chart = functions.get_chart_ts(datagrid, x, y, col, 'plot'+' for ' + str(selected_num_cols[i]))

                    j.altair_chart(
                        chart,
                        use_container_width=True
                    )

                    i += 1
                    cpt+=1



        st.subheader(' GOMPERTZ  : PLOT  - WITH HISTORICAL DATA')
        path = 'data/DATA_TO_SHARE/GRID2917-20220922T122839Z-001/output_model/grid52/grid52_n/hist/df/'
        dfpath = os.listdir(path)
        sum_ = []
        dfs =pd.DataFrame()
        cp = 0
        for p in dfpath :
            f = pd.read_csv(path+p)
            f = f.filter(regex=('\d{4}-?\d{2}-?\d{2}$'))
            sf = f.sum(axis=0)
            # st.write(sf)
            df_ = pd.DataFrame()
            df_['area'] = list(sf.values)
            df_ ['time'] = list(sf.index)
            df_['class'] = p.split('.')[0]  #'df'+str(cp)
            dfs = dfs.append(df_)
            cp += 1
        dfs['time'] = pd.to_datetime(dfs['time']).dt.date
        # st.write(dfs)
        # dfs =  .filter(regex=('\d{4}-?\d{2}-?\d{2}$'))
        # st.write(dfs)
        fig_col1_v1, fig_col2_v2 = st.columns(2)

        # dfs = dfs[dfs['class'] ]== 'df'+str(cp)

        with fig_col1_v1:
            #-------------------------------
            st.write('----------- -----------------')
            # st.write(dfs)
            #----------------------------------
            x='time'
            y ='area'
            col = 'class'
            chart = functions.get_chart_ts(dfs, x, y, col, '')


            st.altair_chart(
                chart,
                use_container_width=True
            )

        ids = dfs['time'].unique()
        ids = ['df_'+str(c) for c in ids]

        num_ids = ids
        # st.write(ids)
        if len(num_ids) == 0:
            st.write('There is no grids in the data.')
        else:
            selected_num_cols = functions.multiselect_container('Choose the DF you want to visualise_:',                                                               num_ids, '!')
            # st.subheader('Distribution of numerical columns')
            # st.subheader("### GOMPERTZ  : ")
            i = 0
            cpt = 0
            while (i < len(selected_num_cols)):
                c1, c2 = st.columns(2)
                for j in [c1, c2]:

                    if (i >= len(selected_num_cols)):
                        break

                    datagrid = dfs[dfs['class']==selected_num_cols[i ] ] .dropna()
                    # st.write('yyyyy')
                    # st.write(datagrid)
                    x = 'time'
                    y = 'area'
                    col = 'class'
                    chart = functions.get_chart_ts(datagrid, x, y, col, 'plot_'+' for ' + str(selected_num_cols[i]))

                    j.altair_chart(
                        chart,
                        use_container_width=True
                    )

                    i += 1
                    cpt+=1


        #--------------CAL S1 EVAL
        gompertz_eval__rain_with_s1.gompertz_eval_rain_with_s1()
        # ids = ids[1500]
        # data = data[data['ID'].isin([ids])].dropna()
        # # data[x] = pd.to_datetime(data[x])
        # # st.write(data)
        # chart = functions.get_chart_ts(data, x, y, col, title)
        # # annotation_layer = functions.chart_ts_update()
        # st.altair_chart(
        #      chart,
        #      use_container_width=True
        #  )

        # st.altair_chart(
        #     (chart + annotation_layer).interactive(),
        #     use_container_width=False
        # )


    # hist_axis = st.sidebar.multiselect(label="Histogram Ingredient", options=measurements, default=["2019", "2020"])

    # bins = st.sidebar.radio(label="Bins :", options=[10,20,30,40,50], index=4)
    # #
    # hist_fig = plt.figure(figsize=(6,4))
    # #
    # hist_ax = hist_fig.add_subplot(111)
    # #
    # sub_breast_cancer_df = datahist[["2019", "2020"]]
    # #
    # sub_breast_cancer_df.plot.hist(bins=bins, alpha=0.7, ax=hist_ax, title="Distribution of Measurements");


    # # measurements = datahist.drop(labels=["year"], axis=1).columns.tolist()
    # hist_axis = st.sidebar.multiselect(label="Histogram Ingredient", options=measurements, default=["2019", "2020"])
    # bins = st.sidebar.radio(label="Bins :", options=[10,20,30,40,50], index=4)

    # if hist_axis:
    #     hist_fig = plt.figure(figsize=(6,4))
    #
    #     hist_ax = hist_fig.add_subplot(111)
    #
    #     sub_breast_cancer_df = datahist[hist_axis]
    #
    #     sub_breast_cancer_df.plot.hist(bins=bins, alpha=0.7, ax=hist_ax, title="Distribution of Measurements");
    # else:
    #     hist_fig = plt.figure(figsize=(6,4))
    #
    #     hist_ax = hist_fig.add_subplot(111)
    #
    #     sub_breast_cancer_df = datahist[["2019", "2020"]]
    #
    #     sub_breast_cancer_df.plot.hist(bins=bins, alpha=0.7, ax=hist_ax, title="Distribution of Measurements");




    #=================================== FEEDBACK =================================
    #----------------------------------------------------------------------------------
    # feebb = st.sidebar.checkbox('Any feedback ?')
    # if feebb:
    #     feedback.main()