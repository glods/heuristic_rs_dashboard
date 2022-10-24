import streamlit as st
import pandas as pd

import functions

import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
import seaborn as sns
import geopandas as gpd
import fiona


def harvest():

    fiona.drvsupport.supported_drivers['geojson'] = 'rw' # enable KML support which is disabled by default
    fiona.drvsupport.supported_drivers['GeoJSON'] = 'rw'
    ##============================= INPUT DATA=============================
    #for grid level


    # gpd.io.file.fiona.drvsupport.supported_drivers['KML'] = 'rw'
    # gpd.io.file.fiona.drvsupport.supported_drivers ['GeoJSON'] = 'rw'

    #How to increase the width of web page
    # st.set_page_config(layout="wide")
    # st.set_page_config(layout="wide", page_icon="☀",)

    # st.markdown("# GRID 2917  VISUALISATION : FLOODING 💦")



    st.sidebar.header("HARVEST ESTIMATES")
    # st.title('GRID 2917  VISUALISATION')


    datagridlevel = 'data/DASHBOARD/streamlit/GRID2917/data/dag_grid2917.csv'
    # DATE_COLUMN = 'date'
    DATE_COLUMN = 'harvest_date'
    i='2019'
    DATA_URL =  'data/DATA_TO_SHARE/GRID2917-20220922T122839Z-001/GRID2917/Dagana/DRY_HOT_SEASON/'+i+'/harvest/harvest_Dagana'+i+'.csv'
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
    data = data.rename(columns={'planted_area':'harvest_area'})

    if st.checkbox('Show spatial points'):
        st.map(data)

    if st.checkbox('Show raw data'):
         st.subheader('Sample data '+str(i))
         st.write(data)
    #========================================================
    data = data[data['harvest_date'] != '0']
    data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN])



    # Some number in the range 0-23
    # hour_to_filter = st.slider('Month', 0, 5, 2)
    # hour_to_filter = st.radio(label="Month :", options=[1, 2, 3, 4, 5], index=2)
    # filtered_data = data[data[DATE_COLUMN].dt.month == hour_to_filter]

    # st.subheader('Flooding areas during the month %s' % hour_to_filter)
    # st.map(filtered_data)
    #================================================




    datav = pd.read_csv(DATA_URL)

    season ='DRY_HOT_SEASON'
    y = ['2019' ,'2020', '2021', '2022']

    datahist  = gpd.GeoDataFrame()
    datahist1  = gpd.GeoDataFrame()
    for i in y :
        datav = pd.read_csv('data/DATA_TO_SHARE/GRID2917-20220922T122839Z-001/GRID2917/Dagana/'+season+'/'+i+'/harvest/harvest_Dagana'+i+'.csv' )
        datav =datav[['ID','harvest_date', 'aoi_area', 'total_harvest_area', 'latitude', 'longitude', 'geometry' ]]
        datav = datav[datav['harvest_date']!='0' ]
        datatmp =gpd.GeoDataFrame()
        # datatmp[i] = pd.to_datetime( datav.flooding_date ).dt.dayofyear

        # cold = [c for c in datav.columns if i in c]
        # doy_col = list(datav[cold] .sum(axis=0).index )
        #
        # datatmp [ doy_col] = [list(datav[cold] .sum(axis=0).values)]

        datatmp['ID'] = datav ['ID']
        datatmp ['year'] =  i
        datatmp [ 'DOY'] = pd.to_datetime( datav.harvest_date ).dt.dayofyear
        datatmp['total_harvest_area'] = datav['total_harvest_area']
        datahist1 = datahist1.append(datatmp)
        #
        datahist['ID'] = datav['ID']
        datahist[i] = pd.to_datetime(datav.harvest_date).dt.dayofyear
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
    'COMPARISON'
                   ]
    # functions.sidebar_space(3)
    vizuals = st.sidebar.multiselect("Choose which visualizations you want to see 👇", all_vizuals)

    num_columns = datahist.select_dtypes(exclude='object').columns
    cat_columns = datahist.select_dtypes(include='object').columns
    # bins = st.sidebar.radio(label="Bins :", options=[10,20,30,40,50,70], index=4)



    if 'Summary statistics' in vizuals:
        st.write('______________________________________________________________________________________________')
        st.write('______________________________________________________________________________________________')


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
            # st.markdown("Boxplot")
            bins = st.slider('slide-bins', 5, 30, 10)

            fig = ff.create_distplot(
                hist_data, group_labels, bin_size=bins)

            # st.plotly_chart(fig, use_container_width=False)
            st.write(fig)

        #--------------------------------------------------
            fig2 = px.box(datahist1, y="DOY", color="year",
                          # or violin, rug
                          hover_data=datahist1.columns, notched=True)
            st.plotly_chart(fig2, use_container_width=True)

        with fig_col2:
        # --------------------------------------------------
            # st.markdown(" ")
            # fig2 = px.box(datahist1, y="DOY", color="year",
            #               # or violin, rug
            #               hover_data=datahist1.columns, notched=True)
            # st.write(fig2)
            fig = sns.displot ( datahist1, x='DOY', hue="year", kind='kde', alpha=0.5 )
            st.pyplot(fig)

        #--------------------------------------------------
            # fig2 = px.box(datahist1, y="total_flooding_area", color="year",
            #                # or violin, rug
            #                hover_data=datahist1.columns, notched=True)
            # st.plotly_chart(fig2, use_container_width=True)
            #



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
            st.subheader("### Histogram: harvesttiming over years : ")
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
                st.subheader('Box plots :  Harvest timing over years')
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
            datag ='data/DATA_TO_SHARE/GRID2917-20220922T122839Z-001/GRID2917/Dagana/DRY_HOT_SEASON/'+yar+'/harvest/harvest_Dagana'+yar+'.csv'
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
                    z=geo['harvest_date'],
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
            title = 'Distribution of estimated harvest events'
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
                legend_title="Distribution of harvest events",
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
                                                                            geocol, '')
                # st.subheader('Distribution of numerical columns')
                st.subheader("### Spatial distribution:harvest timing over years : ")
                i = 0

                while (i < len(selected_num_cols)):
                    c1, c2 = st.columns( 2 )
                    for j in [ c1, c2 ]:

                        if ( i >= len ( selected_num_cols ) ):
                            break
                        # on_column = 'DOY-' + selected_num_cols[i]
                        # title = 'Distribution of harvest events in ' + selected_num_cols[i]
                        # gdftmp = gdfall[['ID',selected_num_cols[i], 'geometry', 'latitude', 'longitude']]
                        # gdftmp= gdftmp.dropna()
                        # gdftmp = gdftmp.rename (columns={selected_num_cols[i]:on_column})

                        on_column = 'DOY-' + selected_num_cols[i]
                        title = 'Distribution of harvest events in ' + selected_num_cols[i]
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
        datagridlevel = 'data/harvest/dag_grid2917dhs.csv'
        data = pd.read_csv(datagridlevel)
        data = data.rename(columns={'planted_area': 'harvest_area'})
        st.write(data)
        date_ = 'date'
        x='DOY'
        y='harvest_area'
        col='year'
        title='Trend of harvested area in the grid'

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
                                                                        num_ids, 'Grid')
            # st.subheader('Distribution of numerical columns')
            st.subheader("### Trend of harvested area : ")
            i = 0
            cpt =0
            while (i < len(selected_num_cols)):
                c1, c2 = st.columns( 2 )
                for j in [ c1, c2 ]:

                    if ( i >= len ( selected_num_cols ) ):
                        break

                    datagrid = data[data['ID'].isin([str(selected_num_cols[i])])].dropna()
                    chart = functions.get_chart_ts(datagrid, x, y, col, title+' ' + str(selected_num_cols[i]) )

                    j.altair_chart(
                        chart,
                        use_container_width=True
                    )

                    i += 1
                    cpt+=1
    #
    # # ========================================================================================================================
    # #-----------------------------------------COMPARE RESULTS ----------------------------------------------------
    # if 'COMPARISON' in vizuals:
    #     # dfsaed = pd.read_excel('/home/glorie/Documents/DASHBOARD/compararison/SAED Predictions.xlsx', index_col=0)
    #     st.subheader("## RESULT COMPARISON  ")
    #     st.markdown(
    #         """
    #         **NOTE - For GOMPERTZ approach:**
    #
    #         - In 2019 : The prediction starts on *2019-04-18*
    #         - In 2020 : The prediction starts on *2020-04-22*
    #         - In 2021 : The prediction starts on *2021-04-22*
    #         - In 2022 : The prediction starts on *2022-04-12*
    #         """
    #     )
    #
    #     {2019: ['2019-04-18',
    #             '2019-04-23',
    #             '2019-04-28',
    #             '2019-05-03',
    #             '2019-05-08',
    #             '2019-05-13',
    #             '2019-05-18',
    #             '2019-05-23',
    #             '2019-05-28'],
    #      2020: ['2020-04-22',
    #             '2020-04-27',
    #             '2020-05-02',
    #             '2020-05-07',
    #             '2020-05-12',
    #             '2020-05-17',
    #             '2020-05-22',
    #             '2020-05-27'],
    #      2021: ['2021-04-22',
    #             '2021-04-27',
    #             '2021-05-02',
    #             '2021-05-07',
    #             '2021-05-12',
    #             '2021-05-17',
    #             '2021-05-22',
    #             '2021-05-27'],
    #      2022: ['2022-04-12',
    #             '2022-04-17',
    #             '2022-04-22',
    #             '2022-04-27',
    #             '2022-05-02',
    #             '2022-05-12',
    #             '2022-05-17',
    #             '2022-05-22']}
    #
    #     data = pd.read_csv('/home/glorie/Documents/DASHBOARD/compararison/data_prediction_dhs.csv')
    #     period ='DHS'
    #     data = data[data['period']==period]
    #
    #     # st.write(data)
    #
    #     # datagridlevel = '/home/glorie/Documents/DASHBOARD/streamlit/GRID2917/data/dag_grid2917.csv'
    #     # data = pd.read_csv(datagridlevel)
    #     # data = data.rename(columns={'planted_area': 'flooded_area'})
    #     date_ = 'date'
    #     x='DOY'
    #     y='flooded_area'
    #     col='data_source'
    #     title=''
    #
    #     data[x] = pd.to_datetime(data[date_]).dt.dayofyear
    #     # data = data.sort_values(by=[x], ascending=False)
    #     data[col] =data[col].astype(str)
    #     data[y] = pd.to_numeric(data[y])
    #
    #     # st.write(data)
    #     ids = data['year'].unique()
    #     # ids = ids[1500:1530]
    #     num_idsc = ids
    #     # st.write(num_ids)
    #
    #     # idsz = ids[0]
    #     # dataz = data[data['ID'].isin([idsz])].dropna()
    #     # # st.write(dataz)
    #     # chart = functions.get_chart_ts(dataz, x, y, col, title)
    #     # # # annotation_layer = functions.chart_ts_update()
    #     # st.altair_chart(
    #     #       chart,
    #     #       use_container_width=True
    #     #   )
    #
    #     # st.altair_chart(
    #     #     (chart + annotation_layer).interactive(),
    #     #     use_container_width=False
    #     # )
    #
    #     if len(num_idsc) == 0:
    #         st.write('There is no grids in the data.')
    #     else:
    #         selected_num_cols = functions.multiselect_container('Choose the year you want to visualise:',
    #                                                                     num_idsc, 'years')
    #         # st.subheader('Distribution of numerical columns')
    #         st.subheader("### Evolution of flooded area : ")
    #         i = 0
    #         cpt =0
    #         while (i < len(selected_num_cols)):
    #             c1, c2 = st.columns( 2 )
    #             for j in [ c1, c2 ]:
    #
    #                 if ( i >= len ( selected_num_cols ) ):
    #                     break
    #
    #                 datagrid = data[data['year'].isin([selected_num_cols[i]])].dropna()
    #                 # st.write(datagrid)
    #                 chart = functions.get_chart_ts(datagrid, x, y, col, title+' in ' + str(selected_num_cols[i]))
    #
    #                 j.altair_chart(
    #                     chart,
    #                     use_container_width=True
    #                 )
    #
    #                 i += 1
    #                 cpt+=1


    # ========================================================================================================================

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