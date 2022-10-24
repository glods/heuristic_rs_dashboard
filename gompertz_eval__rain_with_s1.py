
import streamlit as st
import pandas as pd
import functions
import os
# ========================================================================================================

def gompertz_eval_rain_with_s1():
    st.write('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    st.subheader('EVALUATION OF GOMPERTZ METHOD IN 2022  - SENTINEL 1')
    st.write('______________________________________________________________________________________________')
    latext = r'''
    #### Description
        Detection of the flooding area : the crop map was generated from Sentinel 2;
        Sentinel data are used for the distribution of dates.

    '''
    st.write(latext)


    st.write('----------------------------')

    dataCOM = pd.read_csv('data/DATA_TO_SHARE/GRID52/flooding/with_s1/flooding_Dagana2022.csv')
    dataCOMf = dataCOM.filter(regex=('\d{4}-?\d{2}-?\d{2}$'))
    sf = dataCOMf.sum(axis=0)
    # st.write(sf)
    df_ = pd.DataFrame()
    df_['area_t'] = list(sf.values)
    df_['time_t'] = list(sf.index)
    df_['class'] ='RS'
    dataCOM = df_

    # st.write(dataCOM)
    # df_['class'] = [RS]  # 'df'+str(cp)
    # dfs = dfs.append(df_)
    # dataCOM = dataCOM[dataCOM['year'] == 2022]

    # st.write(dataCOM)

    # st.write('PLEASE WAIT')
    # data = pd.read_csv('data/gompertz_eval/grid52/prediction_growth_ts.csv')
    # ----------------------  ALL DATA
    data = pd.read_csv('data/DATA_TO_SHARE/GRID2917-20220922T122839Z-001/output_model/grid52/grid52_SENT1/prediction_growth_ts.csv')
    gcol = data.columns
    gcol = gcol[1:]
    data = data[gcol]
    data['class'] = 'GOMPERTZ'
    # HIS
    data_hist22 = pd.read_csv('data/DATA_TO_SHARE/GRID2917-20220922T122839Z-001/output_model/grid52/grid52_SENT1/hist/prediction_growth_ts.csv')
    data_hist22['class'] = 'GOMPERTZ_hist'

    # =============


    # st.write(data)
    # data.to_csv('data/gompertz_eval/grid52/gompertz_prediction_rainy_2022.csv')
    # -------------------
    # dataCOM = dataCOM.rename(columns={'date': 'time_t', 'flooded_area': 'area_t', 'data_source': 'class'})
    # dataCOM = dataCOM[['time_t', 'area_t', 'class']]
    # st.write(dataCOM.head())
    datRS = dataCOM[dataCOM['class'] == 'RS']
    datRS['class'] = 'RS'

    # data = data.append(datRS)
    # data.to_csv('data/gompertz_eval/grid52/gompertz_prediction_rainy_2022_up.csv')
    # st.write(datRS)

    datsaed = dataCOM[dataCOM['class'] == 'SAED_achievement']
    datsaed['class'] = 'SAED_achiev'

    datsaedf = dataCOM[dataCOM['class'] == 'SAED_forecast']
    datsaedf['class'] = 'SAED_forecast'


    # st.write(datsaed)
    # data = data.append(datfilter)

    # data = data.rename({ 'area_t':'area' , 'time_t':'time' })
    # st.write(data)

    # MAX FUNCTION
    def max_value(df, col):
        listval = list(df[col].values)
        res = [listval[0]]
        for v in listval[1:]:
            if v > res[-1]:
                res.append(v)
            else:
                res.append(res[-1])
        return res


    res = max_value(data, 'area_t')
    tmpdf = pd.DataFrame()
    tmpdf['time_t'] = data['time_t']
    tmpdf['area_t'] = res
    tmpdf['class'] = 'GMZ_AJUST_PRED'
    # data =data.append(tmpdf)
    data = data.append([datRS, datsaed, datsaedf])
    # data = data.append([datRS])
    data['time_t'] = pd.to_datetime(data['time_t']).dt.date

    # ================
    data_hist22 = data_hist22.append([datRS, datsaed, datsaedf])
    data_hist22['time_t'] = pd.to_datetime(data_hist22['time_t']).dt.date

    # st.subheader('data_hist22')
    # st.write(data_hist22)



    # data_aug.to_csv("data/gompertz_eval/grid52/prediction_growth_ts_from_aug_all.csv")
    # st.write(data_aug)

    fig_col1_pred, fig_col2_pred = st.columns(2)
    with fig_col1_pred:
        # -------------------------------
        st.write('----------- THE PREDICTION STARTS IN JULY -----------------')
        # st.write(data)
        # ----------------------------------
        x = 'time_t'
        y = 'area_t'
        col = 'class'
        chart = functions.get_chart_ts_up(data, x, y, col, '')
        st.altair_chart(
            chart,
            use_container_width=True
        )


    with fig_col2_pred:
        # -------------------------------
        # st.write('-----------  PREDICTION STARTS IN AUG-----------------')
        st.write('-----------  PREDICTION WITH  HISTORICAL DATA----------')

        # st.write(data_hist22)
        x = 'time_t'
        y = 'area_t'
        col = 'class'
        chart = functions.get_chart_ts_up(data_hist22, x, y, col, '')
        st.altair_chart(
            chart,
            use_container_width=True
        )


    st.subheader(' GOMPERTZ  : PLOT  - WITHOUT HISTORICAL DATA')
    path = 'data/DATA_TO_SHARE/GRID2917-20220922T122839Z-001/output_model/grid52/grid52_SENT1/df/'
    dfpath = os.listdir(path)
    sum_ = []
    dfs = pd.DataFrame()
    cp = 0
    for p in dfpath:
        f = pd.read_csv(path + p)
        f = f.filter(regex=('\d{4}-?\d{2}-?\d{2}$'))
        sf = f.sum(axis=0)
        # st.write(sf)
        df_ = pd.DataFrame()
        df_['area'] = list(sf.values)
        df_['time'] = list(sf.index)
        df_['class'] = p.split('.')[0]  # 'df'+str(cp)
        dfs = dfs.append(df_)
        cp += 1
    dfs['time'] = pd.to_datetime(dfs['time']).dt.date
    # st.write(dfs)
    # dfs =  .filter(regex=('\d{4}-?\d{2}-?\d{2}$'))
    # st.write(dfs)
    fig_col1_v1, fig_col2_v2 = st.columns(2)

    # dfs = dfs[dfs['class'] ]== 'df'+str(cp)

    with fig_col1_v1:
        # -------------------------------
        st.write('----------- ---- -----------------')
        # st.write(dfs)
        # ----------------------------------
        x = 'time'
        y = 'area'
        col = 'class'
        chart = functions.get_chart_ts(dfs, x, y, col, '')

        st.altair_chart(
            chart,
            use_container_width=True
        )

    ids = dfs['time'].unique()
    ids = ['df_' + str(c) for c in ids]

    num_ids = ids
    # st.write(ids)
    if len(num_ids) == 0:
        st.write('There is no grids in the data.')
    else:
        selected_num_cols = functions.multiselect_container('Choose the DF you want to visualise_:',
                                                            num_ids, 'Grid_')
        # st.subheader('Distribution of numerical columns')
        # st.subheader("### GOMPERTZ  : ")
        i = 0
        cpt = 0
        while (i < len(selected_num_cols)):
            c1, c2 = st.columns(2)
            for j in [c1, c2]:

                if (i >= len(selected_num_cols)):
                    break

                datagrid = dfs[dfs['class'] == selected_num_cols[i]].dropna()
                # st.write('yyyyy')
                # st.write(datagrid)
                x = 'time'
                y = 'area'
                col = 'class'
                chart = functions.get_chart_ts(datagrid, x, y, col, 'plot' + ' for ' + str(selected_num_cols[i]))

                j.altair_chart(
                    chart,
                    use_container_width=True
                )

                i += 1
                cpt += 1

    st.subheader(' GOMPERTZ  : PLOT  - WITH HISTORICAL DATA')
    path = 'data/DATA_TO_SHARE/GRID2917-20220922T122839Z-001/output_model/grid52/grid52_SENT1/hist/df/'
    path = 'data/DATA_TO_SHARE/GRID2917-20220922T122839Z-001/output_model/grid52/grid52_SENT1/hist/df/'
    dfpath = os.listdir(path)
    sum_ = []
    dfs = pd.DataFrame()
    cp = 0
    for p in dfpath:
        f = pd.read_csv(path + p)
        f = f.filter(regex=('\d{4}-?\d{2}-?\d{2}$'))
        sf = f.sum(axis=0)
        # st.write(sf)
        df_ = pd.DataFrame()
        df_['area'] = list(sf.values)
        df_['time'] = list(sf.index)
        df_['class'] = p.split('.')[0]  # 'df'+str(cp)
        dfs = dfs.append(df_)
        cp += 1
    dfs['time'] = pd.to_datetime(dfs['time']).dt.date
    # st.write(dfs)
    # dfs =  .filter(regex=('\d{4}-?\d{2}-?\d{2}$'))
    # st.write(dfs)
    fig_col1_v1, fig_col2_v2 = st.columns(2)

    # dfs = dfs[dfs['class'] ]== 'df'+str(cp)

    with fig_col1_v1:
        # -------------------------------
        st.write('----------- -----------------')
        # st.write(dfs)
        # ----------------------------------
        x = 'time'
        y = 'area'
        col = 'class'
        chart = functions.get_chart_ts(dfs, x, y, col, '')

        st.altair_chart(
            chart,
            use_container_width=True
        )

    ids = dfs['time'].unique()
    ids = ['df_' + str(c) for c in ids]

    num_ids = ids
    # st.write(ids)
    if len(num_ids) == 0:
        st.write('There is no grids in the data.')
    else:
        selected_num_cols = functions.multiselect_container('Choose the DF you want to visualise__:',
                                                            num_ids, '_')
        # st.subheader('Distribution of numerical columns')
        # st.subheader("### GOMPERTZ  : ")
        i = 0
        cpt = 0
        while (i < len(selected_num_cols)):
            c1, c2 = st.columns(2)
            for j in [c1, c2]:

                if (i >= len(selected_num_cols)):
                    break

                datagrid = dfs[dfs['class'] == selected_num_cols[i]].dropna()
                # st.write('yyyyy')
                # st.write(datagrid)
                x = 'time'
                y = 'area'
                col = 'class'
                chart = functions.get_chart_ts(datagrid, x, y, col, 'plot_' + ' for ' + str(selected_num_cols[i]))

                j.altair_chart(
                    chart,
                    use_container_width=True
                )

                i += 1
                cpt += 1

