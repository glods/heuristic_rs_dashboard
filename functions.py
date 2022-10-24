
import io
import pandas as pd
import streamlit as st
import altair as alt



def df_info(df):
    df.columns = df.columns.str.replace(' ', '_')
    buffer = io.StringIO()
    df.info(buf=buffer)
    s = buffer.getvalue()

    df_info = s.split('\n')

    counts = []
    names = []
    nn_count = []
    dtype = []
    for i in range(5, len(df_info) - 3):
        line = df_info[i].split()
        counts.append(line[0])
        names.append(line[1])
        nn_count.append(line[2])
        dtype.append(line[4])

    df_info_dataframe = pd.DataFrame(
        data={'#': counts, 'Column': names, 'Non-Null Count': nn_count, 'Data Type': dtype})
    return df_info_dataframe.drop('#', axis=1)


def df_isnull(df):
    res = pd.DataFrame(df.isnull().sum()).reset_index()
    res['Percentage'] = round(res[0] / df.shape[0] * 100, 2)
    res['Percentage'] = res['Percentage'].astype(str) + '%'
    return res.rename(columns={'index': 'Column', 0: 'Number of null values'})


def number_of_outliers(df):
    df = df.select_dtypes(exclude='object')

    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1

    ans = ((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).sum()
    df = pd.DataFrame(ans).reset_index().rename(columns={'index': 'column', 0: 'count_of_outliers'})
    return df


def space(num_lines=1):
    for _ in range(num_lines):
        st.write("")


def sidebar_space(num_lines=1):
    for _ in range(num_lines):
        st.sidebar.write("")


def sidebar_multiselect_container(massage, arr, key):
    container = st.sidebar.container()
    select_all_button = st.sidebar.checkbox("Select all for " + key + " plots")
    if select_all_button:
        selected_num_cols = container.multiselect(massage, arr, default=list(arr))
    else:
        selected_num_cols = container.multiselect(massage, arr, default=arr[0])

    return selected_num_cols


def multiselect_container(massage, arr, key):
    container = st.container()
    select_all_button = st.checkbox("Select all for " + key + " plots")
    if select_all_button:
        selected_num_cols = container.multiselect(massage, arr, default=list(arr))
    else:
        selected_num_cols = container.multiselect(massage, arr, default=arr[0])

    return selected_num_cols

from locale import normalize

# Define the base time-series chart.
def get_chart_ts(data, x, y, col , title):
    # data[x]=pd.to_datetime(data[x])
    hover = alt.selection_single(
        fields=[x],
        nearest=True,
        on="mouseover",
        empty="none",
    )

    lines = (
        alt.Chart(data, title=title)
        .mark_line()
        .encode(
            x=x,
            y=y,
            color=col,
        )
    )

    # Draw points on the line, and highlight based on selection
    points = lines.transform_filter(hover).mark_circle(size=65)

    # Draw a rule at the location of the selection
    tooltips = (
        alt.Chart(data)
        .mark_rule()
        .encode(
            x=x,
            y=y,
            opacity=alt.condition(hover, alt.value(0.3), alt.value(0)),
            tooltip=[
                alt.Tooltip(x, title="Date"),
                alt.Tooltip(y, title="Flooding area (ha)"),
                alt.Tooltip(col, title="Name"),
            ],
        )
        .add_selection(hover)
    )

    return (lines + points + tooltips).interactive()



def get_chart_ts_up(data, x, y, col , title):
    # data[x]=pd.to_datetime(data[x])
    hover = alt.selection_single(
        fields=[x],
        nearest=True,
        on="mouseover",
        empty="none",
    )
    lines = (
        alt.Chart(data, title=title)
        .mark_line()
        .encode(
            x=x,
            y=y,
            color=col,
        )
    )
    # Draw points on the line, and highlight based on selection
    # points = lines.transform_filter(hover).mark_circle(size=65)
    points = (
        alt.Chart(data, title=title)
        .mark_point(filled=True, opacity=1).encode(
            x=x,
            y=y,
            color=col,
        )
    )
    # Draw a rule at the location of the selection
    tooltips = (
        alt.Chart(data)
        .mark_rule()
        .encode(
            x=x,
            y=y,
            opacity=alt.condition(hover, alt.value(0.3), alt.value(0)),
            tooltip=[
                alt.Tooltip(x, title="Date"),
                alt.Tooltip(y, title="Flooding area (ha)"),
                alt.Tooltip(col, title="Name"),
            ],
        )
        .add_selection(hover)
    )
    return (lines + points + tooltips).interactive()



def chart_ts_update():
    # Add annotations
    ANNOTATIONS = [
        ("Jan 01, 2008", "Early flooding"),
        # ("Fev 01, 2007", ""),
        # ("Mar 01, 2008", "Market starts again thanks to..."),
        ("May 01, 2009", "Late flooding"),
    ]
    annotations_df = pd.DataFrame(ANNOTATIONS, columns=["date", "event"])
    annotations_df.date = pd.to_datetime(annotations_df.date).dt.month
    annotations_df["y"] = 10

    annotation_layer = (
        alt.Chart(annotations_df)
        .mark_text(size=20, text="⬇", dx=-8, dy=-10, align="left")
        .encode(
            x="date:T",
            y=alt.Y("y:Q"),
            tooltip=["event"],
        )
        .interactive()
    )

    return annotation_layer

# chart = get_chart_ts(source)
