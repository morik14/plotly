from datetime import datetime, timedelta

from dash import Dash, dcc, html, Input, Output, callback
import dash_bootstrap_components as dbc
import dash_ag_grid as dag
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd

pio.templates.default = "gridon"

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])


def calc_sma(df, last_stock, n_days, periods):
    cumsums = df['repair_cumsum'].tail(n_days).to_numpy()
    ave = (cumsums[-1] - cumsums[0]) / n_days

    last_date = df.iloc[-1]['date']
    last_cumsum = df.iloc[-1]['repair_cumsum']

    date = pd.date_range(start=last_date + timedelta(days=1), end=last_date + timedelta(days=periods))
    repair_pred = [last_cumsum + i*ave for i in range(1, periods+1)]
    stock_pred = [last_stock - i*ave for i in range(1, periods+1)]
    data = [date, repair_pred, stock_pred]

    return pd.DataFrame(data, index=["date", "repair_cumsum", "stock"]).T


def generate_table():
    df_table = pd.read_csv('data/table.csv')

    columnDefs = [
        {
            "field": "carrier",
            "checkboxSelection": True,
        },
        {"field": "parts_cd"},
    ]

    defaultColDef = {
        "flex": 1,
        "minWidth": 100,
        "sortable": True, 
        "filter": True,
        "floatingFilter": True
    }

    return dag.AgGrid(
        id="datatable-interactivity",
        columnDefs=columnDefs,
        rowData=df_table.to_dict("records"),
        dashGridOptions={"rowSelection": "multiple"},
        columnSize="sizeToFit",
        defaultColDef=defaultColDef
    )
    

layout = dbc.Container(
    fluid=True,
    className="p-5",
    children=[
        generate_table(),
        html.Hr(),
        html.Div(id="datatable-interactivity-container")
    ]
)


layout2 = dbc.Container(
    fluid=True,
    className="p-5",
    children=[
        generate_table(),
        html.Hr(),
        
        dbc.Card(
            body=True,
            children=[
                dbc.Row(
                    children=[
                        dbc.Col(
                            children=[
                                dbc.Label("Shared x:"),
                                dcc.RadioItems(
                                    id='sharedx', value=False, inline=True,
                                    options=[{'label': 'on', 'value': True}, 
                                            {'label': 'off', 'value': False}],
                                ),
                            ]
                        ),
                        dbc.Col(
                            children=[
                                dbc.Label("Show pred:"),
                                dcc.RadioItems(
                                    id='show-pred', value=False, inline=True,
                                    options=[{'label': 'on', 'value': True}, 
                                            {'label': 'off', 'value': False}],
                                ),
                            ]
                        ),
                    ]
                ),
            ]
        ),
        
        html.Div(id="datatable-interactivity-container")
    ]
)


app.layout = layout2


@callback(
    Output("datatable-interactivity-container", "children"),
    Input("datatable-interactivity", "selectedRows"),
    Input("sharedx", "value"),
    Input("show-pred", "value"),
)
def update_graphs(selected, sharedx, show_pred):
    if not selected:
        return 'No data selected'

    df_table = pd.read_csv('data/table.csv')
    df_repair = pd.read_csv('data/repair_cumsum.csv')
    df_stock = pd.read_csv('data/stock.csv')
    df_repair['date'] = pd.to_datetime(df_repair['date'])
    df_stock['date'] = pd.to_datetime(df_stock['date'])

    columnDefs = [
        {"field": "key"},
        {"field": "value"}
        
    ]

    defaultColDef = {
        "flex": 1,
    }

    print(selected)
    graphs = []
    if sharedx:
        fig = make_subplots(rows=len(selected), cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.02)
        for i, s in enumerate(selected):
            carrier = s['carrier']
            parts_cd = s['parts_cd']
            sdf_table = df_table[(df_table['carrier'] == carrier) & (df_table['parts_cd'] == parts_cd)]
            sdf_repair = df_repair[(df_repair['carrier'] == carrier) & (df_repair['parts_cd'] == parts_cd)]
            sdf_stock = df_stock[(df_stock['carrier'] == carrier) & (df_stock['parts_cd'] == parts_cd)]
            last_date = sdf_stock.iloc[-1]['date']
            last_stock = sdf_stock.iloc[-1]['stock']

            

            fig.add_trace(
                go.Bar(
                    x=sdf_repair['date'],
                    y=sdf_repair['repair_cumsum'],
                    name='repair(actual)',
                    marker_color='rgb(220, 57, 18)',
                ), row=i+1, col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=sdf_stock['date'],
                    y=sdf_stock['stock'],
                    name='stock(actual)',
                    mode='none',
                    line_shape = 'hvh',
                    fill='tozeroy',
                    fillcolor='rgba(16, 150, 24, 0.6)',
                    opacity=0.5
                ), row=i+1, col=1
            )

            if show_pred:
                df_pred = calc_sma(sdf_repair, last_stock, n_days=2, periods=5)

                fig.add_trace(
                    go.Bar(
                        x=df_pred['date'],
                        y=df_pred['repair_cumsum'],
                        name='repair(pred)',
                        marker_color='rgb(254, 203, 82)',
                    ), row=i+1, col=1
                )
                fig.add_trace(
                    go.Scatter(
                        x=df_pred['date'],
                        y=df_pred['stock'],
                        name='stock(pred)',
                        mode='none',
                        line_shape = 'hvh',
                        fill='tozeroy',
                        fillcolor='rgba(171, 99, 250, 0.6)',
                        opacity=0.5
                    ), row=i+1, col=1
                )

            # 個々のグラフ設定
            fig.update_yaxes(
                title=f'{carrier}<br>{parts_cd}',
                row=i+1, col=1
            )
            fig.update_xaxes(
                dtick="M1",
                tickformat="%Y\n%m\n%d",
                ticklabelmode="period",
                row=i+1, col=1
            )
            fig.add_annotation(
                x=last_date,
                y=last_stock,
                text=str(last_stock),
                row=i+1, col=1
            )
        
        fig.update_xaxes(
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(step="all")
                ])
            ), row=1, col=1
        )
        fig.update_xaxes(
            dtick="M1",
            tickformat="%m\n%Y",
            ticklabelmode="period",
            showspikes=True,
            spikesnap="cursor", 
            spikemode="across",
            minor=dict(ticks="inside", showgrid=True)
        )
        fig.update_layout(
            showlegend=False,
            height=200*(len(selected)+1), 
            hovermode="x unified",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                bgcolor='rgba(0,0,0,0)',
            )
        )
        return dcc.Graph(figure=fig),
    else:    
        for i, s in enumerate(selected):
            carrier = s['carrier']
            parts_cd = s['parts_cd']
            sdf_table = df_table[(df_table['carrier'] == carrier) & (df_table['parts_cd'] == parts_cd)]
            sdf_repair = df_repair[(df_repair['carrier'] == carrier) & (df_repair['parts_cd'] == parts_cd)]
            sdf_stock = df_stock[(df_stock['carrier'] == carrier) & (df_stock['parts_cd'] == parts_cd)]
            last_stock = sdf_stock.iloc[-1]['stock']

            df_pred = calc_sma(sdf_repair, last_stock, n_days=2, periods=5)

            fig = go.Figure()
            fig.add_trace(
                go.Bar(
                    x=sdf_repair['date'],
                    y=sdf_repair['repair_cumsum'],
                    name='repair(actual)',
                    marker_color='rgb(220, 57, 18)',
                )
            )
            fig.add_trace(
                go.Bar(
                    x=df_pred['date'],
                    y=df_pred['repair_cumsum'],
                    name='repair(pred)',
                    marker_color='rgb(254, 203, 82)',
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=sdf_stock['date'],
                    y=sdf_stock['stock'],
                    name='stock(actual)',
                    mode='none',
                    line_shape = 'hvh',
                    fill='tozeroy',
                    fillcolor='rgba(16, 150, 24, 0.6)',
                    opacity=0.5
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=df_pred['date'],
                    y=df_pred['stock'],
                    name='stock(pred)',
                    mode='none',
                    line_shape = 'hvh',
                    fill='tozeroy',
                    fillcolor='rgba(171, 99, 250, 0.6)',
                    opacity=0.5
                )
            )
            fig.update_xaxes(
                dtick="M1",
                tickformat="%Y-%m-%d",
                ticklabelmode="period",
                rangeselector=dict(
                    buttons=list([
                        dict(count=1, label="1m", step="month", stepmode="backward"),
                        dict(count=6, label="6m", step="month", stepmode="backward"),
                        dict(count=1, label="1y", step="year", stepmode="backward"),
                        dict(step="all")
                    ])
                )
            )
            fig.update_layout(
                hovermode="x unified",
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1,
                    bgcolor='rgba(0,0,0,0)',
                )
            )

            graph = dcc.Graph(figure=fig)

            tdf = sdf_table.T.reset_index()
            tdf.columns = ['key', 'value']
            table = dag.AgGrid(
                columnDefs=columnDefs,
                rowData=tdf.to_dict("records"),
                dashGridOptions={"rowSelection": "multiple", "headerHeight":0},
                defaultColDef=defaultColDef
            )

            graphs.append(
                html.Div(
                    children=[
                        dbc.Row(
                            className="mb-4",
                            align='center',
                            children=[
                                dbc.Col(
                                    width=3,
                                    className="h-100",
                                    children=table
                                ),
                                dbc.Col(
                                    width=9,
                                    className="h-100",
                                    children=graph
                                ),
                            ]
                        )
                    ]
                )
            )

    return graphs


if __name__ == '__main__':
    app.run(debug=True)
