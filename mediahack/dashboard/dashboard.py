import io
import base64

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State

import pandas as pd
import plotly.express as px

# load data
dash_data = pd.read_csv('data/dashboard_data.csv')
train_segments = pd.read('data/train_segments.csv')
segment_dict = pd.read_excel('data/segment_dict.xlsx')

dash_data['Date'] = pd.to_datetime(dash_data[['Year', 'Month']].assign(DAY=1))
print(dash_data['Date'])

avg_cost_by_ad = dash_data.groupby('Advertisement ID')['Estimated cost RUB'].sum()
advertisers = dash_data[['Advertisement ID', 'Advertiser']].drop_duplicates()

df = train_segments.merge(avg_cost_by_ad, on='Advertisement ID').merge(advertisers, on='Advertisement ID')
df[df['Segment_num'] == 6] = 0
df = df.merge(segment_dict, on='Segment_num')
print(df)

# preprocess data
avg_cost_by_segment = df.groupby('Segment')['Estimated cost RUB'].mean().reset_index()
avg_ads_by_segment = df.groupby('Segment').size().reset_index(name='avg_ads')
media_type_dist_by_segment = df.groupby(['Segment', 'Advertiser']).size().unstack().fillna(0)
popular_advertisers = df['Advertiser'].value_counts().reset_index()
popular_advertisers.columns = ['Advertiser', 'Count']
advertisers_rank_by_cost = df.groupby('Advertiser')['Estimated cost RUB'].sum().sort_values(ascending=False).reset_index()

# create figures
fig1 = px.bar(avg_cost_by_segment, x='Segment', y='Estimated cost RUB', title="Average Ad Cost by Segment")
fig2 = px.bar(avg_ads_by_segment, x='Segment', y='avg_ads', title="Average Number of Ads by Segment")
fig3 = px.bar(media_type_dist_by_segment, barmode='stack', title="Media Companies Distribution by Segment")
fig4 = px.bar(popular_advertisers, x='Advertiser', y='Count', title="Most Popular Advertisers")
fig5 = px.bar(advertisers_rank_by_cost, x='Advertiser', y='Estimated cost RUB', title="Advertisers Ranked By Cost")

fig1.update_layout(
    plot_bgcolor='white',
    paper_bgcolor='LightSteelBlue',
)
fig2.update_layout(
    plot_bgcolor='white',
    paper_bgcolor='LightSteelBlue',
)
fig3.update_layout(
    plot_bgcolor='white',
    paper_bgcolor='LightSteelBlue',
)
fig4.update_layout(
    plot_bgcolor='white',
    paper_bgcolor='LightSteelBlue',
)
fig5.update_layout(
    plot_bgcolor='white',
    paper_bgcolor='LightSteelBlue',
)

min_date = dash_data['Date'].min()
max_date = dash_data['Date'].max()

app = dash.Dash(__name__)

app.layout = html.Div(children=[
    html.H1(children='Advertisement Dashboard', style={
            'textAlign': 'center',
            'color': '#7FDBFF',
            'padding': '1em 0',  
            'background': '#111111', 
            'fontFamily': 'Arial, Helvetica, sans-serif'
        }
    ),

    html.Div(children='''
        An overview of advertisement performance.
    ''', style={
        'textAlign': 'center',
        'color': '#111111',
        'padding': '1em', 
        'margin': '0 0 2em 0',
        'background': '#7FDBFF', 
        'fontFamily': 'Arial, Helvetica, sans-serif'
    }),

    dcc.Upload(
        id='upload-data',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select a .csv File')
        ]),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px',
            'cursor': 'pointer'
        },
        multiple=False  
    ),
    html.Div(id='output-data-upload'),


    html.Label('Date Range',  style={
            'textAlign': 'center',
            'color': '#111111',
            'fontFamily': 'Arial, Helvetica, sans-serif',
            'fontSize': '20px',
        }
    ),
    dcc.RangeSlider(
        id='date-range-slider',
        min=min_date.timestamp(),  
        max=max_date.timestamp(),
        value=[min_date.timestamp(), max_date.timestamp()],
        marks = {i: '{}'.format(pd.to_datetime(i, unit='s').strftime('%Y-%m')) 
         for i in range(int(min_date.timestamp()), int(max_date.timestamp()), 60*60*24*60)},
        step=None
    ),


    dcc.Graph(
        id='avg-cost-by-segment-graph',
        figure=fig1
    ),
    
    dcc.Graph(
        id='avg-ads-by-segment-graph',
        figure=fig2
    ),
    
    dcc.Graph(
        id='media-type-dist-by-segment-graph',
        figure=fig3
    ),

    dcc.Graph(
        id='most-popular-advertisers',
        figure=fig4
    ),

    dcc.Graph(
        id='advertisers-rank-by-cost',
        figure=fig5
    )
])


@app.callback(Output('output-data-upload', 'children'),
              Input('upload-data', 'contents'),
              State('upload-data', 'filename'))
def update_output(contents, filename):
    if contents is not None:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        try:
            if 'csv' in filename:
                dash_data = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        except Exception as e:
            print(e)
            return html.Div([
                'There was an error processing this file.'
            ])


@app.callback(
    [Output('avg-cost-by-segment-graph', 'figure'),
     Output('avg-ads-by-segment-graph', 'figure'),
     Output('media-type-dist-by-segment-graph', 'figure'),
     Output('most-popular-advertisers', 'figure'),
     Output('advertisers-rank-by-cost', 'figure')],
    [Input('date-range-slider', 'value')])
def update_graphs(value):
    # Convert timestamp to datetime
    start_date = pd.to_datetime(value[0], unit='s')
    end_date = pd.to_datetime(value[1], unit='s')

    # Filter df with `date` column in the date range
    filtered_dash_data = dash_data[(dash_data['Date'] >= start_date) & (dash_data['Date'] <= end_date)]

    filtered_avg_cost_by_ad = filtered_dash_data.groupby('Advertisement ID')['Estimated cost RUB'].sum()
    filtered_advertisers = filtered_dash_data[['Advertisement ID', 'Advertiser']].drop_duplicates()

    filtered_df = train_segments.merge(filtered_avg_cost_by_ad, on='Advertisement ID').merge(filtered_advertisers, on='Advertisement ID')
    filtered_df[filtered_df['Segment_num'] == 6] = 0
    filtered_df = filtered_df.merge(segment_dict, on='Segment_num')

    popular_advertisers = df['Advertiser'].value_counts().reset_index()
    popular_advertisers.columns = ['Advertiser', 'Count']

    # Recreate all the figures with the filtered data
    fig1 = px.bar(filtered_df.groupby('Segment')['Estimated cost RUB'].mean().reset_index(), x='Segment', y='Estimated cost RUB')
    fig2 = px.bar(filtered_df.groupby('Segment').size().reset_index(name='avg_ads'), x='Segment', y='avg_ads')
    fig3 = px.bar(filtered_df.groupby(['Segment', 'Advertiser']).size().unstack().fillna(0), barmode='stack')
    fig4 = px.bar(popular_advertisers, x='Advertiser', y='Count', title="Most Popular Advertisers")
    fig5 = px.bar(filtered_df.groupby('Advertiser')['Estimated cost RUB'].sum().sort_values(ascending=False).reset_index(), x='Advertiser', y='Estimated cost RUB', title="Advertisers Ranked By Cost")

    return [fig1, fig2, fig3, fig4, fig5] 


if __name__ == '__main__':
    app.run_server(debug=True)