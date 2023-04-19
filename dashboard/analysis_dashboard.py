#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 16:54:01 2023

@author: huangrunzhe
"""

'''
Overview:
A movie analysis dashboard to provide visual charts. 
The dashboard's functionality includes filtering options, user inputs, and interactive elements. 

The metrics and visualizations we consider for this dashboard:
1. Movie rating distribution: 
    Display a histogram or box plot of movie ratings to understand the overall rating distribution in the dataset.

2. Average ratings by genre: 
    Display a bar chart showing the average rating for each genre.

3. User rating patterns: 
    Display a heat map or scatter plot showing how users rate movies, revealing trends or patterns.

4. Genre popularity over time: 
    Display a line chart showing the popularity of genres over the years based on the number of movies released or average ratings.

5. Top movies by genre & by year: 
    Display a bar chart showing the top-rated movies in each genre & by release year(can switch by a button).
'''

from data_processing import data, years, max_year, min_year, genres_list
from analysis_app import app, color_scale 
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import plotly.graph_objs as go
from dash.dependencies import Input, Output

## Creating metrics and visualizations

# Dashboard title
title = html.Div([
    html.Div([
        html.H1(children=[
            html.Strong("Movie Analysis Dashboard")],
            style={"text-align": "center"})
    ], className="col", style={"display": "flex", "justify-content": "center"})
], className="row")

# Graphs and sliders for row1
row1 = html.Div([
    html.Div([
        html.P("Annual Movie Voters' Number"),
        dcc.Graph(id="line1", style={"height": "90%", "width": "98%"}, config=dict(displayModeBar=False))
    ], className="col-6 chart_div"),
    html.Div([
        html.P("Average Annual Rating Score"),
        dcc.Graph(id="line2", style={"height": "90%", "width": "98%"}, config=dict(displayModeBar=False)),
    ], className="col-6 chart_div"),    
], className="row")

year_range_slider = html.Div([
    html.Div([
        dcc.RangeSlider(
            id='year-range-slider',
            min=min_year,
            max=max_year,
            value=[min_year, max_year],
            marks={i: f'{i}' for i in range(min_year, max_year + 1) 
                   if i % 50 == 0 or i == min_year or i == max_year})
        ], className='col-6', style={'padding': '2px', 'margin': '0px 5px 0px'})
    ], className='row justify-content-center')

# Interactive dropdown option
dropdowns = html.Div([
    html.Div([
        dcc.Dropdown(id='dropdown1',  # Year
                     options=[{'label': 'Year{}'.format(year), 'value': year} for year in sorted(years)],
                     value=None, style={'width': '90%'})
    ], className='col-4', style={'padding': '2px', 'margin': '0px 5px 0px'}),

    html.Div([
        dcc.Dropdown(id='dropdown2',  # Label
                     options=[{'label': option, 'value': option} for option in sorted(genres_list)],
                     value=None, style={'width': '90%'})
    ], className='col-4', style={'padding': '2px', 'margin': '0px 5px 0px'}),

    html.Div([
        dcc.Dropdown(id='dropdown3', # Year
                     options=[{'label': 'Year{}'.format(year), 'value': year} for year in sorted(years)],
                     value=None, style={'width': '90%'})
    ], className='col-4', style={'padding': '2px', 'margin': '0px 5px 0px'})
], className='row', style={'display': 'flex', 'justify-content': 'center', 'align-items': 'center'})

# Graphs for row2
row2 = html.Div([
    html.Div([
        html.P("Top10 Movie Ratings by Year and Genre"),
        dcc.Graph(id="bar1", style={"height": "90%", "width": "98%"}, config=dict(displayModeBar=False)),
    ], className="col-6 chart_div"),
    html.Div([
        html.P("Average Annual Movie Rating by Genre"),
        dcc.Graph(id="bar2", style={"height": "90%", "width": "98%"}, config=dict(displayModeBar=False)),
    ], className="col-6 chart_div"),
], className='row')

# Graphs for row3
row3 = html.Div([
    html.Div([
        html.P("Movie Rating Score Heat Map"),
        dcc.Graph(id="heatmap", style={"height": "90%", "width": "98%"}, config=dict(displayModeBar=False)),
    ], className="col-12 chart_div"),
    ], className='row')

# General layout
app.layout = html.Div([
    html.Div([
        title,
        row1,
        year_range_slider,
        row2,
        dropdowns,
        row3,
    ], style={'margin': '0% 30px'}),
])


## Callbacks

def create_line1_figure(df):
    df_type_visit_sum = pd.DataFrame(df['vote_count'].groupby(df['release_date']).sum())
    trace = go.Scatter(
        x=df_type_visit_sum.index,
        y=df_type_visit_sum['vote_count'],
        line=dict(color=color_scale[1])
    )
    layout = go.Layout(
        margin=dict(l=40, r=40, t=10, b=50),
        yaxis=dict(gridcolor='#e2e2e2'),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(
        tickmode='array',
        tickvals=[1874, 1900, 1950, 2000, 2020],
        ticktext=['1874', '1900', '1950', '2000', '2020'])
    )
    return go.Figure(data=[trace], layout=layout)

def create_line2_figure(df):
    df['vote_average'] = df['vote_average'].astype('float')
    df_type_visit_mean = pd.DataFrame(df['vote_average'].groupby(df['release_date']).mean())
    trace = go.Scatter(
    x=df_type_visit_mean.index,
    y=df_type_visit_mean['vote_average'],
    line=dict(color=color_scale[3])
    )
    layout = go.Layout(
    margin=dict(l=40, r=40, t=10, b=50),
    yaxis=dict(gridcolor='#e2e2e2'),
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
     xaxis=dict(
        tickmode='array',
        tickvals=[1874, 1900, 1950, 2000, 2020],
        ticktext=['1874', '1900', '1950', '2000', '2020'])
    )
    return go.Figure(data=[trace], layout=layout)

@app.callback(
    [Output('line1', 'figure'),
     Output('line2', 'figure')],
    [Input('year-range-slider', 'value')])
def update_line_charts(year_range):
    start_year, end_year = year_range

    # Filter data based on the selected year range
    filtered_data = data[(pd.to_datetime(data['release_date']).dt.year >= start_year) 
                         & (pd.to_datetime(data['release_date']).dt.year <= end_year)]
    # Create the figures for the two line charts
    line1_figure = create_line1_figure(filtered_data)
    line2_figure = create_line2_figure(filtered_data)
    return line1_figure, line2_figure

# Top 10 movie and ratings by year and genre
@app.callback(
    Output('bar1', 'figure'),
    [Input('dropdown1', 'value'),
     Input('dropdown2', 'value')])
def get_bar1(value1, value2):
    if value1 != None and value2 != None:
        dff = data
        df = dff.groupby('release_date').get_group(value1)
        if value2 not in df['genres_split'].unique():
            trace = go.Bar()
            layout = go.Layout(title="No data for the selected genre",
                               margin=dict(l=40, r=40, t=40, b=50),
                               paper_bgcolor='rgba(0, 0, 0, 0)',
                               plot_bgcolor='rgba(0, 0, 0, 0)')
        else:
            df_genre = df.groupby('genres_split').get_group(value2)
            df_genre = df_genre.sort_values(by='vote_average', ascending=False)
            trace = go.Bar(
                x=df_genre['title'][0:9],
                y=df_genre['vote_average'][0:9],
                text=df_genre['vote_average'][0:9],
                textposition='auto',
                marker=dict(color='#33ffe6'))
            layout = go.Layout(
                margin=dict(l=40, r=40, t=10, b=50),
                yaxis=dict(gridcolor='#e2e2e2'),
                paper_bgcolor='rgba(0, 0, 0, 0)',
                plot_bgcolor='rgba(0, 0, 0, 0)')
    else:
        trace = go.Bar()
        layout = go.Layout(title="Please select the year or genre",
                           margin=dict(l=40, r=40, t=40, b=50),
                           paper_bgcolor='rgba(0, 0, 0, 0)',
                           plot_bgcolor='rgba(0, 0, 0, 0)')  
    return go.Figure(data=[trace], layout=layout)

# By year - Average score for each genre
@app.callback(
    Output('bar2','figure'), 
    [Input("dropdown3", "value")])
def get_bar2(value):
    if value == None:
        trace = go.Bar()
        layout = go.Layout(title="Please select the year",
                           margin=dict(l=40, r=40, t=40, b=50),
                           paper_bgcolor='rgba(0, 0, 0, 0)',
                           plot_bgcolor='rgba(0, 0, 0, 0)')  
    else:
        dff = data
        df=dff.groupby('release_date').get_group(value)
        df['vote_average'] = df['vote_average'].astype('float')
        df_type_visit_mean = pd.DataFrame(df['vote_average'].groupby(df['genres_split']).agg('mean').round(2))
        trace = go.Bar(
            x=df_type_visit_mean.index,
            y=df_type_visit_mean['vote_average'],
            text=df_type_visit_mean['vote_average'],
            textposition='auto',
            marker=dict(color='#33ffe6'))
        layout = go.Layout(
            margin=dict(l=40, r=40, t=10, b=50),
            yaxis=dict(gridcolor='#e2e2e2'),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)')
    return go.Figure(data=[trace], layout=layout)

# By year - Heat map
@app.callback(
    Output('heatmap','figure'),
    [Input("dropdown3", "value")])
def get_heatmap(value):
    if value == None:
        trace = go.Bar()
        layout = go.Layout(title="Please select the year",
                           margin=dict(l=40, r=40, t=40, b=50),
                           paper_bgcolor='rgba(0, 0, 0, 0)',
                           plot_bgcolor='rgba(0, 0, 0, 0)')  
    else:
        dff=data
        df=dff.groupby('release_date').get_group(value)
        cross = pd.crosstab(df['scale'],df['genres_split'])
        cross.sort_index(inplace=True)
        trace = go.Heatmap(
            x=cross.columns,
            y=cross.index,
            z=cross.values,
            colorscale="Greens",
            reversescale=False,
            xgap=4,
            ygap=5,
            showscale=False)
        layout = go.Layout(
            margin=dict(l=50, r=40, t=30, b=50))
    return go.Figure(data=[trace], layout=layout)

## Run dashboard app
app.run_server(debug=True,threaded=True)

