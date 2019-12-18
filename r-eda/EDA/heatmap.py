import pandas as pd
import plotly.express as px
import plotly.io as pio
pio.renderers.default = "browser"
import plotly.graph_objects as go
from geopy import geocoders
import re

gn = geocoders.GeoNames(username='Zackhardtoname')

def get_loc(str):
    str = re.sub(r"(\w)([A-Z])", r"\1 \2", str)
    try:
        res = gn.geocode(str, exactly_one=False, country="US")
        if not res:
            str = str.split()[0]
            res = gn.geocode(str, exactly_one=False, country="US")
        return res[0].raw["adminCode1"]
    except:
        try:
            res = gn.geocode(str, exactly_one=False, country="US")
            if not res:
                str = str.split()[0]
                res = gn.geocode(str, exactly_one=False, country="US")
            return res[0].raw["adminCode1"]
        except:
            return None

df = pd.read_csv('./data/original.csv')
df = df[df["type"] == "organic"]
df = df.groupby('region', as_index=False)['AveragePrice'].mean()
df = df[df["region"] != "TotalUS"]
df['state_code'] = df['region'].apply(lambda str: get_loc(str))

fig = go.Figure(data=go.Choropleth(
    locations=df['state_code'],
    z=df['AveragePrice'].astype(float),
    locationmode='USA-states',
    colorscale='Reds',
    autocolorscale=False,
    text=df['region'], # hover text
    marker_line_color='white', # line markers between states
    colorbar_title="USD"
))

fig.update_layout(
    title_text='Average Price for Organic Avocados from 2015 to 2018 in Most U.S. States',
    title_x=0.5,
    geo = dict(
        scope='usa',
        projection=go.layout.geo.Projection(type = 'albers usa'),
        showlakes=True, # lakes
        lakecolor='rgb(255, 255, 255)'),
)

fig.show()

df.to_pickle("./data/prices_states.pkl")
