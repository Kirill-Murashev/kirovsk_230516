import pandas as pd
import folium
from folium.plugins import HeatMap, MarkerCluster

# Read and preprocess data
file_path = 'kirovsk_230516.csv'
kir = pd.read_csv(file_path, sep=';')
print(kir)

# Unit price
# Create specific sub dataframe
df = kir[['latitude', 'longitude', 'unit_price']]
print(df)

# Create a folium map centered on the average latitude and longitude
center_lat = df['latitude'].mean()
center_lon = df['longitude'].mean()
m = folium.Map(location=[center_lat, center_lon], zoom_start=12)

# Create a MarkerCluster layer for the overlapping markers
marker_cluster = MarkerCluster().add_to(m)

# Create individual markers for each object and add them to the MarkerCluster layer
for index, row in df.iterrows():
    popup_text = f"Unit price: {row['unit_price']}"
    folium.Marker(location=[row['latitude'], row['longitude']], popup=folium.Popup(popup_text)).add_to(marker_cluster)

# Create a HeatMap layer using the price data
heat_data = [[row['latitude'], row['longitude'], row['unit_price']] for index, row in df.iterrows()]
HeatMap(heat_data, gradient={0.2: 'blue', 0.4: 'green', 0.6: 'yellow', 0.8: 'orange', 1: 'red'}).add_to(m)

# Create a color scale legend
colormap = folium.LinearColormap(colors=['blue', 'green', 'yellow', 'orange', 'red'], vmin=df['unit_price'].min(),
                                 vmax=df['unit_price'].max())
colormap.caption = 'Unit price'
m.add_child(colormap)

# Save the map as an HTML file
m.save('heatmap_unit_price.html')

# Discount
# Create specific sub dataframe
df = kir[['latitude', 'longitude', 'total_discount']]
df['total_discount'] = 1 - df['total_discount']
print(df)

# Create a folium map centered on the average latitude and longitude
center_lat = df['latitude'].mean()
center_lon = df['longitude'].mean()
m = folium.Map(location=[center_lat, center_lon], zoom_start=12)

# Create a MarkerCluster layer for the overlapping markers
marker_cluster = MarkerCluster().add_to(m)

# Create individual markers for each object and add them to the MarkerCluster layer
for index, row in df.iterrows():
    popup_text = f"Expected discount: {row['total_discount']}"
    folium.Marker(location=[row['latitude'], row['longitude']], popup=folium.Popup(popup_text)).add_to(marker_cluster)

# Create a HeatMap layer using the price data
heat_data = [[row['latitude'], row['longitude'], row['total_discount']] for index, row in df.iterrows()]
HeatMap(heat_data, gradient={0.2: 'blue', 0.4: 'green', 0.6: 'yellow', 0.8: 'orange', 1: 'red'}).add_to(m)

# Create a color scale legend
colormap = folium.LinearColormap(colors=['blue', 'green', 'yellow', 'orange', 'red'], vmin=df['total_discount'].min(),
                                 vmax=df['total_discount'].max())
colormap.caption = 'Expected discount'
m.add_child(colormap)

# Save the map as an HTML file
m.save('heatmap_total_discount.html')

# Age
# Create specific sub dataframe
df = kir[['latitude', 'longitude', 'age']]
print(df)

# Create a folium map centered on the average latitude and longitude
center_lat = df['latitude'].mean()
center_lon = df['longitude'].mean()
m = folium.Map(location=[center_lat, center_lon], zoom_start=12)

# Create a MarkerCluster layer for the overlapping markers
marker_cluster = MarkerCluster().add_to(m)

# Create individual markers for each object and add them to the MarkerCluster layer
for index, row in df.iterrows():
    popup_text = f"Age: {row['age']}"
    folium.Marker(location=[row['latitude'], row['longitude']], popup=folium.Popup(popup_text)).add_to(marker_cluster)

# Create a HeatMap layer using the price data
heat_data = [[row['latitude'], row['longitude'], row['age']] for index, row in df.iterrows()]
HeatMap(heat_data, gradient={0.2: 'blue', 0.4: 'green', 0.6: 'yellow', 0.8: 'orange', 1: 'red'}).add_to(m)

# Create a color scale legend
colormap = folium.LinearColormap(colors=['blue', 'green', 'yellow', 'orange', 'red'], vmin=df['age'].min(),
                                 vmax=df['age'].max())
colormap.caption = 'Age'
m.add_child(colormap)

# Save the map as an HTML file
m.save('heatmap_age.html')

# Floors
# Create specific sub dataframe
df = kir[['latitude', 'longitude', 'floors']]
print(df)

# Create a folium map centered on the average latitude and longitude
center_lat = df['latitude'].mean()
center_lon = df['longitude'].mean()
m = folium.Map(location=[center_lat, center_lon], zoom_start=12)

# Create a MarkerCluster layer for the overlapping markers
marker_cluster = MarkerCluster().add_to(m)

# Create individual markers for each object and add them to the MarkerCluster layer
for index, row in df.iterrows():
    popup_text = f"Number of floors: {row['floors']}"
    folium.Marker(location=[row['latitude'], row['longitude']], popup=folium.Popup(popup_text)).add_to(marker_cluster)

# Create a HeatMap layer using the price data
heat_data = [[row['latitude'], row['longitude'], row['floors']] for index, row in df.iterrows()]
HeatMap(heat_data, gradient={0.2: 'blue', 0.4: 'green', 0.6: 'yellow', 0.8: 'orange', 1: 'red'}).add_to(m)

# Create a color scale legend
colormap = folium.LinearColormap(colors=['blue', 'green', 'yellow', 'orange', 'red'], vmin=df['floors'].min(),
                                 vmax=df['floors'].max())
colormap.caption = 'Number of floors'
m.add_child(colormap)

# Save the map as an HTML file
m.save('heatmap_floors.html')