# Import libraries
import pandas as pd
import folium
from clustering import hierarchical_clustering, kmeans_clustering, spectral_clustering_auto
import random


# jitter function: adds small random number to coordinate
def add_jitter(coord, amount=0.001):
    return coord + amount * (random.random() - 0.5)


# Import data
market = pd.read_csv('market.csv')
print(market)
print(market.columns)

# Perform hierarchical clustering
market = hierarchical_clustering(market, ['is_isolated', 'is_first', 'is_chattels', 'is_nice_view',
                                 'is_view_to_both', 'rooms', 'walls_type', 'finishing_type',
                                 'is_elevators', 'epoch', 'square_PC_1'])
print(market)





cluster_labels = 'hcl_cluster'

# Let's create a color map
colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred',
          'lightred', 'beige', 'darkblue', 'darkgreen', 'cadetblue',
          'darkpurple', 'white', 'pink', 'lightblue', 'lightgreen',
          'gray', 'black', 'lightgray']

# Create a map
m = folium.Map(location=[market['latitude'].mean(), market['longitude'].mean()], zoom_start=15)

# Add points to the map with jitter
for idx, row in market.iterrows():
    folium.CircleMarker([add_jitter(row['latitude']), add_jitter(row['longitude'])],
                        radius=5,
                        color=colors[row[cluster_labels] % len(colors)],  # Use color corresponding to cluster
                        fill=True,
                        fill_opacity=0.6
                        ).add_to(m)

# Display the map
m.save('/home/kaarlahti/PycharmProjects/kirovsk_230516/maps/h_cluster_map.html')

# Perform k-means clustering
market = kmeans_clustering(market, ['is_isolated', 'is_first', 'is_chattels', 'is_nice_view',
                                 'is_view_to_both', 'rooms', 'walls_type', 'finishing_type',
                                 'is_elevators', 'epoch', 'square_PC_1'])
print(market)

cluster_labels = 'kcl_cluster'

# Let's create a color map
colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred',
          'lightred', 'beige', 'darkblue', 'darkgreen', 'cadetblue',
          'darkpurple', 'white', 'pink', 'lightblue', 'lightgreen',
          'gray', 'black', 'lightgray']

# Create a map
m = folium.Map(location=[market['latitude'].mean(), market['longitude'].mean()], zoom_start=15)

# Add points to the map with jitter
for idx, row in market.iterrows():
    folium.CircleMarker([add_jitter(row['latitude']), add_jitter(row['longitude'])],
                        radius=5,
                        color=colors[row[cluster_labels] % len(colors)],  # Use color corresponding to cluster
                        fill=True,
                        fill_opacity=0.6
                        ).add_to(m)

# Display the map
m.save('/home/kaarlahti/PycharmProjects/kirovsk_230516/maps/km_cluster_map.html')

# # Perform DBSCAN clustering
# market = dbscan_clustering(market, ['is_isolated', 'is_first', 'is_chattels', 'is_nice_view',
#                                  'is_view_to_both', 'rooms', 'walls_type', 'finishing_type',
#                                  'is_elevators', 'epoch', 'square_PC_1'])
# print(market)
#
# cluster_labels = 'dbscan_cluster'
#
# # Let's create a color map
# colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred',
#           'lightred', 'beige', 'darkblue', 'darkgreen', 'cadetblue',
#           'darkpurple', 'white', 'pink', 'lightblue', 'lightgreen',
#           'gray', 'black', 'lightgray']
#
# # Create a map
# m = folium.Map(location=[market['latitude'].mean(), market['longitude'].mean()], zoom_start=15)
#
# # Add points to the map with jitter
# for idx, row in market.iterrows():
#     folium.CircleMarker([add_jitter(row['latitude']), add_jitter(row['longitude'])],
#                         radius=5,
#                         color=colors[row[cluster_labels] % len(colors)],  # Use color corresponding to cluster
#                         fill=True,
#                         fill_opacity=0.6
#                         ).add_to(m)
#
# # Display the map
# m.save('/home/kaarlahti/PycharmProjects/kirovsk_230516/maps/dbscan_cluster_map.html')

# # Perform meanshift clustering
# market = meanshift_clustering(market, ['is_isolated', 'is_first', 'is_chattels', 'is_nice_view',
#                                  'is_view_to_both', 'rooms', 'walls_type', 'finishing_type',
#                                  'is_elevators', 'epoch', 'square_PC_1'])
# print(market)
#
# cluster_labels = 'ms_cluster'
#
# # Let's create a color map
# colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred',
#           'lightred', 'beige', 'darkblue', 'darkgreen', 'cadetblue',
#           'darkpurple', 'white', 'pink', 'lightblue', 'lightgreen',
#           'gray', 'black', 'lightgray']
#
# # Create a map
# m = folium.Map(location=[market['latitude'].mean(), market['longitude'].mean()], zoom_start=15)
#
# # Add points to the map with jitter
# for idx, row in market.iterrows():
#     folium.CircleMarker([add_jitter(row['latitude']), add_jitter(row['longitude'])],
#                         radius=5,
#                         color=colors[row[cluster_labels] % len(colors)],  # Use color corresponding to cluster
#                         fill=True,
#                         fill_opacity=0.6
#                         ).add_to(m)
#
# # Display the map
# m.save('/home/kaarlahti/PycharmProjects/kirovsk_230516/maps/ms_cluster_map.html')

# # Perform spectral clustering
# market = spectral_clustering_auto(market, ['is_isolated', 'is_first', 'is_chattels', 'is_nice_view',
#                                  'is_view_to_both', 'rooms', 'walls_type', 'finishing_type',
#                                  'is_elevators', 'epoch', 'square_PC_1'], max_clusters=10)
# print(market)
#
# cluster_labels = 's_cluster'
#
# # Let's create a color map
# colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred',
#           'lightred', 'beige', 'darkblue', 'darkgreen', 'cadetblue',
#           'darkpurple', 'white', 'pink', 'lightblue', 'lightgreen',
#           'gray', 'black', 'lightgray']
#
# # Create a map
# m = folium.Map(location=[market['latitude'].mean(), market['longitude'].mean()], zoom_start=15)
#
# # Add points to the map with jitter
# for idx, row in market.iterrows():
#     folium.CircleMarker([add_jitter(row['latitude']), add_jitter(row['longitude'])],
#                         radius=5,
#                         color=colors[row[cluster_labels] % len(colors)],  # Use color corresponding to cluster
#                         fill=True,
#                         fill_opacity=0.6
#                         ).add_to(m)
#
# # Display the map
# m.save('/home/kaarlahti/PycharmProjects/kirovsk_230516/maps/s_cluster_map.html')