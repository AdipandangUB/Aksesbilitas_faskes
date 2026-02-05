import streamlit as st
import osmnx as ox
import networkx as nx
import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import folium
from streamlit_folium import st_folium, folium_static
from shapely.geometry import Point, Polygon, MultiPolygon, LineString, box
import warnings
import time
import math
from folium.plugins import MiniMap, Fullscreen
from shapely.ops import unary_union
from pyproj import CRS
from scipy.spatial import ConvexHull

# Suppress warnings
warnings.filterwarnings('ignore')

# Konfigurasi halaman
st.set_page_config(
    page_title="Analisis Jangkauan Fasilitas Kesehatan dengan Network Coverage",
    page_icon="🏥📍",
    layout="wide"
)

# Judul aplikasi
st.title("🏥📍 Analisis Spasial Jangkauan Fasilitas Kesehatan")
st.markdown("**Analisis zona jangkauan dengan dua metode coverage dari titik analisis**")

# ============================================================
# FUNGSI KONVERSI MODA TRANSPORTASI
# ============================================================
def convert_transport_mode(mode_bahasa):
    """
    Konversi mode transportasi dari bahasa Indonesia ke format OSMnx
    """
    conversion_map = {
        "jalan kaki": "walk",
        "sepeda": "bike", 
        "mobil/motor": "drive"
    }
    return conversion_map.get(mode_bahasa, "walk")

def get_default_speed(mode_bahasa):
    """
    Mendapatkan kecepatan default berdasarkan mode transportasi
    """
    speed_map = {
        "jalan kaki": 5.0,
        "sepeda": 15.0,
        "mobil/motor": 30.0
    }
    return speed_map.get(mode_bahasa, 5.0)

# ============================================================
# FUNGSI UTILITAS UMUM
# ============================================================
def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Menghitung jarak antara dua titik koordinat menggunakan formula Haversine.
    """
    R = 6371000  # Earth's radius in meters
    
    # Konversi ke radian
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)
    
    # Perbedaan
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    
    # Formula Haversine
    a = math.sin(dlat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    
    distance = R * c
    return distance

def get_health_facilities(bbox):
    """
    Mendapatkan fasilitas kesehatan dari OSM dalam bounding box
    """
    north, south, east, west = bbox
    
    try:
        # Coba ambil fasilitas kesehatan dengan tags spesifik
        tags_list = [
            {'amenity': 'hospital'},
            {'amenity': 'clinic'},
            {'healthcare': 'hospital'},
            {'healthcare': 'clinic'},
            {'amenity': 'pharmacy'},
            {'amenity': 'doctors'}
        ]
        
        all_facilities = []
        
        for tags in tags_list:
            try:
                facilities = ox.features_from_bbox(
                    north, south, east, west,
                    tags=tags
                )
                if not facilities.empty:
                    all_facilities.append(facilities)
            except:
                continue
        
        if all_facilities:
            combined = gpd.GeoDataFrame(pd.concat(all_facilities, ignore_index=True))
            # Hapus duplikat
            if 'geometry' in combined.columns:
                combined = combined.drop_duplicates(subset=['geometry'])
            
            # Pastikan ada kolom nama
            if 'name' not in combined.columns:
                combined['name'] = 'Fasilitas Kesehatan'
            
            return combined
        else:
            # Fallback: ambil semua amenity
            try:
                all_amenities = ox.features_from_bbox(
                    north, south, east, west,
                    tags={'amenity': True}
                )
                # Filter untuk fasilitas kesehatan
                health_keywords = ['hospital', 'clinic', 'health', 'doctor', 'pharmacy']
                mask = all_amenities['amenity'].astype(str).str.lower().isin(
                    [kw.lower() for kw in health_keywords]
                )
                filtered = all_amenities[mask]
                
                if 'name' not in filtered.columns:
                    filtered['name'] = 'Fasilitas Kesehatan'
                
                return filtered
            except:
                return gpd.GeoDataFrame()
                
    except Exception as e:
        st.warning(f"Tidak bisa mengambil data fasilitas: {str(e)}")
        return gpd.GeoDataFrame()

# ============================================================
# METODE-METODE COVERAGE AREA
# ============================================================

def get_network_from_point(location_point, network_type, radius):
    """Mendapatkan jaringan jalan dari titik analisis"""
    try:
        graph = ox.graph_from_point(
            location_point,
            dist=radius,
            network_type=network_type,
            simplify=True,
            truncate_by_edge=True
        )
        
        if len(graph.nodes()) == 0:
            st.error(f"❌ Tidak ada jaringan jalan ditemukan dari titik {location_point}")
            return None
        
        return graph
        
    except Exception as e:
        st.error(f"❌ Gagal mengambil jaringan dari titik: {str(e)}")
        return None

def find_start_node_from_point(graph, location_point):
    """Mencari node terdekat dari titik analisis"""
    try:
        # Proyeksikan graph untuk akurasi
        graph_proj = ox.project_graph(graph)
        
        # Temukan node terdekat dari titik analisis
        start_node = ox.distance.nearest_nodes(
            graph_proj, 
            location_point[1],  # longitude
            location_point[0]   # latitude
        )
        
        # Dapatkan koordinat node
        node_data = graph_proj.nodes[start_node]
        start_coords = (node_data['x'], node_data['y'])
        
        return start_node, graph_proj, start_coords
        
    except Exception as e:
        st.error(f"❌ Gagal menemukan node dari titik: {str(e)}")
        return None, None, None

# ============================================================
# METODE 1: SERVICE AREA - VERSI BARU DARI TITIK PENGAMATAN
# ============================================================
def calculate_service_area_coverage(graph_proj, reachable_nodes, distances, max_distance,
                                   location_point, buffer_distance=100):
    """Menghitung coverage sebagai service area dari titik pengamatan"""
    try:
        if not reachable_nodes:
            return create_simple_buffer_from_point(location_point, max_distance)
        
        # 1. Kumpulkan semua nodes yang terjangkau beserta jaraknya
        node_coords_with_distances = []
        for node in reachable_nodes:
            try:
                node_data = graph_proj.nodes[node]
                if 'x' in node_data and 'y' in node_data:
                    distance = distances.get(node, max_distance)
                    # Normalisasi jarak (0-1) untuk bobot
                    weight = 1.0 - (distance / max_distance)
                    node_coords_with_distances.append({
                        'coords': (node_data['x'], node_data['y']),
                        'distance': distance,
                        'weight': weight
                    })
            except:
                continue
        
        if not node_coords_with_distances:
            return create_simple_buffer_from_point(location_point, max_distance)
        
        # 2. Hitung centroid berbobot dari nodes yang terjangkau
        total_weight = sum(item['weight'] for item in node_coords_with_distances)
        
        if total_weight > 0:
            weighted_x = sum(item['coords'][0] * item['weight'] for item in node_coords_with_distances) / total_weight
            weighted_y = sum(item['coords'][1] * item['weight'] for item in node_coords_with_distances) / total_weight
            centroid = (weighted_x, weighted_y)
        else:
            # Jika tidak ada bobot, gunakan rata-rata sederhana
            avg_x = sum(item['coords'][0] for item in node_coords_with_distances) / len(node_coords_with_distances)
            avg_y = sum(item['coords'][1] for item in node_coords_with_distances) / len(node_coords_with_distances)
            centroid = (avg_x, avg_y)
        
        # 3. Dapatkan koordinat titik awal (dalam proyeksi graph)
        lat, lon = location_point
        try:
            # Coba dapatkan koordinat titik awal dalam sistem proyeksi graph
            transformer_to_crs = Transformer.from_crs('EPSG:4326', graph_proj.graph['crs'], always_xy=True)
            start_x, start_y = transformer_to_crs.transform(lon, lat)
            start_point_proj = (start_x, start_y)
        except:
            # Jika gagal, gunakan centroid sebagai titik awal
            start_point_proj = centroid
        
        # 4. Buat polygon service area dengan pendekatan radial dari titik awal
        # Kumpulkan semua titik untuk convex hull
        all_points = [start_point_proj]  # Mulai dari titik awal
        
        # Tambahkan nodes terjangkau
        for item in node_coords_with_distances:
            all_points.append(item['coords'])
        
        # 5. Buat convex hull dari semua titik
        if len(all_points) >= 3:
            points_array = np.array(all_points)
            try:
                hull = ConvexHull(points_array)
                hull_points = points_array[hull.vertices]
                polygon = Polygon(hull_points)
                
                # Buffer untuk smoothing dan memperluas area
                if buffer_distance > 0:
                    polygon = polygon.buffer(buffer_distance)
                
                # Tambahkan buffer tambahan di sekitar titik awal untuk memastikan coverage
                start_buffer = Point(start_point_proj).buffer(max_distance * 0.2)
                polygon = unary_union([polygon, start_buffer]).convex_hull
                
                return polygon
                
            except Exception:
                # Fallback: buat polygon dari nodes terjauh
                # Urutkan nodes berdasarkan jarak dari titik awal
                sorted_nodes = sorted(node_coords_with_distances, key=lambda x: x['distance'], reverse=True)
                
                if len(sorted_nodes) >= 3:
                    # Ambil nodes terjauh sebagai boundary
                    boundary_points = [start_point_proj]
                    for i in range(min(8, len(sorted_nodes))):  # Ambil maksimal 8 nodes terjauh
                        boundary_points.append(sorted_nodes[i]['coords'])
                    
                    points_array = np.array(boundary_points)
                    try:
                        hull = ConvexHull(points_array)
                        hull_points = points_array[hull.vertices]
                        polygon = Polygon(hull_points)
                        
                        if buffer_distance > 0:
                            polygon = polygon.buffer(buffer_distance)
                        
                        return polygon
                    except:
                        pass
        
        # 6. Fallback: buat buffer dari titik awal dengan radius yang disesuaikan
        # Gunakan rata-rata jarak nodes yang terjangkau
        avg_distance = sum(item['distance'] for item in node_coords_with_distances) / len(node_coords_with_distances)
        adjusted_distance = min(max_distance, avg_distance * 1.2)  # 20% lebih besar dari rata-rata
        
        return create_simple_buffer_from_point(location_point, adjusted_distance)
        
    except Exception as e:
        st.error(f"Error dalam service area coverage: {str(e)}")
        return create_simple_buffer_from_point(location_point, max_distance)

# ============================================================
# METODE 2: BUFFER DARI TITIK
# ============================================================
def calculate_buffer_coverage(location_point, max_distance, shape='Lingkaran'):
    """Menghitung coverage sebagai buffer dari titik"""
    try:
        lat, lon = location_point
        
        # Konversi meter ke derajat dengan akurasi yang lebih baik
        buffer_deg_lat = max_distance / 111320  # 111.32 km per degree latitude
        
        # Untuk longitude, perlu mempertimbangkan latitude
        buffer_deg_lon = max_distance / (111320 * math.cos(math.radians(lat)))
        
        if shape == 'Lingkaran':
            # Buffer lingkaran - versi lebih sederhana
            center_point = Point(lon, lat)
            # Konversi max_distance ke derajat (aproksimasi)
            approx_buffer_deg = max_distance / 111000  # 111 km per degree
            polygon = center_point.buffer(approx_buffer_deg, resolution=32)  # 32 sisi untuk lingkaran halus
            
        elif shape == 'Persegi':
            # Buffer persegi (bounding box)
            min_lon = lon - buffer_deg_lon
            max_lon = lon + buffer_deg_lon
            min_lat = lat - buffer_deg_lat
            max_lat = lat + buffer_deg_lat
            
            polygon = box(min_lon, min_lat, max_lon, max_lat)
            
        elif shape == 'Kapsul':
            # Buffer kapsul (lingkaran memanjang di arah timur-barat)
            circle1 = Point(lon - buffer_deg_lon/2, lat).buffer(buffer_deg_lat/2, resolution=16)
            circle2 = Point(lon + buffer_deg_lon/2, lat).buffer(buffer_deg_lat/2, resolution=16)
            rectangle = box(lon - buffer_deg_lon/2, lat - buffer_deg_lat/3, 
                           lon + buffer_deg_lon/2, lat + buffer_deg_lat/3)
            
            polygon = unary_union([circle1, circle2, rectangle])
        
        # Pastikan polygon valid
        if polygon.is_empty or not polygon.is_valid:
            # Fallback ke buffer lingkaran sederhana
            center_point = Point(lon, lat)
            approx_buffer_deg = max_distance / 111000
            polygon = center_point.buffer(approx_buffer_deg, resolution=16)
        
        return polygon
        
    except Exception as e:
        st.error(f"Error dalam buffer coverage: {str(e)}")
        # Fallback ekstrem
        return Point(location_point[1], location_point[0]).buffer(0.01)

# ============================================================
# FUNGSI UTAMA UNTUK METODE BUFFER YANG DIPERBAIKI
# ============================================================
def create_simple_buffer_from_point(location_point, max_distance, **kwargs):
    """Membuat buffer sederhana dari titik analisis"""
    shape = kwargs.get('buffer_shape', 'Lingkaran')
    return calculate_buffer_coverage(location_point, max_distance, shape)

def calculate_network_coverage_from_point(graph_proj, start_node, start_coords, 
                                        max_distance, location_point, method="Service Area", 
                                        **kwargs):
    """
    Menghitung coverage area dari titik analisis dengan dua metode
    """
    try:
        # Jika metode adalah Buffer dari Titik, langsung buat buffer tanpa analisis jaringan
        if method == "Buffer dari Titik":
            shape = kwargs.get('buffer_shape', 'Lingkaran')
            coverage_polygon = calculate_buffer_coverage(location_point, max_distance, shape)
            return coverage_polygon, []
        
        # Untuk Service Area, gunakan analisis jaringan
        # 1. Hitung jarak dari TITIK ANALISIS ke semua nodes
        distances = {}
        try:
            distances = nx.single_source_dijkstra_path_length(
                graph_proj, 
                start_node, 
                weight='length',
                cutoff=max_distance
            )
        except Exception as e:
            # Alternatif: hitung jarak Euclidean sebagai fallback
            for node in graph_proj.nodes():
                try:
                    node_data = graph_proj.nodes[node]
                    if 'x' in node_data and 'y' in node_data:
                        dx = node_data['x'] - start_coords[0]
                        dy = node_data['y'] - start_coords[1]
                        euclidean_dist = math.sqrt(dx**2 + dy**2)
                        if euclidean_dist <= max_distance:
                            distances[node] = euclidean_dist
                except:
                    continue
        
        if not distances:
            shape = kwargs.get('buffer_shape', 'Lingkaran')
            return create_simple_buffer_from_point(location_point, max_distance, buffer_shape=shape), []
        
        # 2. Filter nodes yang terjangkau
        reachable_nodes = []
        reachable_coords = []
        
        for node, dist in distances.items():
            if dist <= max_distance:
                reachable_nodes.append(node)
                node_data = graph_proj.nodes[node]
                if 'x' in node_data and 'y' in node_data:
                    reachable_coords.append((node_data['x'], node_data['y']))
        
        # 3. Hitung coverage area berdasarkan metode
        coverage_polygon = None
        
        if method == "Service Area":
            coverage_polygon = calculate_service_area_coverage(
                graph_proj, reachable_nodes, distances, max_distance,
                location_point, buffer_distance=kwargs.get('service_buffer', 100)
            )
        
        # 4. Kumpulkan edges yang terjangkau (kecuali untuk buffer)
        reachable_edges = []
        if method != "Buffer dari Titik":
            for u, v, data in graph_proj.edges(data=True):
                if u in reachable_nodes or v in reachable_nodes:
                    if 'geometry' in data:
                        reachable_edges.append(data['geometry'])
                    else:
                        u_coords = (graph_proj.nodes[u]['x'], graph_proj.nodes[u]['y'])
                        v_coords = (graph_proj.nodes[v]['x'], graph_proj.nodes[v]['y'])
                        reachable_edges.append(LineString([u_coords, v_coords]))
        
        return coverage_polygon, reachable_edges
        
    except Exception as e:
        st.error(f"Error dalam calculate_network_coverage: {str(e)}")
        shape = kwargs.get('buffer_shape', 'Lingkaran')
        return create_simple_buffer_from_point(location_point, max_distance, buffer_shape=shape), []

# ============================================================
# FUNGSI ANALISIS UTAMA - DIPERBAIKI
# ============================================================
def analyze_from_point_main(location_point, network_type, speed_kmh, radius_m, time_limits_min, 
                           method="Service Area", **kwargs):
    """
    Fungsi utama analisis dari titik dengan perhitungan fasilitas kesehatan
    """
    try:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Untuk metode Buffer, kita tidak perlu jaringan
        if method != "Buffer dari Titik":
            # 1. Dapatkan jaringan dari titik
            status_text.text("📍 Mendapatkan jaringan dari titik analisis...")
            graph = get_network_from_point(location_point, network_type, radius_m)
            if graph is None:
                return None, None, None, None, None
            
            progress_bar.progress(25)
            
            # 2. Temukan node awal dari titik
            status_text.text("📍 Mencari node awal dari titik analisis...")
            start_node, graph_proj, start_coords = find_start_node_from_point(graph, location_point)
            if start_node is None:
                return None, None, None, None, None
            
            progress_bar.progress(40)
        else:
            # Untuk metode Buffer, buat graph dummy
            graph_proj = None
            start_node = None
            start_coords = None
            progress_bar.progress(40)
        
        # 3. Dapatkan fasilitas kesehatan
        status_text.text("🏥 Mencari fasilitas dari titik analisis...")
        
        # Buat bbox dari titik
        lat, lon = location_point
        R = 6378137
        lat_offset = (radius_m / R) * (180 / math.pi)
        lon_offset = (radius_m / (R * math.cos(math.pi * lat / 180))) * (180 / math.pi)
        
        bbox = (
            lat + lat_offset,  # north
            lat - lat_offset,  # south
            lon + lon_offset,  # east
            lon - lon_offset   # west
        )
        
        health_facilities = get_health_facilities(bbox)
        
        progress_bar.progress(50)
        
        # 4. Hitung coverage area untuk setiap waktu
        status_text.text("🧮 Menghitung coverage dari titik analisis...")
        accessibility_zones = {}
        reachable_edges_dict = {}
        
        speed_m_per_min = (speed_kmh * 1000) / 60  # m/menit
        
        for time_limit in sorted(time_limits_min):
            max_distance = speed_m_per_min * time_limit
            
            # Tambahkan parameter tambahan berdasarkan metode
            method_kwargs = kwargs.copy()
            
            if method == "Buffer dari Titik":
                # Untuk metode buffer, langsung hitung tanpa jaringan
                shape = kwargs.get('buffer_shape', 'Lingkaran')
                coverage_polygon = calculate_buffer_coverage(location_point, max_distance, shape)
                reachable_edges = []
            else:
                # Untuk Service Area
                method_kwargs['service_buffer'] = kwargs.get('service_buffer', 100)
                
                coverage_polygon, reachable_edges = calculate_network_coverage_from_point(
                    graph_proj, start_node, start_coords, max_distance, location_point, method, **method_kwargs
                )
            
            if coverage_polygon and not coverage_polygon.is_empty:
                # Hitung luas
                area_m2 = coverage_polygon.area
                area_km2 = area_m2 / 1000000
                
                # Untuk buffer, langsung gunakan polygon yang sudah dalam WGS84
                if method == "Buffer dari Titik":
                    wgs84_polygon = coverage_polygon
                else:
                    # Konversi ke WGS84 untuk metode jaringan
                    try:
                        transformer = Transformer.from_crs(graph_proj.graph['crs'], 'EPSG:4326', always_xy=True)
                        
                        if hasattr(coverage_polygon, 'exterior'):
                            exterior_coords = list(coverage_polygon.exterior.coords)
                            wgs84_coords = []
                            for x, y in exterior_coords:
                                lon_conv, lat_conv = transformer.transform(x, y)
                                wgs84_coords.append((lon_conv, lat_conv))
                            wgs84_polygon = Polygon(wgs84_coords)
                        else:
                            wgs84_polygon = coverage_polygon
                            
                    except:
                        buffer_deg = max_distance / 111320
                        wgs84_polygon = Point(location_point[1], location_point[0]).buffer(buffer_deg)
                
                # Hitung fasilitas yang terjangkau
                accessible_facilities = []
                if not health_facilities.empty and wgs84_polygon:
                    for idx, facility in health_facilities.iterrows():
                        try:
                            if hasattr(facility.geometry, 'within'):
                                if facility.geometry.within(wgs84_polygon):
                                    fac_name = facility.get('name', 'Fasilitas Kesehatan')
                                    fac_type = facility.get('amenity', facility.get('healthcare', 'Kesehatan'))
                                    
                                    # Hitung jarak ke titik awal
                                    if hasattr(facility.geometry, 'x'):
                                        fac_lat, fac_lon = facility.geometry.y, facility.geometry.x
                                        distance = haversine_distance(lat, lon, fac_lat, fac_lon)
                                        travel_time = distance / speed_m_per_min
                                        
                                        accessible_facilities.append({
                                            'name': str(fac_name),
                                            'type': str(fac_type),
                                            'geometry': facility.geometry,
                                            'distance_m': distance,
                                            'travel_time_min': travel_time,
                                            'coordinates': (fac_lat, fac_lon)
                                        })
                        except:
                            continue
                
                # Simpan statistik nodes dan edges untuk metode buffer
                nodes_count = len(graph_proj.nodes()) if graph_proj else 0
                edges_count = len(graph_proj.edges()) if graph_proj else 0
                
                accessibility_zones[time_limit] = {
                    'geometry': wgs84_polygon,
                    'geometry_projected': coverage_polygon,
                    'max_distance': max_distance,
                    'area_sqkm': area_km2,
                    'calculation_method': method,
                    'nodes_count': nodes_count,
                    'edges_count': edges_count,
                    'reachable_edges': len(reachable_edges),
                    'accessible_facilities': accessible_facilities,
                    'facilities_count': len(accessible_facilities)
                }
                
                reachable_edges_dict[time_limit] = reachable_edges
        
        progress_bar.progress(100)
        status_text.text("✅ Analisis dari titik selesai!")
        time.sleep(0.5)
        status_text.empty()
        
        return graph_proj, health_facilities, accessibility_zones, reachable_edges_dict
        
    except Exception as e:
        st.error(f"❌ Error dalam analisis dari titik: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return None, None, None, None

# ============================================================
# FUNGSI PEMBUATAN PETA - DIPERBAIKI
# ============================================================
def create_comprehensive_map(location_point, accessibility_zones, health_facilities):
    """
    Membuat peta interaktif yang komprehensif
    """
    try:
        m = folium.Map(location=location_point, zoom_start=14, 
                      tiles='OpenStreetMap', control_scale=True)
        
        # Tambahkan marker titik awal
        folium.Marker(
            location=location_point,
            popup='<b>📍 Titik Analisis</b><br>Lokasi awal perhitungan jangkauan',
            tooltip='Titik Analisis',
            icon=folium.Icon(color='red', icon='bullseye', prefix='fa')
        ).add_to(m)
        
        # Warna untuk zona aksesibilitas
        colors = {
            5: '#FFEBEE',
            10: '#FFCDD2',
            15: '#EF9A9A',
            20: '#E57373',
            25: '#EF5350',
            30: '#F44336'
        }
        
        # Tambahkan zona aksesibilitas
        for time_limit, zone_data in accessibility_zones.items():
            color = colors.get(time_limit, '#EF9A9A')
            
            try:
                if 'geometry' in zone_data and zone_data['geometry']:
                    polygon = zone_data['geometry']
                    
                    if hasattr(polygon, 'exterior'):
                        coords = list(polygon.exterior.coords)
                        folium_coords = [(lat, lon) for lon, lat in coords]
                        
                        folium.Polygon(
                            locations=folium_coords,
                            color=color,
                            fill=True,
                            fill_color=color,
                            fill_opacity=0.3,
                            weight=2,
                            popup=f'<b>Zona {time_limit} menit</b><br>'
                                  f'Metode: {zone_data.get("calculation_method", "Network")}<br>'
                                  f'Luas: {zone_data["area_sqkm"]:.2f} km²<br>'
                                  f'Jarak maks: {zone_data["max_distance"]:.0f} m<br>'
                                  f'Fasilitas: {zone_data.get("facilities_count", 0)}',
                            tooltip=f'Zona Aksesibilitas {time_limit} menit'
                        ).add_to(m)
                    else:
                        # Jika tidak ada exterior, coba buat circle sebagai fallback
                        folium.Circle(
                            location=location_point,
                            radius=zone_data['max_distance'],
                            color=color,
                            fill=True,
                            fill_color=color,
                            fill_opacity=0.3,
                            popup=f'<b>Zona {time_limit} menit (Buffer)</b><br>'
                                  f'Luas: {zone_data["area_sqkm"]:.2f} km²<br>'
                                  f'Jarak: {zone_data["max_distance"]:.0f} m'
                        ).add_to(m)
                        
            except Exception as e:
                # Fallback: buat circle sederhana
                folium.Circle(
                    location=location_point,
                    radius=zone_data['max_distance'],
                    color=color,
                    fill=True,
                    fill_color=color,
                    fill_opacity=0.3,
                    popup=f'Zona {time_limit} menit (fallback)'
                ).add_to(m)
        
        # Tambahkan fasilitas kesehatan yang dapat diakses
        facility_colors = {
            'hospital': 'red',
            'clinic': 'blue',
            'doctors': 'green',
            'pharmacy': 'orange',
            'default': 'purple'
        }
        
        facility_icons = {
            'hospital': 'hospital',
            'clinic': 'medkit',
            'doctors': 'user-md',
            'pharmacy': 'pills',
            'default': 'heart'
        }
        
        # Simpan fasilitas yang sudah ditambahkan
        added_facilities = set()
        
        for time_limit, zone_data in accessibility_zones.items():
            if 'accessible_facilities' in zone_data:
                for facility in zone_data['accessible_facilities']:
                    try:
                        fac_key = f"{facility['coordinates'][0]},{facility['coordinates'][1]}"
                        
                        if fac_key in added_facilities:
                            continue
                        
                        added_facilities.add(fac_key)
                        
                        fac_type = str(facility['type']).lower()
                        
                        # Tentukan warna dan ikon
                        color = facility_colors['default']
                        icon = facility_icons['default']
                        
                        for key in ['hospital', 'clinic', 'doctors', 'pharmacy']:
                            if key in fac_type:
                                color = facility_colors[key]
                                icon = facility_icons.get(key, facility_icons['default'])
                                break
                        
                        # Buat popup content
                        popup_content = f"""
                        <div style="min-width: 200px;">
                            <h4 style="margin-bottom: 5px; color: #2c3e50;">{facility['name']}</h4>
                            <hr style="margin: 5px 0;">
                            <p style="margin: 2px 0;"><b>Jenis:</b> {facility['type']}</p>
                            <p style="margin: 2px 0;"><b>Waktu tempuh:</b> {facility['travel_time_min']:.1f} menit</p>
                            <p style="margin: 2px 0;"><b>Jarak:</b> {facility['distance_m']:.0f} m</p>
                        </div>
                        """
                        
                        folium.Marker(
                            location=facility['coordinates'],
                            popup=folium.Popup(popup_content, max_width=300),
                            tooltip=f"{facility['name']} ({facility['travel_time_min']:.1f} menit)",
                            icon=folium.Icon(color=color, icon=icon, prefix='fa')
                        ).add_to(m)
                        
                    except Exception as e:
                        continue
        
        # Tambahkan fasilitas lain yang tidak dapat diakses
        if not health_facilities.empty:
            for idx, facility in health_facilities.iterrows():
                try:
                    if hasattr(facility.geometry, 'x'):
                        coords = (facility.geometry.y, facility.geometry.x)
                        fac_key = f"{coords[0]},{coords[1]}"
                        
                        if fac_key not in added_facilities:
                            fac_name = str(facility.get('name', 'Fasilitas Tanpa Nama'))
                            
                            folium.CircleMarker(
                                location=coords,
                                radius=3,
                                color='gray',
                                fill=True,
                                fill_color='gray',
                                fill_opacity=0.5,
                                popup=f"<b>{fac_name}</b><br><i>(di luar zona jangkauan)</i>",
                                tooltip=fac_name
                            ).add_to(m)
                            
                except Exception:
                    continue
        
        # Tambahkan layer control
        folium.LayerControl().add_to(m)
        
        # Tambahkan mini map
        minimap = MiniMap(toggle_display=True)
        m.add_child(minimap)
        
        # Tambahkan fitur fullscreen
        Fullscreen().add_to(m)
        
        return m
        
    except Exception as e:
        st.error(f"Error membuat peta: {str(e)}")
        # Return peta dasar dengan titik analisis
        m = folium.Map(location=location_point, zoom_start=14)
        folium.Marker(location_point, popup='Titik Analisis').add_to(m)
        return m

# ============================================================
# SIDEBAR INPUT
# ============================================================
with st.sidebar:
    st.header("📍 Parameter Titik Analisis")
    
    input_method = st.radio(
        "Metode Input Titik Analisis:",
        ["Pilih Kota", "Input Manual"]
    )
    
    if input_method == "Pilih Kota":
        kota_options = {
            "Malang": (-7.9819, 112.6200),
            "Jakarta": (-6.2088, 106.8456),
            "Bandung": (-6.9175, 107.6191),
            "Surabaya": (-7.2575, 112.7521),
            "Yogyakarta": (-7.7956, 110.3695),
            "Semarang": (-6.9667, 110.4167),
            "Denpasar": (-8.6705, 115.2126),
            "Medan": (3.5952, 98.6722),
            "Makassar": (-5.1477, 119.4327)
        }
        
        selected_city = st.selectbox("Pilih Kota:", list(kota_options.keys()))
        location_point = kota_options[selected_city]
        st.success(f"📍 **Titik Analisis:** {selected_city}: {location_point}")
        
    else:
        st.write("Masukkan koordinat titik analisis:")
        lat = st.number_input("Latitude:", value=-6.2088, format="%.6f")
        lon = st.number_input("Longitude:", value=106.8456, format="%.6f")
        location_point = (lat, lon)
        st.success(f"📍 **Titik Analisis:** {lat}, {lon}")
    
    st.subheader("⚙️ Pengaturan Analisis")
    
    network_type_bahasa = st.selectbox(
        "Mode Transportasi:",
        ["jalan kaki", "sepeda", "mobil/motor"],
        index=0,
        help="Jenis jaringan yang digunakan untuk analisis"
    )
    
    network_type_osmnx = convert_transport_mode(network_type_bahasa)
    
    travel_speed = st.slider(
        "Kecepatan (km/jam):",
        min_value=1.0,
        max_value=60.0,
        value=get_default_speed(network_type_bahasa),
        step=0.5
    )
    
    search_radius = st.slider(
        "Radius Jaringan (meter):",
        min_value=500,
        max_value=5000,
        value=2000,
        step=100
    )
    
    time_limits = st.multiselect(
        "Batas Waktu (menit):",
        [5, 10, 15, 20, 25, 30],
        default=[15, 25]
    )
    
    # Pilihan metode perhitungan luas - HANYA 2 METODE
    area_calculation_method = st.selectbox(
        "Metode Coverage Area:",
        [
            "Service Area", 
            "Buffer dari Titik"
        ],
        index=0,
        help="Metode untuk menghitung coverage area dari titik analisis"
    )
    
    # Parameter tambahan berdasarkan metode
    if area_calculation_method == "Service Area":
        service_buffer = st.slider(
            "Buffer Service Area (meter):",
            20, 5000, 100, 10  # Diubah dari 500 menjadi 5000
        )
    
    elif area_calculation_method == "Buffer dari Titik":
        buffer_shape = st.selectbox(
            "Bentuk Buffer:",
            ["Lingkaran", "Persegi", "Kapsul"],
            index=0
        )
    
    analyze_button = st.button("🚀 Jalankan Analisis", type="primary", use_container_width=True)
    
    st.markdown("---")
    st.info("""
    **📌 Panduan Metode:**
    
    1. **Service Area**: 
       - Convex hull berbobot dari titik pengamatan dan nodes terjangkau
       - Mempertimbangkan struktur jaringan jalan
       - Akurat untuk analisis berbasis jaringan
       - **Buffer hingga 5000 meter** untuk area coverage yang lebih luas
    
    2. **Buffer dari Titik**: 
       - Buffer sederhana dari titik analisis
       - Tidak memerlukan analisis jaringan jalan
       - Cepat dan sederhana untuk estimasi awal
    
    **🎯 Rekomendasi:**
    - **Service Area**: Untuk analisis akurat berdasarkan jaringan jalan
    - **Buffer dari Titik**: Untuk analisis cepat dan sederhana
    """)

# ============================================================
# MAIN APPLICATION
# ============================================================
if analyze_button and time_limits:
    # Kumpulkan parameter tambahan
    kwargs = {}
    if area_calculation_method == "Service Area":
        kwargs['service_buffer'] = service_buffer
    
    elif area_calculation_method == "Buffer dari Titik":
        kwargs['buffer_shape'] = buffer_shape
    
    # Jalankan analisis
    with st.spinner("📍 Sedang menganalisis dari titik..."):
        result = analyze_from_point_main(
            location_point, 
            network_type_osmnx, 
            travel_speed, 
            search_radius, 
            time_limits, 
            area_calculation_method,
            **kwargs
        )
    
    if result[0] is not None or area_calculation_method == "Buffer dari Titik":
        graph, health_facilities, accessibility_zones, reachable_edges = result
        
        # Tampilkan hasil dalam tab
        tab1, tab2, tab3, tab4 = st.tabs(["🗺️ Peta Interaktif", "📊 Dashboard", "🏥 Fasilitas", "📈 Analisis"])
        
        with tab1:
            st.subheader("🗺️ Peta Jangkauan Fasilitas Kesehatan")
            
            if accessibility_zones:
                m = create_comprehensive_map(location_point, accessibility_zones, health_facilities)
                st_folium(m, width=1200, height=600, returned_objects=[])
            else:
                st.warning("⚠️ Tidak ada zona jangkauan yang dapat dihitung.")

        with tab2:
            st.subheader("📊 Dashboard Analisis")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if graph:
                    st.metric(
                        "📌 Node Jaringan",
                        f"{len(graph.nodes()):,}",
                        help="Jumlah simpul dalam jaringan jalan"
                    )
                elif area_calculation_method == "Buffer dari Titik":
                    st.metric(
                        "📌 Metode",
                        "Buffer Langsung",
                        help="Metode buffer tidak memerlukan analisis jaringan"
                    )
            
            with col2:
                if graph:
                    st.metric(
                        "🛣️ Edge Jaringan",
                        f"{len(graph.edges()):,}",
                        help="Jumlah segmen jalan"
                    )
                elif area_calculation_method == "Buffer dari Titik":
                    st.metric(
                        "🛣️ Bentuk Buffer",
                        buffer_shape,
                        help="Bentuk buffer yang digunakan"
                    )
            
            with col3:
                total_facilities = health_facilities.shape[0] if not health_facilities.empty else 0
                st.metric(
                    "🏥 Total Fasilitas",
                    total_facilities,
                    help="Total fasilitas kesehatan dalam area"
                )
            
            # Tampilkan informasi mode transportasi
            col4, col5 = st.columns(2)
            with col4:
                st.metric(
                    "🚗 Mode Transportasi",
                    network_type_bahasa.capitalize(),
                    delta="Terpilih"
                )
            with col5:
                st.metric(
                    "⚡ Kecepatan",
                    f"{travel_speed} km/jam",
                    help="Kecepatan yang digunakan untuk perhitungan"
                )
            
            # Tampilkan zona jangkauan
            st.subheader("📍 Zona Jangkauan")
            
            if accessibility_zones:
                zones_data = []
                for time_limit, zone_data in accessibility_zones.items():
                    zones_data.append({
                        "⏱️ Batas Waktu": f"{time_limit} menit",
                        "🔧 Metode": zone_data.get('calculation_method', area_calculation_method),
                        "📐 Luas Area": f"{zone_data['area_sqkm']:.2f} km²",
                        "🎯 Jangkauan": f"{zone_data['max_distance']:.0f} m",
                        "🏥 Fasilitas": f"{zone_data.get('facilities_count', 0)}",
                        "📊 Node Jaringan": zone_data['nodes_count'] if area_calculation_method != "Buffer dari Titik" else "N/A"
                    })
                
                zones_df = pd.DataFrame(zones_data)
                st.dataframe(zones_df, use_container_width=True, hide_index=True)
            else:
                st.info("ℹ️ Tidak ada zona jangkauan yang dapat dihitung.")
        
        with tab3:
            st.subheader("🏥 Fasilitas Kesehatan yang Dapat Diakses")
            
            if accessibility_zones and any('accessible_facilities' in zone and zone['accessible_facilities'] 
                                         for zone in accessibility_zones.values()):
                for time_limit in sorted(time_limits):
                    if time_limit in accessibility_zones:
                        facilities = accessibility_zones[time_limit].get('accessible_facilities', [])
                        
                        with st.expander(f"🏥 Fasilitas dalam {time_limit} menit ({len(facilities)})", expanded=False):
                            if facilities:
                                # Buat dataframe fasilitas
                                fac_data = []
                                for fac in facilities:
                                    fac_data.append({
                                        "🏷️ Nama": fac['name'][:50] + "..." if len(fac['name']) > 50 else fac['name'],
                                        "🔧 Jenis": fac['type'],
                                        "📏 Jarak (m)": f"{fac['distance_m']:.0f}",
                                        "⏱️ Waktu (menit)": f"{fac['travel_time_min']:.1f}",
                                        "📍 Latitude": f"{fac['coordinates'][0]:.4f}",
                                        "📍 Longitude": f"{fac['coordinates'][1]:.4f}"
                                    })
                                
                                fac_df = pd.DataFrame(fac_data)
                                st.dataframe(fac_df, use_container_width=True, hide_index=True)
                            else:
                                st.info(f"ℹ️ Tidak ada fasilitas yang dapat diakses dalam {time_limit} menit.")
            else:
                st.warning("⚠️ Tidak ada fasilitas kesehatan yang dapat diakses dalam area ini.")
        
        with tab4:
            st.subheader("📈 Analisis dan Visualisasi")
            
            if accessibility_zones:
                col1, col2 = st.columns(2)
                
                with col1:
                    # Plot area zona
                    fig, ax = plt.subplots(figsize=(10, 6))
                    
                    time_limits_list = list(accessibility_zones.keys())
                    areas = [accessibility_zones[t]['area_sqkm'] for t in time_limits_list]
                    
                    # Warna gradient
                    colors = plt.cm.YlOrRd(np.linspace(0.4, 0.9, len(time_limits_list)))
                    
                    bars = ax.bar([str(t) for t in time_limits_list], areas, color=colors, edgecolor='black')
                    ax.set_xlabel('Batas Waktu (menit)', fontsize=12, fontweight='bold')
                    ax.set_ylabel('Luas Area (km²)', fontsize=12, fontweight='bold')
                    ax.set_title('Luas Area Jangkauan', fontsize=14, fontweight='bold')
                    
                    # Tambahkan nilai di atas bar
                    for bar in bars:
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height,
                                f'{height:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
                    
                    plt.grid(axis='y', alpha=0.3)
                    plt.tight_layout()
                    st.pyplot(fig)
                
                with col2:
                    # Plot jumlah fasilitas per zona
                    fig2, ax2 = plt.subplots(figsize=(10, 6))
                    
                    facilities_count = []
                    for time_limit in sorted(time_limits):
                        if time_limit in accessibility_zones:
                            facilities_count.append(accessibility_zones[time_limit].get('facilities_count', 0))
                        else:
                            facilities_count.append(0)
                    
                    if sum(facilities_count) > 0:
                        # Warna gradient
                        colors2 = plt.cm.Blues(np.linspace(0.4, 0.9, len(time_limits)))
                        
                        bars2 = ax2.bar([str(t) for t in sorted(time_limits)], facilities_count, 
                                       color=colors2, edgecolor='black')
                        ax2.set_xlabel('Batas Waktu (menit)', fontsize=12, fontweight='bold')
                        ax2.set_ylabel('Jumlah Fasilitas', fontsize=12, fontweight='bold')
                        ax2.set_title('Fasilitas yang Dapat Diakses', fontsize=14, fontweight='bold')
                        
                        # Tambahkan nilai di atas bar
                        for bar in bars2:
                            height = bar.get_height()
                            ax2.text(bar.get_x() + bar.get_width()/2., height,
                                    f'{height}', ha='center', va='bottom', fontsize=10, fontweight='bold')
                        
                        plt.grid(axis='y', alpha=0.3)
                    else:
                        ax2.text(0.5, 0.5, 'Tidak ada data fasilitas', 
                                ha='center', va='center', transform=ax2.transAxes, 
                                fontsize=12, fontweight='bold')
                        ax2.set_title('Tidak Ada Fasilitas Ditemukan', fontsize=14, fontweight='bold')
                    
                    plt.tight_layout()
                    st.pyplot(fig2)
            
                # Ringkasan statistik
                st.subheader("📋 Ringkasan Analisis")
                
                total_facilities_all = sum(zone.get('facilities_count', 0) for zone in accessibility_zones.values())
                
                if total_facilities_all > 0:
                    # Hitung statistik
                    unique_names = set()
                    all_travel_times = []
                    all_distances = []
                    
                    for zone_data in accessibility_zones.values():
                        if 'accessible_facilities' in zone_data:
                            for fac in zone_data['accessible_facilities']:
                                unique_names.add(fac['name'])
                                all_travel_times.append(fac['travel_time_min'])
                                all_distances.append(fac['distance_m'])
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("🏷️ Fasilitas Unik", len(unique_names))
                    
                    with col2:
                        avg_time = np.mean(all_travel_times) if all_travel_times else 0
                        st.metric("⏱️ Rata-rata Waktu", f"{avg_time:.1f} menit")
                    
                    with col3:
                        avg_distance = np.mean(all_distances) if all_distances else 0
                        st.metric("📏 Rata-rata Jarak", f"{avg_distance:.0f} m")
                    
                    with col4:
                        min_time = min(all_travel_times) if all_travel_times else 0
                        st.metric("⚡ Waktu Terdekat", f"{min_time:.1f} menit")
                else:
                    st.info("ℹ️ Tidak ada data fasilitas untuk ditampilkan.")
    
    else:
        st.error("❌ Analisis gagal. Periksa parameter dan coba lagi.")

elif not analyze_button:
    # Tampilkan halaman awal jika analisis belum dijalankan
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ## 📋 Panduan Penggunaan
        
        1. **📍 Pilih lokasi** di sidebar (kota atau koordinat manual)
        2. **⚙️ Atur parameter**:
           - Mode transportasi (jalan kaki/sepeda/mobil/motor)
           - Kecepatan perjalanan
           - Radius pencarian (500-5000m)
           - Batas waktu jangkauan (menit)
           - Metode coverage area (hanya 2 pilihan)
        3. **🚀 Klik "Jalankan Analisis"** untuk memulai
        
        ## 🎯 Fitur Utama

        - **2 metode coverage area** yang disederhanakan
        - **Service Area**: Analisis akurat berbasis jaringan jalan
        - **Buffer dari Titik**: Analisis cepat dan sederhana
        - **Identifikasi fasilitas kesehatan** yang terjangkau
        - **Peta interaktif** dengan filter dan layer
        - **Dashboard statistik** dan visualisasi
        """)
    
    with col2:
        st.markdown("""
        ## 💡 Tips Optimal
        
        ### Untuk Hasil Terbaik:
        - Gunakan **radius 1000-2000m** untuk analisis cepat
        - Pilih **mode 'jalan kaki'** untuk analisis pejalan kaki
        - **Batas waktu 15-25 menit** memberikan hasil optimal
        
        ### Rekomendasi Metode:
        - **Service Area**: Untuk analisis akurat berbasis jaringan
        - **Buffer dari Titik**: Untuk estimasi cepat tanpa jaringan
        """)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; padding: 20px;'>
        <p style='font-size: 1.1em; font-weight: bold; color: #2c3e50;'>
        🏥📍 <b>Analisis Jangkauan Fasilitas Kesehatan dengan Network Coverage</b> v3.0
        </p>
        <p style='font-size: 0.9em; color: #7f8c8d;'>
        Developer: Adipandang Yudono, S.Si., MURP., PhD (Spatial Analysis, Architecture System, Scrypt Developer, WebGIS Analytics) & dr. Aurick Yudha Nagara, Sp.EM, KPEC (Health Facilities Analysis)
        <br>
        Ragam aplikasi yang digunakan: Streamlit • OSMnx • Folium • GeoPandas • NetworkX • SciPy
        <br>
        Data sumber: © OpenStreetMap contributors
        <br>
        2026
        </p>
    </div>
    """, 
    unsafe_allow_html=True
)

# CSS tambahan
st.markdown("""
<style>
    /* Tombol utama */
    .stButton > button {
        background: linear-gradient(45deg, #4B79A1, #283E51);
        color: white;
        border: none;
        padding: 0.75rem 1.5rem;
        border-radius: 10px;
        font-weight: bold;
        font-size: 1.1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(75, 121, 161, 0.3);
    }
    
    /* Header styling */
    .st-emotion-cache-10trblm {
        background: linear-gradient(90deg, #4B79A1, #283E51);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding: 10px 0;
    }
    
    /* Card styling */
    [data-testid="stMetric"] {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 15px;
        border-left: 4px solid #4B79A1;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 1rem;
        padding: 0 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
        font-weight: bold;
        transition: all 0.3s;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #e9ecef;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #4B79A1 !important;
        color: white !important;
        box-shadow: 0 2px 4px rgba(75, 121, 161, 0.3);
    }
    
    /* Dataframe styling */
    .dataframe {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    /* Progress bar */
    .stProgress > div > div {
        background: linear-gradient(90deg, #4B79A1, #6C8EBF);
    }
</style>
""", unsafe_allow_html=True)
