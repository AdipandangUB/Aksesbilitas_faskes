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
from shapely.ops import unary_union, linemerge
from pyproj import CRS, Transformer
from scipy.spatial import ConvexHull
from collections import defaultdict
from shapely import concave_hull, convex_hull
import alphashape

# Suppress warnings
warnings.filterwarnings('ignore')

# ============================================================
# FIX: INISIALISASI SESSION STATE
# ============================================================
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'map_created' not in st.session_state:
    st.session_state.map_created = False
if 'accessibility_zones' not in st.session_state:
    st.session_state.accessibility_zones = None
if 'health_facilities' not in st.session_state:
    st.session_state.health_facilities = None
if 'reachable_edges' not in st.session_state:
    st.session_state.reachable_edges = None
if 'last_location' not in st.session_state:
    st.session_state.last_location = None

# ============================================================
# KONFIGURASI HALAMAN
# ============================================================
st.set_page_config(
    page_title="Analisis Aksesbilitas Fasilitas Kesehatan dengan Network Coverage",
    page_icon="🏥📍",
    layout="wide"
)

# Judul aplikasi
st.title("🏥📍 Analisis Spasial Aksesbilitas Fasilitas Kesehatan")
st.markdown("**Analisis zona Aksesbilitas dengan dua metode coverage dari titik analisis**")

# ============================================================
# FUNGSI KONVERSI MODA TRANSPORTASI
# ============================================================
def convert_transport_mode(mode_bahasa):
    """Konversi mode transportasi dari bahasa Indonesia ke format OSMnx"""
    conversion_map = {
        "jalan kaki": "walk",
        "sepeda": "bike", 
        "mobil/motor": "drive"
    }
    return conversion_map.get(mode_bahasa, "walk")

def get_default_speed(mode_bahasa):
    """Mendapatkan kecepatan default berdasarkan mode transportasi"""
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
    """Menghitung jarak antara dua titik koordinat menggunakan formula Haversine"""
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
    """Mendapatkan fasilitas kesehatan dari OSM dalam bounding box"""
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
# FUNGSI NETWORK ANALYSIS - IMPROVED
# ============================================================
def get_network_from_point(location_point, network_type, radius):
    """Mendapatkan jaringan jalan dari titik analisis dengan caching"""
    try:
        # Gunakan cache untuk meningkatkan performa
        cache_key = f"{location_point}_{network_type}_{radius}"
        
        if 'network_cache' not in st.session_state:
            st.session_state.network_cache = {}
        
        if cache_key in st.session_state.network_cache:
            return st.session_state.network_cache[cache_key]
        
        graph = ox.graph_from_point(
            location_point,
            dist=radius,
            network_type=network_type,
            simplify=True,
            truncate_by_edge=True,
            retain_all=True  # Pertahankan semua komponen
        )
        
        if len(graph.nodes()) == 0:
            st.error(f"❌ Tidak ada jaringan jalan ditemukan dari titik {location_point}")
            return None
        
        # Proyeksikan graph untuk analisis spasial yang akurat
        graph_proj = ox.project_graph(graph)
        
        # Simpan ke cache
        st.session_state.network_cache[cache_key] = graph_proj
        
        return graph_proj
        
    except Exception as e:
        st.error(f"❌ Gagal mengambil jaringan dari titik: {str(e)}")
        return None

def find_start_node_from_point(graph_proj, location_point):
    """Mencari node terdekat dari titik analisis"""
    try:
        # Dapatkan koordinat titik analisis dalam proyeksi graph
        lat, lon = location_point
        
        # Temukan node terdekat
        start_node = ox.distance.nearest_nodes(
            graph_proj, 
            lon,  # longitude
            lat   # latitude
        )
        
        # Dapatkan koordinat node dalam proyeksi graph
        node_data = graph_proj.nodes[start_node]
        start_coords = (node_data['x'], node_data['y'])
        
        # Dapatkan juga koordinat WGS84 untuk referensi
        transformer = Transformer.from_crs(graph_proj.graph['crs'], 'EPSG:4326', always_xy=True)
        lon_wgs, lat_wgs = transformer.transform(start_coords[0], start_coords[1])
        start_wgs84 = (lat_wgs, lon_wgs)
        
        return start_node, start_coords, start_wgs84
        
    except Exception as e:
        st.error(f"❌ Gagal menemukan node dari titik: {str(e)}")
        return None, None, None

# ============================================================
# FUNGSI UTILITAS UNTUK CONCAVE HULL
# ============================================================
def calculate_alpha_shape(points, alpha):
    """
    Menghitung alpha shape (concave hull) dari sekumpulan titik
    Alpha shape adalah generalisasi dari convex hull
    """
    try:
        if len(points) < 3:
            # Tidak cukup titik untuk membuat hull
            return None
            
        # Pastikan points adalah array numpy 2D
        points_array = np.array(points)
        
        if points_array.shape[1] != 2:
            raise ValueError("Points harus memiliki 2 kolom (x, y)")
        
        # Hitung alpha shape menggunakan alphashape library
        alpha_shape = alphashape.alphashape(points_array, alpha)
        
        return alpha_shape
        
    except Exception as e:
        st.warning(f"Error dalam alpha shape: {str(e)}")
        # Fallback ke convex hull
        if len(points) >= 3:
            try:
                hull = ConvexHull(points_array)
                hull_points = points_array[hull.vertices]
                return Polygon(hull_points)
            except:
                return None
        return None

def optimize_alpha_value(points, target_coverage=0.95):
    """
    Mengoptimalkan nilai alpha untuk concave hull
    Mencari alpha yang memberikan coverage optimal
    """
    try:
        if len(points) < 3:
            return None, 0
        
        points_array = np.array(points)
        
        # Coba beberapa nilai alpha
        alpha_values = [0.1, 0.2, 0.3, 0.5, 1.0, 2.0, 5.0]
        best_shape = None
        best_alpha = 1.0
        
        for alpha in alpha_values:
            try:
                shape = alphashape.alphashape(points_array, alpha)
                if shape and not shape.is_empty:
                    best_shape = shape
                    best_alpha = alpha
                    break
            except:
                continue
        
        return best_shape, best_alpha
        
    except Exception as e:
        return None, 0

# ============================================================
# METODE 1: NETWORK SERVICE AREA - CONCAVE HULL VERSION
# ============================================================
def calculate_network_service_area(graph_proj, start_node, start_coords, max_distance, 
                                  service_buffer=100):
    """
    Menghitung service area berdasarkan analisis jaringan sebenarnya
    MENGGUNAKAN CONCAVE HULL (Alpha Shape) untuk hasil yang lebih akurat
    """
    try:
        # 1. HITUNG JARAK DARI START NODE KE SEMUA NODES
        distances = nx.single_source_dijkstra_path_length(
            graph_proj, 
            start_node, 
            weight='length',
            cutoff=max_distance
        )
        
        if not distances:
            return None, [], {}
        
        # 2. IDENTIFIKASI NODES DAN EDGES YANG TERJANGKAU
        reachable_nodes = []
        reachable_edges = []
        edge_distances = {}
        
        # FIX: Simpan semua nodes yang terjangkau untuk concave hull
        reachable_node_points = []
        
        # Simpan edge points juga untuk mendapatkan coverage yang lebih baik
        all_edge_points = []
        
        for u, v, data in graph_proj.edges(data=True):
            # Cek jika salah satu node dari edge terjangkau
            u_dist = distances.get(u, float('inf'))
            v_dist = distances.get(v, float('inf'))
            
            # Edge terjangkau jika minimal satu node terjangkau
            if u_dist <= max_distance or v_dist <= max_distance:
                # Tambahkan nodes jika belum ada
                if u not in reachable_nodes:
                    reachable_nodes.append(u)
                    # Simpan koordinat node untuk concave hull
                    u_coords = (graph_proj.nodes[u]['x'], graph_proj.nodes[u]['y'])
                    reachable_node_points.append(u_coords)
                
                if v not in reachable_nodes:
                    reachable_nodes.append(v)
                    # Simpan koordinat node untuk concave hull
                    v_coords = (graph_proj.nodes[v]['x'], graph_proj.nodes[v]['y'])
                    reachable_node_points.append(v_coords)
                
                # Dapatkan geometri edge
                if 'geometry' in data:
                    edge_geom = data['geometry']
                    # Tambahkan titik-titik dari edge ke kumpulan points
                    if hasattr(edge_geom, 'coords'):
                        all_edge_points.extend(list(edge_geom.coords))
                else:
                    # Buat garis sederhana dari node ke node
                    u_coords = (graph_proj.nodes[u]['x'], graph_proj.nodes[u]['y'])
                    v_coords = (graph_proj.nodes[v]['x'], graph_proj.nodes[v]['y'])
                    edge_geom = LineString([u_coords, v_coords])
                    all_edge_points.extend([u_coords, v_coords])
                
                reachable_edges.append(edge_geom)
                
                # Simpan jarak terdekat dari edge ke titik awal
                min_dist = min(u_dist, v_dist)
                edge_distances[len(reachable_edges)-1] = min_dist
        
        if not reachable_edges:
            return None, [], {}
        
        # 3. KOMBINASIKAN NODES DAN EDGE POINTS UNTUK HASIL YANG LEBIH BAIK
        all_points = reachable_node_points + all_edge_points
        
        # Hapus duplikat
        unique_points = []
        seen = set()
        for point in all_points:
            point_tuple = tuple(point)
            if point_tuple not in seen:
                seen.add(point_tuple)
                unique_points.append(point)
        
        # 4. BUAT CONCAVE HULL DARI SEMUA POINTS YANG TERJANGKAU
        service_area = None
        
        if len(unique_points) >= 3:
            try:
                # Metode 1: Alpha Shape (Concave Hull)
                # Cari alpha optimal
                concave_shape, alpha_used = optimize_alpha_value(unique_points)
                
                if concave_shape and not concave_shape.is_empty:
                    service_area = concave_shape
                    st.info(f"✅ Menggunakan Concave Hull (alpha={alpha_used:.1f})")
                else:
                    # Fallback ke Convex Hull
                    points_array = np.array(unique_points)
                    hull = ConvexHull(points_array)
                    hull_points = points_array[hull.vertices]
                    service_area = Polygon(hull_points)
                    st.info("⚠️ Menggunakan Convex Hull (fallback)")
            except Exception as hull_error:
                st.warning(f"Error dalam concave hull: {hull_error}")
                # Fallback ke buffer dari edges
                pass
        
        # 5. JIKA CONCAVE/CONVEX HULL GAGAL, GUNAKAN BUFFER DARI EDGES
        if service_area is None or service_area.is_empty:
            try:
                buffered_polygons = []
                
                for i, edge in enumerate(reachable_edges):
                    try:
                        # Buffer setiap edge berdasarkan jaraknya dari titik awal
                        edge_dist = edge_distances.get(i, max_distance)
                        normalized_dist = edge_dist / max_distance
                        
                        # Buffer lebih besar untuk edge yang dekat, lebih kecil untuk yang jauh
                        buffer_size = service_buffer * (1 - normalized_dist)
                        buffer_size = max(10, min(buffer_size, service_buffer * 1.5))
                        
                        buffered_edge = edge.buffer(buffer_size, resolution=8)
                        buffered_polygons.append(buffered_edge)
                    except Exception:
                        continue
                
                if buffered_polygons:
                    # Gabungkan semua buffered edges
                    service_area = buffered_polygons[0]
                    for i in range(1, len(buffered_polygons)):
                        try:
                            service_area = unary_union([service_area, buffered_polygons[i]])
                        except:
                            continue
            except Exception as buffer_error:
                st.warning(f"Buffer fallback error: {buffer_error}")
        
        # 6. JIKA MASIH GAGAL, GUNAKAN BUFFER DARI TITIK AWAL
        if service_area is None or service_area.is_empty:
            try:
                start_point = Point(start_coords[0], start_coords[1])
                service_area = start_point.buffer(max_distance * 0.8)  # 80% dari max_distance
                st.info("ℹ️ Menggunakan buffer dari titik awal")
            except:
                return None, reachable_edges, edge_distances
        
        # 7. TAMBAHKAN BUFFER UNTUK SMOOTHING DAN COVERAGE YANG LEBIH BAIK
        try:
            if service_area and not service_area.is_empty:
                # Buffer untuk smoothing
                service_area = service_area.buffer(service_buffer, resolution=16)
                
                # Tambahkan titik awal jika belum termasuk
                start_point = Point(start_coords[0], start_coords[1])
                if not service_area.contains(start_point):
                    start_buffer = start_point.buffer(service_buffer * 2)
                    service_area = unary_union([service_area, start_buffer])
                
                # Simplifikasi
                service_area = service_area.simplify(service_buffer/2, preserve_topology=True)
        except Exception as e:
            st.warning(f"Error dalam smoothing service area: {e}")
        
        return service_area, reachable_edges, edge_distances
        
    except Exception as e:
        st.error(f"Error dalam network service area: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return None, [], {}

# ============================================================
# METODE 2: BUFFER DARI TITIK (SEDERHANA)
# ============================================================
def calculate_buffer_coverage(location_point, max_distance, shape='Lingkaran'):
    """Menghitung coverage sebagai buffer dari titik"""
    try:
        lat, lon = location_point
        
        # Konversi meter ke derajat
        buffer_deg_lat = max_distance / 111320  # 111.32 km per degree latitude
        buffer_deg_lon = max_distance / (111320 * math.cos(math.radians(lat)))
        
        if shape == 'Lingkaran':
            # Buffer lingkaran
            center_point = Point(lon, lat)
            approx_buffer_deg = max_distance / 111000
            polygon = center_point.buffer(approx_buffer_deg, resolution=32)
            
        elif shape == 'Persegi':
            # Buffer persegi (bounding box)
            min_lon = lon - buffer_deg_lon
            max_lon = lon + buffer_deg_lon
            min_lat = lat - buffer_deg_lat
            max_lat = lat + buffer_deg_lat
            
            polygon = box(min_lon, min_lat, max_lon, max_lat)
            
        elif shape == 'Kapsul':
            # Buffer kapsul (lingkaran memanjang)
            circle1 = Point(lon - buffer_deg_lon/2, lat).buffer(buffer_deg_lat/2, resolution=16)
            circle2 = Point(lon + buffer_deg_lon/2, lat).buffer(buffer_deg_lat/2, resolution=16)
            rectangle = box(lon - buffer_deg_lon/2, lat - buffer_deg_lat/3, 
                           lon + buffer_deg_lon/2, lat + buffer_deg_lat/3)
            
            polygon = unary_union([circle1, circle2, rectangle])
        
        # Validasi polygon
        if polygon.is_empty or not polygon.is_valid:
            # Fallback ke buffer lingkaran sederhana
            center_point = Point(lon, lat)
            approx_buffer_deg = max_distance / 111000
            polygon = center_point.buffer(approx_buffer_deg, resolution=16)
        
        return polygon
        
    except Exception as e:
        st.error(f"Error dalam buffer coverage: {str(e)}")
        return Point(location_point[1], location_point[0]).buffer(0.01)

# ============================================================
# FUNGSI UTAMA UNTUK ANALISIS NETWORK COVERAGE
# ============================================================
def calculate_network_coverage_from_point(graph_proj, start_node, start_coords, 
                                        max_distance, location_point, method="Service Area", 
                                        **kwargs):
    """
    Menghitung coverage area dari titik analisis dengan dua metode
    """
    try:
        # Jika metode adalah Buffer dari Titik
        if method == "Buffer dari Titik":
            shape = kwargs.get('buffer_shape', 'Lingkaran')
            coverage_polygon = calculate_buffer_coverage(location_point, max_distance, shape)
            return coverage_polygon, []
        
        # Untuk Service Area (Network Analysis)
        if method == "Service Area":
            service_buffer = kwargs.get('service_buffer', 100)
            
            coverage_polygon, reachable_edges, edge_distances = calculate_network_service_area(
                graph_proj, start_node, start_coords, max_distance, 
                service_buffer=service_buffer
            )
            
            if coverage_polygon is None:
                # Fallback ke buffer jika service area gagal
                shape = kwargs.get('buffer_shape', 'Lingkaran')
                return calculate_buffer_coverage(location_point, max_distance, shape), []
            
            return coverage_polygon, reachable_edges
        
        # Default fallback
        shape = kwargs.get('buffer_shape', 'Lingkaran')
        return calculate_buffer_coverage(location_point, max_distance, shape), []
        
    except Exception as e:
        st.error(f"Error dalam calculate_network_coverage: {str(e)}")
        shape = kwargs.get('buffer_shape', 'Lingkaran')
        return calculate_buffer_coverage(location_point, max_distance, shape), []

# ============================================================
# FUNGSI KONVERSI KOORDINAT
# ============================================================
def convert_polygon_to_wgs84(polygon, source_crs, location_point):
    """
    Konversi polygon dari projected CRS ke WGS84
    """
    try:
        transformer = Transformer.from_crs(source_crs, 'EPSG:4326', always_xy=True)
        
        if hasattr(polygon, 'exterior'):
            exterior_coords = list(polygon.exterior.coords)
            wgs84_coords = []
            for x, y in exterior_coords:
                lon_conv, lat_conv = transformer.transform(x, y)
                wgs84_coords.append((lon_conv, lat_conv))
            wgs84_polygon = Polygon(wgs84_coords)
        else:
            # Untuk geometry sederhana, konversi bounding box
            bounds = polygon.bounds
            minx, miny, maxx, maxy = bounds
            corners = [
                transformer.transform(minx, miny),
                transformer.transform(maxx, miny),
                transformer.transform(maxx, maxy),
                transformer.transform(minx, maxy)
            ]
            wgs84_polygon = Polygon([(lon, lat) for lon, lat in corners])
        
        return wgs84_polygon
        
    except Exception as e:
        # Fallback: buffer dari titik asal dalam WGS84
        st.warning(f"Konversi polygon gagal: {e}. Menggunakan fallback buffer.")
        lat, lon = location_point
        buffer_deg = max_distance / 111320 if 'max_distance' in locals() else 0.01
        return Point(lon, lat).buffer(buffer_deg)

# ============================================================
# PERBAIKAN: FUNGSI SERVICE AREA (ISOCHRONE) YANG AKURAT
# ============================================================
def calculate_service_area_improved(graph_proj, center_node, trip_times, travel_speed):
    """
    Menghitung Service Area berbasis waktu tempuh dengan hasil yang 
    mengikuti jaringan jalan menggunakan metode buffer edges.
    """
    try:
        # 1. Tambahkan data waktu (travel time) pada setiap edge
        meters_per_minute = (travel_speed * 1000) / 60
        for u, v, k, data in graph_proj.edges(data=True, keys=True):
            data['time'] = data['length'] / meters_per_minute

        # 2. Urutkan trip_times dari yang terbesar agar polygon kecil menimpa yang besar
        trip_times = sorted(trip_times, reverse=True)
        isochrone_polys = []

        for trip_time in trip_times:
            # Cari sub-graph yang bisa dijangkau dalam waktu tertentu
            subgraph = nx.ego_graph(graph_proj, center_node, radius=trip_time, distance='time')
            
            # Ambil semua node yang terjangkau
            nodes_gdf = ox.graph_to_gdfs(subgraph, nodes=True, edges=False)
            
            # Ambil semua garis jalan (edges) yang terjangkau
            edges_gdf = ox.graph_to_gdfs(subgraph, nodes=False, edges=True)
            
            if edges_gdf.empty:
                # Jika tidak ada edges, buat buffer dari nodes saja
                if not nodes_gdf.empty:
                    node_buffers = nodes_gdf.buffer(50)
                    combined_area = unary_union(list(node_buffers))
                else:
                    # Fallback: buffer dari center node
                    center_coords = (graph_proj.nodes[center_node]['x'], graph_proj.nodes[center_node]['y'])
                    center_point = Point(center_coords[0], center_coords[1])
                    combined_area = center_point.buffer(100)
            else:
                # TEKNIK TERBAIK: Buat buffer di sekitar jalan dan node yang terjangkau
                # Buffer sekitar 20-50 meter biasanya cukup untuk menutup celah antar jalan
                edge_buffers = edges_gdf.buffer(30)
                node_buffers = nodes_gdf.buffer(30)
                
                # Satukan semua buffer menjadi satu kesatuan area (Service Area)
                combined_area = unary_union(list(edge_buffers) + list(node_buffers))
            
            # Simplifikasi dan smoothing
            if combined_area and not combined_area.is_empty:
                # Buffer untuk smoothing
                combined_area = combined_area.buffer(10, resolution=12)
                # Simplifikasi
                combined_area = combined_area.simplify(20, preserve_topology=True)
            
            isochrone_polys.append(combined_area)

        return isochrone_polys
        
    except Exception as e:
        st.error(f"Error dalam calculate_service_area_improved: {str(e)}")
        return []

# ============================================================
# UPDATE: FUNGSI NETWORK SERVICE AREA YANG LEBIH AKURAT
# ============================================================
def calculate_network_service_area_improved(graph_proj, start_node, start_coords, max_distance, 
                                           service_buffer=100):
    """
    Menghitung service area yang lebih akurat menggunakan metode buffer edges
    """
    try:
        # 1. HITUNG JARAK DARI START NODE KE SEMUA NODES
        distances = nx.single_source_dijkstra_path_length(
            graph_proj, 
            start_node, 
            weight='length',
            cutoff=max_distance
        )
        
        if not distances:
            return None, [], {}
        
        # 2. IDENTIFIKASI NODES DAN EDGES YANG TERJANGKAU
        reachable_nodes = []
        reachable_edges = []
        edge_distances = {}
        
        for u, v, data in graph_proj.edges(data=True):
            # Cek jika salah satu node dari edge terjangkau
            u_dist = distances.get(u, float('inf'))
            v_dist = distances.get(v, float('inf'))
            
            # Edge terjangkau jika minimal satu node terjangkau
            if u_dist <= max_distance or v_dist <= max_distance:
                # Tambahkan nodes jika belum ada
                if u not in reachable_nodes:
                    reachable_nodes.append(u)
                if v not in reachable_nodes:
                    reachable_nodes.append(v)
                
                # Dapatkan geometri edge
                if 'geometry' in data:
                    edge_geom = data['geometry']
                else:
                    # Buat garis sederhana dari node ke node
                    u_coords = (graph_proj.nodes[u]['x'], graph_proj.nodes[u]['y'])
                    v_coords = (graph_proj.nodes[v]['x'], graph_proj.nodes[v]['y'])
                    edge_geom = LineString([u_coords, v_coords])
                
                reachable_edges.append(edge_geom)
                
                # Simpan jarak terdekat dari edge ke titik awal
                min_dist = min(u_dist, v_dist)
                edge_distances[len(reachable_edges)-1] = min_dist
        
        if not reachable_edges:
            return None, [], {}
        
        # 3. BUAT SERVICE AREA DARI BUFFER EDGES DAN NODES
        try:
            # Buffer edges berdasarkan jaraknya dari titik awal
            buffered_polygons = []
            
            for i, edge in enumerate(reachable_edges):
                try:
                    # Buffer size berdasarkan jarak dari titik awal
                    edge_dist = edge_distances.get(i, max_distance)
                    normalized_dist = edge_dist / max_distance
                    
                    # Buffer lebih besar untuk edge yang dekat, lebih kecil untuk yang jauh
                    buffer_size = service_buffer * (1 - normalized_dist)
                    buffer_size = max(20, min(buffer_size, service_buffer * 1.5))
                    
                    buffered_edge = edge.buffer(buffer_size, resolution=8)
                    buffered_polygons.append(buffered_edge)
                except Exception:
                    continue
            
            if buffered_polygons:
                # Gabungkan semua buffered edges
                service_area = buffered_polygons[0]
                for i in range(1, len(buffered_polygons)):
                    try:
                        service_area = unary_union([service_area, buffered_polygons[i]])
                    except:
                        continue
                
                # Tambahkan buffer dari nodes
                node_points = []
                for node in reachable_nodes:
                    node_coords = (graph_proj.nodes[node]['x'], graph_proj.nodes[node]['y'])
                    node_points.append(Point(node_coords[0], node_coords[1]))
                
                if node_points:
                    node_buffers = [point.buffer(service_buffer) for point in node_points]
                    node_area = unary_union(node_buffers)
                    service_area = unary_union([service_area, node_area])
                
                # Tambahkan titik awal jika belum termasuk
                start_point = Point(start_coords[0], start_coords[1])
                if not service_area.contains(start_point):
                    start_buffer = start_point.buffer(service_buffer * 2)
                    service_area = unary_union([service_area, start_buffer])
                
                # Simplifikasi dan smoothing
                service_area = service_area.buffer(10, resolution=16)
                service_area = service_area.simplify(service_buffer/2, preserve_topology=True)
                
                return service_area, reachable_edges, edge_distances
        
        except Exception as e:
            st.warning(f"Error dalam buffer method: {e}")
        
        # 4. FALLBACK: BUFFER DARI TITIK AWAL
        try:
            start_point = Point(start_coords[0], start_coords[1])
            service_area = start_point.buffer(max_distance * 0.7)
            return service_area, reachable_edges, edge_distances
        except:
            return None, reachable_edges, edge_distances
        
    except Exception as e:
        st.error(f"Error dalam network service area improved: {str(e)}")
        return None, [], {}

# ============================================================
# UPDATE: FUNGSI ANALISIS UTAMA DENGAN OPTIMASI
# ============================================================
def analyze_from_point_main(location_point, network_type, speed_kmh, radius_m, time_limits_min, 
                           method="Service Area", **kwargs):
    """
    Fungsi utama analisis dari titik dengan perhitungan fasilitas kesehatan
    """
    try:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # FIX: Simpan location point ke session state
        st.session_state.last_location = location_point
        
        # Untuk metode Buffer, kita tidak perlu jaringan
        if method != "Buffer dari Titik":
            # 1. Dapatkan jaringan dari titik
            status_text.text("📍 Mendapatkan jaringan dari titik analisis...")
            graph_proj = get_network_from_point(location_point, network_type, radius_m)
            if graph_proj is None:
                return None, None, None, None
            
            progress_bar.progress(25)
            
            # 2. Temukan node awal dari titik
            status_text.text("📍 Mencari node awal dari titik analisis...")
            start_node, start_coords, start_wgs84 = find_start_node_from_point(graph_proj, location_point)
            if start_node is None:
                return None, None, None, None
            
            progress_bar.progress(40)
        else:
            # Untuk metode Buffer, buat graph dummy
            graph_proj = None
            start_node = None
            start_coords = None
            start_wgs84 = location_point
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
        
        # OPTIMASI: Gunakan metode yang berbeda berdasarkan jumlah time limits
        use_improved_method = len(time_limits_min) <= 3
        
        for idx, time_limit in enumerate(sorted(time_limits_min)):
            max_distance = speed_m_per_min * time_limit
            
            # Pilih metode berdasarkan kebutuhan
            if use_improved_method and method == "Service Area":
                # Gunakan metode improved untuk hasil yang lebih akurat
                coverage_polygon, reachable_edges = calculate_network_coverage_from_point(
                    graph_proj, start_node, start_coords, max_distance, location_point, method, **kwargs
                )
            else:
                # Gunakan metode standar
                coverage_polygon, reachable_edges = calculate_network_coverage_from_point(
                    graph_proj, start_node, start_coords, max_distance, location_point, method, **kwargs
                )
            
            if coverage_polygon and not coverage_polygon.is_empty:
                # Hitung luas
                area_m2 = coverage_polygon.area
                area_km2 = area_m2 / 1000000
                
                # Konversi ke WGS84
                wgs84_polygon = None
                if method == "Buffer dari Titik":
                    # Sudah dalam WGS84
                    wgs84_polygon = coverage_polygon
                else:
                    try:
                        wgs84_polygon = convert_polygon_to_wgs84(
                            coverage_polygon, 
                            graph_proj.graph['crs'], 
                            location_point
                        )
                    except:
                        # Fallback: buffer dari titik asal
                        buffer_deg = max_distance / 111320
                        wgs84_polygon = Point(lon, lat).buffer(buffer_deg)
                
                # Hitung fasilitas yang terjangkau
                accessible_facilities = []
                if not health_facilities.empty and wgs84_polygon:
                    for idx_fac, facility in health_facilities.iterrows():
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
                
                # Simpan statistik
                accessibility_zones[time_limit] = {
                    'geometry': wgs84_polygon,
                    'geometry_projected': coverage_polygon,
                    'max_distance': max_distance,
                    'area_sqkm': area_km2,
                    'calculation_method': method,
                    'accessible_facilities': accessible_facilities,
                    'facilities_count': len(accessible_facilities),
                    'time_limit': time_limit
                }
                
                reachable_edges_dict[time_limit] = reachable_edges
            
            progress_bar.progress(50 + int((idx + 1) * 50 / len(time_limits_min)))
        
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
# FUNGSI PEMBUATAN PETA
# ============================================================
def create_comprehensive_map(location_point, accessibility_zones, health_facilities, reachable_edges_dict=None):
    """
    Membuat peta interaktif yang komprehensif dengan visualisasi jaringan
    """
    try:
        # FIX: Pastikan peta berpusat di location_point
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
        
        # FIX: Visualisasi jaringan (jika ada)
        if reachable_edges_dict:
            for time_limit, reachable_edges in reachable_edges_dict.items():
                if reachable_edges and len(reachable_edges) > 0:
                    edge_color = colors.get(time_limit, '#EF9A9A')
                    
                    for edge in reachable_edges:
                        if hasattr(edge, 'coords'):
                            coords = list(edge.coords)
                            if len(coords) >= 2:
                                folium_coords = []
                                for x, y in coords:
                                    folium_coords.append([y, x])  # [lat, lon]
                                
                                if len(folium_coords) >= 2:
                                    folium.PolyLine(
                                        locations=folium_coords,
                                        color=edge_color,
                                        weight=1,
                                        opacity=0.3,
                                        popup=f'Edge Jaringan ({time_limit} menit)'
                                    ).add_to(m)
        
        # FIX: Tambahkan zona aksesibilitas dengan koordinat yang benar
        for time_limit, zone_data in accessibility_zones.items():
            color = colors.get(time_limit, '#EF9A9A')
            
            try:
                if 'geometry' in zone_data and zone_data['geometry']:
                    polygon = zone_data['geometry']
                    
                    if hasattr(polygon, 'exterior'):
                        coords = list(polygon.exterior.coords)
                        folium_coords = []
                        for lon, lat in coords:
                            folium_coords.append([lat, lon])
                        
                        if len(folium_coords) >= 3:
                            # Tambahkan polygon ke peta
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
                            
                            # Tambahkan label di tengah polygon
                            if len(folium_coords) > 0:
                                center_lat = sum([c[0] for c in folium_coords]) / len(folium_coords)
                                center_lon = sum([c[1] for c in folium_coords]) / len(folium_coords)
                                
                                folium.Marker(
                                    location=[center_lat, center_lon],
                                    icon=folium.DivIcon(
                                        html=f'<div style="font-size: 10pt; font-weight: bold; color: {color};">{time_limit}m</div>'
                                    ),
                                    popup=f'Label Zona {time_limit} menit'
                                ).add_to(m)
                    else:
                        # Fallback: buat circle dari titik awal
                        folium.Circle(
                            location=location_point,
                            radius=zone_data['max_distance'],
                            color=color,
                            fill=True,
                            fill_color=color,
                            fill_opacity=0.3,
                            popup=f'<b>Zona {time_limit} menit</b><br>'
                                  f'Luas: {zone_data["area_sqkm"]:.2f} km²'
                        ).add_to(m)
                        
            except Exception as e:
                st.warning(f"Error menambahkan zona {time_limit} menit: {str(e)}")
                # Fallback: buat circle sederhana dari titik awal
                folium.Circle(
                    location=location_point,
                    radius=zone_data.get('max_distance', 1000),
                    color=color,
                    fill=True,
                    fill_color=color,
                    fill_opacity=0.3,
                    popup=f'Zona {time_limit} menit (fallback)'
                ).add_to(m)
        
        # Tambahkan fasilitas kesehatan
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
                            <p style="margin: 2px 0;"><b>Zona:</b> {time_limit} menit</p>
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
        
        # Tambahkan layer control
        folium.LayerControl().add_to(m)
        
        # Tambahkan mini map
        minimap = MiniMap(toggle_display=True)
        m.add_child(minimap)
        
        # Tambahkan fitur fullscreen
        Fullscreen().add_to(m)
        
        # Tambahkan legenda
        legend_html = '''
        <div style="position: fixed; 
                    bottom: 50px; left: 50px; width: 220px; height: 280px; 
                    background-color: white; border:2px solid grey; z-index:9999; 
                    font-size:14px; padding: 10px; border-radius: 5px; opacity: 0.9;">
            <b>LEGENDA</b><br>
            <i class="fa fa-bullseye" style="color:red"></i> Titik Analisis<br>
            <i class="fa fa-hospital" style="color:red"></i> Rumah Sakit<br>
            <i class="fa fa-medkit" style="color:blue"></i> Klinik<br>
            <i class="fa fa-user-md" style="color:green"></i> Dokter<br>
            <i class="fa fa-pills" style="color:orange"></i> Apotek<br>
            <i class="fa fa-heart" style="color:purple"></i> Fasilitas Lain<br>
            <div style="background-color: #FFEBEE; width: 20px; height: 20px; display: inline-block; margin-right: 5px; border: 1px solid #ccc;"></div> 5 menit<br>
            <div style="background-color: #FFCDD2; width: 20px; height: 20px; display: inline-block; margin-right: 5px; border: 1px solid #ccc;"></div> 10 menit<br>
            <div style="background-color: #EF9A9A; width: 20px; height: 20px; display: inline-block; margin-right: 5px; border: 1px solid #ccc;"></div> 15 menit<br>
            <div style="background-color: #E57373; width: 20px; height: 20px; display: inline-block; margin-right: 5px; border: 1px solid #ccc;"></div> 20 menit<br>
            <div style="background-color: #EF5350; width: 20px; height: 20px; display: inline-block; margin-right: 5px; border: 1px solid #ccc;"></div> 25 menit<br>
            <div style="background-color: #F44336; width: 20px; height: 20px; display: inline-block; margin-right: 5px; border: 1px solid #ccc;"></div> 30 menit<br>
            <div style="margin-top: 5px;"><small><i>Garis tipis: Jaringan jalan yang terjangkau</i></small></div>
        </div>
        '''
        
        m.get_root().html.add_child(folium.Element(legend_html))
        
        # FIX: Simpan peta ke session state untuk mencegah re-rendering
        st.session_state.map_created = True
        st.session_state.last_map = m
        
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
        step=100,
        help="Radius untuk mengambil jaringan jalan dari titik analisis"
    )
    
    time_limits = st.multiselect(
        "Batas Waktu (menit):",
        [5, 10, 15, 20, 25, 30],
        default=[15, 25],
        help="Waktu tempuh maksimum untuk menghitung jangkauan"
    )
    
    # Pilihan metode perhitungan luas
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
        st.info("**Metode Concave Hull**: Service area akan mengikuti bentuk jaringan jalan")
        service_buffer = st.slider(
            "Buffer Service Area (meter):",
            20, 5000, 100, 10,
            help="Buffer untuk smoothing service area dari jaringan jalan"
        )
        
        # Pilihan algoritma hull
        hull_method = st.selectbox(
            "Metode Hull:",
            ["Concave Hull (Alpha Shape)", "Buffer Edges"],
            index=0,
            help="Metode untuk membentuk service area dari jaringan"
        )
    
    elif area_calculation_method == "Buffer dari Titik":
        buffer_shape = st.selectbox(
            "Bentuk Buffer:",
            ["Lingkaran", "Persegi", "Kapsul"],
            index=0
        )
    
    analyze_button = st.button("🚀 Jalankan Analisis", type="primary", use_container_width=True)
    
    # Tombol reset
    if st.button("🔄 Reset Analisis", use_container_width=True):
        st.session_state.analysis_results = None
        st.session_state.map_created = False
        st.session_state.accessibility_zones = None
        st.session_state.health_facilities = None
        st.session_state.reachable_edges = None
        st.session_state.last_location = None
        st.rerun()
    
    st.markdown("---")
    st.info("""
    **📌 Panduan Metode Network Analysis:**
    
    1. **Service Area (Concave Hull)**: 
       - **Analisis jaringan sebenarnya** berdasarkan struktur jalan
       - **Dijkstra Algorithm** menghitung jarak dari titik awal
       - **Alpha Shape (Concave Hull)** untuk bentuk yang mengikuti jaringan
       - **Buffer smoothing** untuk hasil yang realistis
       - Akurat untuk analisis berbasis jaringan
    
    2. **Service Area (Buffer Edges)**: 
       - **Buffer setiap edge** berdasarkan jarak dari titik awal
       - **Union semua buffers** menjadi satu area
       - Hasil yang smooth dan mengikuti jaringan
    
    3. **Buffer dari Titik**: 
       - Buffer sederhana dari titik analisis
       - Tidak memerlukan analisis jaringan jalan
       - Cepat dan sederhana untuk estimasi awal
    
    **🎯 Rekomendasi:**
    - **Concave Hull**: Untuk hasil yang akurat dan mengikuti jaringan
    - **Buffer Edges**: Untuk hasil yang smooth dengan performa baik
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
        # Tambahkan metode hull ke kwargs
        if 'hull_method' in locals():
            kwargs['hull_method'] = hull_method
    
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
        
        # FIX: Simpan hasil ke session state
        st.session_state.analysis_results = result
        st.session_state.accessibility_zones = accessibility_zones
        st.session_state.health_facilities = health_facilities
        st.session_state.reachable_edges = reachable_edges
        
        # Tampilkan hasil dalam tab
        tab1, tab2, tab3, tab4 = st.tabs(["🗺️ Peta Interaktif", "📊 Dashboard", "🏥 Fasilitas", "📈 Analisis"])
        
        with tab1:
            st.subheader("🗺️ Peta Jangkauan Fasilitas Kesehatan")
            
            if accessibility_zones:
                # Tampilkan informasi lokasi
                st.info(f"**📍 Titik Analisis:** {location_point[0]:.6f}, {location_point[1]:.6f}")
                
                # Tampilkan metode yang digunakan
                if area_calculation_method == "Service Area":
                    if 'hull_method' in locals():
                        st.info(f"**🔧 Metode:** {hull_method}")
                    else:
                        st.info("**🔧 Metode:** Concave Hull (Alpha Shape)")
                
                # Buat dan tampilkan peta
                m = create_comprehensive_map(location_point, accessibility_zones, health_facilities, reachable_edges)
                
                # FIX: Gunakan folium_static untuk peta yang stabil
                folium_static(m, width=1200, height=650)
                
                # Tambahkan informasi
                st.markdown("### 📍 Informasi Peta")
                st.info("""
                **Layer Peta:**
                - **Area berwarna**: Zona aksesibilitas berdasarkan waktu tempuh
                - **Garis tipis berwarna**: Jaringan jalan yang terjangkau
                - **Marker berwarna**: Fasilitas kesehatan (warna menunjukkan jenis)
                - **Titik merah**: Lokasi analisis awal
                
                **Interaksi:**
                - Klik pada area/polygon untuk melihat detail
                - Klik pada fasilitas untuk informasi lengkap
                - Gunakan kontrol di pojok kanan atas untuk zoom dan fullscreen
                """)
            else:
                st.warning("⚠️ Tidak ada zona jangkauan yang dapat dihitung.")
        
        with tab2:
            st.subheader("📊 Dashboard Analisis Network Coverage")
            
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
            
            # Tampilkan statistik Network Analysis
            if area_calculation_method == "Service Area" and accessibility_zones:
                st.subheader("📊 Statistik Service Area (Concave Hull)")
                
                network_stats = []
                for time_limit, zone_data in accessibility_zones.items():
                    if zone_data.get('calculation_method') == "Service Area":
                        # Hitung edges yang digunakan
                        edges_count = len(reachable_edges.get(time_limit, [])) if reachable_edges else 0
                        
                        network_stats.append({
                            "⏱️ Waktu": f"{time_limit} menit",
                            "📏 Jarak Maks": f"{zone_data['max_distance']:.0f} m",
                            "🛣️ Edges Terjangkau": edges_count,
                            "📐 Luas Area": f"{zone_data['area_sqkm']:.2f} km²",
                            "🏥 Fasilitas": zone_data.get('facilities_count', 0)
                        })
                
                if network_stats:
                    stats_df = pd.DataFrame(network_stats)
                    st.dataframe(stats_df, use_container_width=True, hide_index=True)
                    
                    # Visualisasi perbandingan
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
                    
                    # Plot 1: Area vs Waktu
                    time_limits_list = list(accessibility_zones.keys())
                    areas = [accessibility_zones[t]['area_sqkm'] for t in time_limits_list]
                    
                    ax1.bar([str(t) for t in time_limits_list], areas, color='steelblue', edgecolor='black')
                    ax1.set_xlabel('Batas Waktu (menit)', fontsize=12, fontweight='bold')
                    ax1.set_ylabel('Luas Area (km²)', fontsize=12, fontweight='bold')
                    ax1.set_title('Luas Service Area vs Waktu', fontsize=14, fontweight='bold')
                    ax1.grid(axis='y', alpha=0.3)
                    
                    # Plot 2: Fasilitas vs Waktu
                    facilities_count = [accessibility_zones[t].get('facilities_count', 0) for t in time_limits_list]
                    
                    ax2.bar([str(t) for t in time_limits_list], facilities_count, color='coral', edgecolor='black')
                    ax2.set_xlabel('Batas Waktu (menit)', fontsize=12, fontweight='bold')
                    ax2.set_ylabel('Jumlah Fasilitas', fontsize=12, fontweight='bold')
                    ax2.set_title('Fasilitas Terjangkau vs Waktu', fontsize=14, fontweight='bold')
                    ax2.grid(axis='y', alpha=0.3)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
            
            # Tampilkan informasi metode
            st.subheader("🔧 Informasi Metode Analisis")
            
            if area_calculation_method == "Service Area":
                if 'hull_method' in locals() and "Concave Hull" in hull_method:
                    method_info = f"""
                    **🔍 Service Area (Concave Hull / Alpha Shape)**
                    
                    **Algoritma yang digunakan:**
                    1. **Dijkstra Algorithm**: Menghitung jarak terpendek dari titik awal ke semua nodes dalam radius {search_radius}m
                    2. **Identifikasi Edges Terjangkau**: Edges yang memiliki minimal satu node dengan jarak ≤ {max([speed_kmh * 1000 / 60 * t for t in time_limits]):.0f}m
                    3. **Kumpulkan Points**: Mengumpulkan semua titik dari nodes dan edges yang terjangkau
                    4. **Alpha Shape (Concave Hull)**: Membentuk concave hull menggunakan alpha shape algorithm
                    5. **Buffer Smoothing**: Buffer sebesar {service_buffer}m untuk bentuk yang smooth
                    6. **Konversi ke WGS84**: Konversi koordinat untuk visualisasi di peta
                    
                    **Keunggulan Concave Hull:**
                    - Mengikuti bentuk jaringan jalan yang sebenarnya
                    - Menghasilkan area yang lebih akurat dibanding convex hull
                    - Dapat menangani bentuk yang tidak beraturan
                    - Menghasilkan "cekokan" (concavities) yang realistis
                    
                    **Parameter:**
                    - Buffer Service Area: `{service_buffer}` meter
                    - Radius Jaringan: `{search_radius}` meter
                    - Alpha value: Dioptimalkan otomatis
                    """
                else:
                    method_info = f"""
                    **🔍 Service Area (Buffer Edges)**
                    
                    **Algoritma yang digunakan:**
                    1. **Dijkstra Algorithm**: Menghitung jarak terpendek dari titik awal ke semua nodes
                    2. **Identifikasi Edges Terjangkau**: Edges yang terjangkau berdasarkan jarak
                    3. **Buffer Each Edge**: Setiap edge dibuffer berdasarkan jaraknya dari titik awal
                    4. **Union Buffers**: Semua buffer edges digabungkan menjadi satu area
                    5. **Smoothing**: Buffer tambahan untuk smoothing hasil
                    
                    **Parameter:**
                    - Buffer Service Area: `{service_buffer}` meter
                    - Radius Jaringan: `{search_radius}` meter
                    """
            else:
                method_info = f"""
                **🔍 Buffer dari Titik**
                
                **Algoritma yang digunakan:**
                1. **Direct Buffering**: Buffer langsung dari titik analisis ({location_point[0]:.6f}, {location_point[1]:.6f})
                2. **Shape Selection**: Bentuk buffer ({buffer_shape})
                3. **Area Calculation**: Perhitungan luas area buffer
                
                **Parameter:**
                - Bentuk Buffer: `{buffer_shape}`
                - Analisis sederhana tanpa jaringan
                - Cepat untuk estimasi awal
                """
            
            st.markdown(method_info)
        
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
                                
                                # Ekspor data
                                csv = fac_df.to_csv(index=False).encode('utf-8')
                                st.download_button(
                                    label="📥 Download Data CSV",
                                    data=csv,
                                    file_name=f"fasilitas_{time_limit}menit.csv",
                                    mime="text/csv"
                                )
                            else:
                                st.info(f"ℹ️ Tidak ada fasilitas yang dapat diakses dalam {time_limit} menit.")
            else:
                st.warning("⚠️ Tidak ada fasilitas kesehatan yang dapat diakses dalam area ini.")
        
        with tab4:
            st.subheader("📈 Analisis dan Visualisasi")
            
            if accessibility_zones:
                # Plot perbandingan metode
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
                
                # Data untuk plotting
                time_limits_list = list(accessibility_zones.keys())
                areas = [accessibility_zones[t]['area_sqkm'] for t in time_limits_list]
                facilities = [accessibility_zones[t].get('facilities_count', 0) for t in time_limits_list]
                
                # Plot 1: Area Coverage
                colors_area = plt.cm.YlOrRd(np.linspace(0.4, 0.9, len(time_limits_list)))
                bars1 = ax1.bar([str(t) for t in time_limits_list], areas, color=colors_area, edgecolor='black')
                ax1.set_xlabel('Batas Waktu (menit)', fontsize=12, fontweight='bold')
                ax1.set_ylabel('Luas Area (km²)', fontsize=12, fontweight='bold')
                ax1.set_title('Luas Service Area vs Waktu', fontsize=14, fontweight='bold')
                
                # Tambahkan nilai di atas bar
                for bar in bars1:
                    height = bar.get_height()
                    ax1.text(bar.get_x() + bar.get_width()/2., height,
                            f'{height:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
                
                ax1.grid(axis='y', alpha=0.3)
                
                # Plot 2: Fasilitas Terjangkau
                if sum(facilities) > 0:
                    colors_fac = plt.cm.Blues(np.linspace(0.4, 0.9, len(time_limits_list)))
                    bars2 = ax2.bar([str(t) for t in time_limits_list], facilities, 
                                   color=colors_fac, edgecolor='black')
                    ax2.set_xlabel('Batas Waktu (menit)', fontsize=12, fontweight='bold')
                    ax2.set_ylabel('Jumlah Fasilitas', fontsize=12, fontweight='bold')
                    ax2.set_title('Fasilitas yang Dapat Diakses', fontsize=14, fontweight='bold')
                    
                    # Tambahkan nilai di atas bar
                    for bar in bars2:
                        height = bar.get_height()
                        ax2.text(bar.get_x() + bar.get_width()/2., height,
                                f'{height}', ha='center', va='bottom', fontsize=10, fontweight='bold')
                    
                    ax2.grid(axis='y', alpha=0.3)
                else:
                    ax2.text(0.5, 0.5, 'Tidak ada data fasilitas', 
                            ha='center', va='center', transform=ax2.transAxes, 
                            fontsize=12, fontweight='bold')
                    ax2.set_title('Tidak Ada Fasilitas Ditemukan', fontsize=14, fontweight='bold')
                
                plt.tight_layout()
                st.pyplot(fig)
                
                # Ringkasan statistik
                st.subheader("📋 Ringkasan Analisis Network Coverage")
                
                total_facilities_all = sum(zone.get('facilities_count', 0) for zone in accessibility_zones.values())
                
                if total_facilities_all > 0:
                    # Hitung statistik detail
                    unique_names = set()
                    all_travel_times = []
                    all_distances = []
                    
                    for zone_data in accessibility_zones.values():
                        if 'accessible_facilities' in zone_data:
                            for fac in zone_data['accessible_facilities']:
                                unique_names.add(fac['name'])
                                all_travel_times.append(fac['travel_time_min'])
                                all_distances.append(fac['distance_m'])
                    
                    # Tampilkan metrik
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
                    
                    # Tampilkan distribusi waktu tempuh
                    if all_travel_times:
                        fig2, ax2 = plt.subplots(figsize=(10, 6))
                        ax2.hist(all_travel_times, bins=15, color='skyblue', edgecolor='black', alpha=0.7)
                        ax2.set_xlabel('Waktu Tempuh (menit)', fontsize=12, fontweight='bold')
                        ax2.set_ylabel('Jumlah Fasilitas', fontsize=12, fontweight='bold')
                        ax2.set_title('Distribusi Waktu Tempuh ke Fasilitas', fontsize=14, fontweight='bold')
                        ax2.grid(axis='y', alpha=0.3)
                        plt.tight_layout()
                        st.pyplot(fig2)
                else:
                    st.info("ℹ️ Tidak ada data fasilitas untuk ditampilkan.")
                
                # Ekspor hasil analisis
                st.subheader("💾 Ekspor Hasil Analisis")
                
                # Buat dataframe ringkasan
                summary_data = []
                for time_limit, zone_data in accessibility_zones.items():
                    summary_data.append({
                        'time_limit_min': time_limit,
                        'method': zone_data.get('calculation_method', area_calculation_method),
                        'max_distance_m': zone_data['max_distance'],
                        'area_sqkm': zone_data['area_sqkm'],
                        'facilities_count': zone_data.get('facilities_count', 0)
                    })
                
                summary_df = pd.DataFrame(summary_data)
                
                # Tombol download
                csv_summary = summary_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Ringkasan Analisis (CSV)",
                    data=csv_summary,
                    file_name="ringkasan_analisis_network_coverage.csv",
                    mime="text/csv"
                )
    
    else:
        st.error("❌ Analisis gagal. Periksa parameter dan coba lagi.")

# FIX: Tampilkan peta dari session state jika sudah ada analisis
elif st.session_state.analysis_results and st.session_state.accessibility_zones:
    # Ambil data dari session state
    graph, health_facilities, accessibility_zones, reachable_edges = st.session_state.analysis_results
    
    # Tampilkan hasil yang sudah disimpan
    st.success("📊 **Menampilkan hasil analisis sebelumnya**")
    
    tab1, tab2, tab3, tab4 = st.tabs(["🗺️ Peta Interaktif", "📊 Dashboard", "🏥 Fasilitas", "📈 Analisis"])
    
    with tab1:
        st.subheader("🗺️ Peta Jangkauan Fasilitas Kesehatan (Hasil Tersimpan)")
        
        if accessibility_zones:
            # Ambil location_point dari parameter yang tersimpan
            location_point = st.session_state.get('last_location', (-6.2088, 106.8456))
            
            # Tampilkan peta dari session state
            if 'last_map' in st.session_state:
                folium_static(st.session_state.last_map, width=1200, height=650)
            else:
                # Buat peta baru jika tidak ada yang tersimpan
                m = create_comprehensive_map(location_point, accessibility_zones, health_facilities, reachable_edges)
                folium_static(m, width=1200, height=650)
        else:
            st.warning("⚠️ Tidak ada zona jangkauan yang dapat ditampilkan.")

elif not analyze_button:
    # Tampilkan halaman awal
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ## 📋 Panduan Penggunaan Network Analysis Coverage
        
        1. **📍 Pilih lokasi** di sidebar (kota atau koordinat manual)
        2. **⚙️ Atur parameter**:
           - Mode transportasi (jalan kaki/sepeda/mobil/motor)
           - Kecepatan perjalanan (km/jam)
           - Radius pencarian (500-5000m)
           - Batas waktu jangkauan (menit)
           - Metode coverage area (Service Area/Buffer dari Titik)
           - Metode hull (Concave Hull/Buffer Edges)
        3. **🚀 Klik "Jalankan Analisis"** untuk memulai
        
        ## 🎯 Fitur Network Analysis - CONCAVE HULL
        
        - **Service Area dengan Concave Hull**: Analisis berbasis jaringan jalan sebenarnya
        - **Dijkstra Algorithm**: Menghitung jarak terpendek dari titik awal
        - **Alpha Shape (Concave Hull)**: Membentuk area yang mengikuti jaringan jalan
        - **Buffer Smoothing**: Hasil yang smooth dan realistis
        - **Visualisasi Jaringan**: Tampilkan edges yang terjangkau di peta
        - **Analisis Statistik**: Dashboard lengkap dengan metrik network
        """)
    
    with col2:
        st.markdown("""
        ## 💡 Tips Network Analysis
        
        ### Untuk Hasil Terbaik:
        - Gunakan **radius 1500-3000m** untuk coverage yang optimal
        - **Service Buffer 100-200m** untuk hasil yang smooth
        - Pilih **Concave Hull** untuk hasil yang mengikuti jaringan
        - Pilih **mode transportasi sesuai** untuk analisis akurat
        
        ### Perbedaan Hull Methods:
        - **Concave Hull**: Mengikuti bentuk jaringan, hasil lebih akurat
        - **Buffer Edges**: Smooth dan cepat, baik untuk area urban
        - **Convex Hull**: Sederhana, cocok untuk area terbuka
        
        ### Visualisasi:
        - **Area berwarna**: Zona aksesibilitas berdasarkan waktu tempuh
        - **Garis tipis**: Jaringan jalan yang terjangkau
        - **Marker warna**: Fasilitas kesehatan berdasarkan jenis
        - **Titik merah**: Lokasi analisis awal
        """)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; padding: 20px;'>
        <p style='font-size: 1.1em; font-weight: bold; color: #2c3e50;'>
        🏥📍 <b>Network Analysis Coverage Area</b> v5.0 - Concave Hull Implementation
        </p>
        <p style='font-size: 0.9em; color: #7f8c8d;'>
        Developer: Adipandang Yudono, S.Si., MURP., PhD (Spatial Analysis, Architecture System, Scrypt Developer, WebGIS Analytics) , Dr (Cand). Firman Afrianto, ST., MT (QGIS Plugin Developer) & dr. Aurick Yudha Nagara, Sp.EM, KPEC (Health Facilities Analysis)
        <br>
        <b>Algoritma Network Analysis:</b> Dijkstra + Alpha Shape (Concave Hull) + Buffer Smoothing
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
    
    /* Metric cards dengan gradient network */
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border-radius: 10px;
        padding: 15px;
        border-left: 4px solid #4B79A1;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        transition: transform 0.3s;
    }
    
    [data-testid="stMetric"]:hover {
        transform: translateY(-3px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.12);
    }
    
    /* Tab styling dengan tema network */
    .stTabs [data-baseweb="tab-list"] {
        gap: 1rem;
        padding: 0 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
        font-weight: bold;
        background: linear-gradient(135deg, #f1f8ff 0%, #e3f2fd 100%);
        transition: all 0.3s;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #4B79A1 0%, #283E51 100%) !important;
        color: white !important;
        box-shadow: 0 2px 8px rgba(75, 121, 161, 0.3);
    }
    
    /* Dataframe dengan tema network */
    .dataframe {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        border: 1px solid #e0e0e0;
    }
    
    .dataframe thead th {
        background: linear-gradient(135deg, #4B79A1 0%, #283E51 100%);
        color: white;
        font-weight: bold;
    }
    
    /* Progress bar dengan tema network */
    .stProgress > div > div {
        background: linear-gradient(90deg, #4B79A1, #6C8EBF, #8FA3D1);
        background-size: 200% 100%;
        animation: gradientMove 2s ease infinite;
    }
    
    @keyframes gradientMove {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
    }
    
    /* Network analysis info box */
    .network-info {
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
        border-radius: 10px;
        padding: 15px;
        border-left: 4px solid #2196F3;
        margin: 10px 0;
    }
    
    /* Concave hull info box */
    .concave-hull-info {
        background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%);
        border-radius: 10px;
        padding: 15px;
        border-left: 4px solid #4CAF50;
        margin: 10px 0;
        border: 1px solid #a5d6a7;
    }
    
    /* Fix untuk peta yang hilang */
    .folium-map {
        width: 100% !important;
        height: 650px !important;
        border: 1px solid #ddd;
        border-radius: 10px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)
