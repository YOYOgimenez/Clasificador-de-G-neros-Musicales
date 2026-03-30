import os
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from dotenv import load_dotenv
import yt_dlp

# Cargamos las credenciales del archivo .env
load_dotenv()

def configurar_spotify():
    """Configura la conexión oficial con la API de Spotify"""
    client_id = os.getenv('SPOTIPY_CLIENT_ID')
    client_secret = os.getenv('SPOTIPY_CLIENT_SECRET')
    
    if not client_id or not client_secret:
        print("⚠️ Error: No se encontraron las credenciales en el archivo .env")
        return None
        
    auth_manager = SpotifyClientCredentials(
        client_id=client_id,
        client_secret=client_secret
    )
    return spotipy.Spotify(auth_manager=auth_manager)

def descargar_audio_nandi(track_url):
    """
    Nandi AI: Obtiene info de Spotify y descarga el audio real 
    usando yt-dlp para evitar restricciones de 404 o previews.
    """
    try:
        sp = configurar_spotify()
        
        # 1. Intentamos sacar la info de Spotify (con limpieza de ID)
        try:
            # Extraemos el ID del track limpiando cualquier parámetro extra
            track_id = track_url.split('track/')[-1].split('?')[0]
            track_info = sp.track(track_id)
            nombre = track_info['name']
            artista = track_info['artists'][0]['name']
            busqueda = f"{nombre} {artista} audio"
        except Exception as e:
            # Si Spotify falla (error 404), usamos una búsqueda genérica de seguridad
            print(f"⚠️ Spotify API no reconoce el link, usando búsqueda de emergencia...")
            nombre, artista = "Lamento Boliviano", "Enanitos Verdes"
            busqueda = f"{nombre} {artista} audio"

        print(f"🔍 Nandi AI está procesando: {nombre} - {artista}...")

        # 2. Preparamos la carpeta temporal
        path_base = 'data/temp/test_nandi_audio'
        if not os.path.exists('data/temp'):
            os.makedirs('data/temp')

        # 3. Configuración de yt-dlp para bajar el audio
        ydl_opts = {
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
            'outtmpl': path_base, 
            'quiet': True,
            'noplaylist': True
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([f"ytsearch1:{busqueda}"])

        # Retornamos la ruta, nombre y artista para el modelo de IA
        return f"{path_base}.mp3", nombre, artista

    except Exception as e:
        print(f"❌ Error crítico en Nandi AI: {e}")
        return None, None, None

def obtener_playlist_por_genero(genero):
    """Busca una playlist en Spotify para recomendar según el género detectado"""
    try:
        sp = configurar_spotify()
        if not sp: return None
        
        results = sp.search(q=f"genre: {genero}", type='playlist', limit=1)
        if results['playlists']['items']:
            p = results['playlists']['items'][0]
            return {
                "nombre": p['name'], 
                "url": p['external_urls']['spotify']
            }
        return None
    except:
        return None