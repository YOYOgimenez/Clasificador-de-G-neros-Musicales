import os
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import yt_dlp

def configurar_spotify():
    """Configura la conexión oficial con la API de Spotify usando Secrets de Streamlit"""
    # Streamlit Cloud lee estos datos de la pestaña 'Secrets' automáticamente
    client_id = os.getenv('SPOTIPY_CLIENT_ID')
    client_secret = os.getenv('SPOTIPY_CLIENT_SECRET')
    
    if not client_id or not client_secret:
        print("⚠️ Error: No se encontraron las credenciales de Spotify en los Secrets de Streamlit.")
        return None
        
    auth_manager = SpotifyClientCredentials(
        client_id=client_id,
        client_secret=client_secret
    )
    return spotipy.Spotify(auth_manager=auth_manager)

def descargar_audio_nandi(track_url):
    """
    Nandi AI: Obtiene info de Spotify y descarga el audio real 
    usando yt-dlp con User-Agent para evitar Error 403 y videos no disponibles.
    """
    try:
        sp = configurar_spotify()
        if not sp:
            return None, None, None
        
        # 1. Obtener información de la canción
        try:
            # Limpiamos el ID del track
            track_id = track_url.split('track/')[-1].split('?')[0]
            track_info = sp.track(track_id)
            nombre = track_info['name']
            artista = track_info['artists'][0]['name']
            busqueda = f"{nombre} {artista} audio official"
        except Exception as e:
            print(f"⚠️ Spotify API error: {e}. Usando link como búsqueda directa.")
            nombre, artista = "Audio", "Desconocido"
            busqueda = track_url

        # 2. Carpeta temporal (compatible con Linux/Streamlit)
        path_base = 'data/temp/test_nandi_audio'
        if not os.path.exists('data/temp'):
            os.makedirs('data/temp', exist_ok=True)

        # 3. Configuración de yt-dlp con "Disfraz" y búsqueda inteligente
        ydl_opts = {
            'format': 'bestaudio/best',
            # User-Agent para que parezca un navegador real
            'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
            'outtmpl': path_base, 
            'quiet': True,
            'noplaylist': True,
            'nocheckcertificate': True,
            'default_search': 'ytsearch1', # Busca y elige el primer resultado
            'ignoreerrors': True,          # Si un video está caído, no frena todo el script
            'no_warnings': True,
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # Buscamos la canción en YouTube
            error_code = ydl.download([f"ytsearch1:{busqueda}"])
            if error_code != 0 and not os.path.exists(f"{path_base}.mp3"):
                 raise Exception("No se pudo descargar ningún video disponible.")

        return f"{path_base}.mp3", nombre, artista

    except Exception as e:
        print(f"❌ Error crítico en Nandi AI: {e}")
        return None, None, None

def obtener_playlist_por_genero(genero):
    """Busca una playlist en Spotify para recomendar según el género detectado"""
    try:
        sp = configurar_spotify()
        if not sp: return None
        
        # Buscamos playlist por género
        results = sp.search(q=f"genre:{genero}", type='playlist', limit=1)
        if results['playlists']['items']:
            p = results['playlists']['items'][0]
            return {
                "nombre": p['name'], 
                "url": p['external_urls']['spotify']
            }
        return None
    except:
        return None