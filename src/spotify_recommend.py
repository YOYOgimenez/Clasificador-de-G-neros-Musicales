import os
import requests  # Importante: para bajar el audio directo de Spotify
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import yt_dlp

def configurar_spotify():
    """Configura la conexión oficial con la API de Spotify usando Secrets de Streamlit"""
    client_id = os.getenv('SPOTIPY_CLIENT_ID')
    client_secret = os.getenv('SPOTIPY_CLIENT_SECRET')
    
    if not client_id or not client_secret:
        print("⚠️ Error: No se encontraron las credenciales de Spotify en los Secrets.")
        return None
        
    auth_manager = SpotifyClientCredentials(
        client_id=client_id,
        client_secret=client_secret
    )
    return spotipy.Spotify(auth_manager=auth_manager)

def descargar_audio_nandi(track_url):
    """
    Nandi AI: Intenta bajar la preview oficial de Spotify primero.
    Si no existe, usa yt-dlp como respaldo.
    """
    try:
        sp = configurar_spotify()
        if not sp:
            return None, None, None
        
        # 1. Obtener información de la canción desde Spotify
        track_id = track_url.split('track/')[-1].split('?')[0]
        track_info = sp.track(track_id)
        nombre = track_info['name']
        artista = track_info['artists'][0]['name']
        preview_url = track_info.get('preview_url') # Link directo al mp3 de 30s

        # 2. Configurar ruta temporal
        path_base = 'data/temp/test_nandi_audio.mp3'
        if not os.path.exists('data/temp'):
            os.makedirs('data/temp', exist_ok=True)

        # 3. LÓGICA DE DESCARGA:
        # PRIORIDAD A: Preview oficial de Spotify (Inmune a bloqueos de YouTube)
        if preview_url:
            print(f"🚀 Bajando preview oficial de Spotify: {nombre}...")
            r = requests.get(preview_url, timeout=10)
            if r.status_code == 200:
                with open(path_base, 'wb') as f:
                    f.write(r.content)
                return path_base, nombre, artista

        # PRIORIDAD B: Búsqueda en YouTube (Si Spotify no tiene preview)
        print(f"⚠️ Sin preview oficial, intentando búsqueda alternativa en YT...")
        busqueda = f"{nombre} {artista} lyrics"
        ydl_opts = {
            'format': 'bestaudio/best',
            'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
            'outtmpl': 'data/temp/test_nandi_audio', # yt-dlp le agrega el .mp3 solo
            'quiet': True,
            'no_warnings': True,
            'default_search': 'ytsearch1',
            'nocheckcertificate': True,
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([f"ytsearch1:{busqueda}"])

        return f"data/temp/test_nandi_audio.mp3", nombre, artista

    except Exception as e:
        print(f"❌ Error en la descarga: {e}")
        return None, None, None

def obtener_playlist_por_genero(genero):
    """Busca una playlist en Spotify para recomendar según el género detectado"""
    try:
        sp = configurar_spotify()
        if not sp: return None
        
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