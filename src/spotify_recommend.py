import os
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import yt_dlp

def configurar_spotify():
    """Configura la conexión oficial con la API de Spotify usando Secrets de Streamlit"""
    # En Streamlit Cloud, os.getenv saca los datos de la pestaña 'Secrets' automáticamente
    client_id = os.getenv('SPOTIPY_CLIENT_ID')
    client_secret = os.getenv('SPOTIPY_CLIENT_SECRET')
    
    if not client_id or not client_secret:
        # Quitamos la mención al archivo .env para no confundir en la nube
        print("⚠️ Error: No se encontraron las credenciales de Spotify en los Secrets.")
        return None
        
    auth_manager = SpotifyClientCredentials(
        client_id=client_id,
        client_secret=client_secret
    )
    return spotipy.Spotify(auth_manager=auth_manager)

def descargar_audio_nandi(track_url):
    """
    Nandi AI: Obtiene info de Spotify y descarga el audio real 
    usando yt-dlp con 'disfraz' para evitar el Error 403.
    """
    try:
        sp = configurar_spotify()
        if not sp:
            return None, None, None
        
        # 1. Intentamos sacar la info de Spotify
        try:
            track_id = track_url.split('track/')[-1].split('?')[0]
            track_info = sp.track(track_id)
            nombre = track_info['name']
            artista = track_info['artists'][0]['name']
            busqueda = f"{nombre} {artista} audio"
        except Exception as e:
            print(f"⚠️ Spotify API no reconoce el link, usando búsqueda de emergencia...")
            nombre, artista = "Unknown", "Artist"
            busqueda = track_url # Usamos el link directo si la API falla

        # 2. Preparamos la carpeta temporal (Path compatible con Linux/Streamlit)
        path_base = 'data/temp/test_nandi_audio'
        if not os.path.exists('data/temp'):
            os.makedirs('data/temp', exist_ok=True)

        # 3. Configuración de yt-dlp BLINDADA contra Error 403
        ydl_opts = {
            'format': 'bestaudio/best',
            # EL DISFRAZ: Esto hace que YouTube crea que sos un Chrome normal
            'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
            'outtmpl': path_base, 
            'quiet': True,
            'noplaylist': True,
            'nocheckcertificate': True, # Ignora errores de certificados SSL
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # Buscamos en YouTube el audio de la canción de Spotify
            ydl.download([f"ytsearch1:{busqueda}"])

        # Retornamos la ruta, nombre y artista
        return f"{path_base}.mp3", nombre, artista

    except Exception as e:
        print(f"❌ Error crítico en Nandi AI: {e}")
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