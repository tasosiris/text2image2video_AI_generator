
import os
import shutil
import yt_dlp

def download_playlist_audio(playlist_url, output_dir):
    """
    Downloads audio from a YouTube playlist to a specified directory,
    clearing the directory first.
    """
    if os.path.exists(output_dir):
        print(f"Clearing directory: {output_dir}")
        for filename in os.listdir(output_dir):
            file_path = os.path.join(output_dir, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')
    else:
        os.makedirs(output_dir)

    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'outtmpl': os.path.join(output_dir, '%(title)s.%(ext)s'),
        'ignoreerrors': True,  # Continue on download errors
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        print(f"Starting download from playlist: {playlist_url}")
        ydl.download([playlist_url])
        print("Playlist download finished.")

if __name__ == '__main__':
    playlist_url = 'https://www.youtube.com/playlist?list=PLfP6i5T0-DkIA0Bny2MV8lEF2rTI2TzdF'
    output_directory = 'music/'
    
    download_playlist_audio(playlist_url, output_directory)
    print("Download script complete.")

