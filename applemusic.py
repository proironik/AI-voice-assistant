"""
Apple Music (Windows) — open only
─────────────────────────────────
This is intentionally minimal: it opens the Apple Music app, nothing more.

Why there is no "play <song>" here
──────────────────────────────────
The Windows Apple Music app (AppleInc.AppleMusicWin) ignores deep-link *paths*.
Every URL form was tested live on this machine — itmss://, music://, and plain
https://music.apple.com/ track, album, and search links — and every one simply
brought the app to the foreground on its last screen without navigating to the
item. The OS registers music.apple.com as an app-associated host, so even the
https link is handed to the app rather than a browser, and the app discards the
path.

Real name-based playback would require Apple MusicKit, which needs the paid
Apple Developer membership. Rather than pretend, the assistant opens the app and
tells the user to pick the song themselves (see commands._plan_music). Playback
control once something *is* playing already works through the Windows media
session (see playback.py), the same way it does for Spotify.

If a future app build honours deep links, resolution can come back: the free
iTunes Search API (https://itunes.apple.com/search) returns real
music.apple.com song/album/artist URLs and would drop straight into a resolver
shaped like ytmusic.py. It does not index playlists (entity=playlist -> HTTP
400), so those could never be resolved this way.
"""

# US storefront. The browse page is only used to launch the app.
STOREFRONT = "us"
HOME_URL = f"https://music.apple.com/{STOREFRONT}/browse"
