"""
app.py - Vercel Serverless & Local WSGI Entrypoint for Digital Heart Twin
"""
from api.index import app, handler

if __name__ == "__main__":
    import os
    from wsgiref.simple_server import make_server
    port = int(os.environ.get("PORT", 8000))
    print(f"Digital Heart Twin running on http://localhost:{port}")
    server = make_server("0.0.0.0", port, app)
    server.serve_forever()
