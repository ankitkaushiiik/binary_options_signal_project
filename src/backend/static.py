from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from pathlib import Path
import os

def setup_static_routes(app: FastAPI):
    """Setup routes for serving static files and HTML content."""
    project_root = Path(__file__).parent.parent.parent
    html_file = project_root / 'test_client.html'
    favicon_path = project_root / 'favicon.ico'

    @app.get('/')
    async def serve_html():
        if html_file.exists():
            with open(html_file) as f:
                content = f.read()
            return HTMLResponse(content=content)
        raise HTTPException(status_code=404, detail='HTML file not found')

    @app.get('/favicon.ico')
    async def favicon():
        if favicon_path.exists():
            return FileResponse(favicon_path)
        # Return a 204 No Content if favicon doesn't exist
        return Response(status_code=204)