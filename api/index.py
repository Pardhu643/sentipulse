import os
import sys

# Add project root to Python path for Vercel Serverless runtime
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app

# Export Flask instance as WSGI app for Vercel
app.debug = False
