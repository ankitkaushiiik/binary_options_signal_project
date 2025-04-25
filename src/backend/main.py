# File: src/backend/main.py

import asyncio
import json
import redis.asyncio as redis # Use async redis client
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Response
from typing import List, Set
import os
import traceback
from .static import setup_static_routes

# --- Configuration ---
REDIS_HOST = 'localhost'
REDIS_PORT = 6379
REDIS_CHANNEL = 'signals_channel' # Channel to listen on

app = FastAPI(title="Signal API")

# Setup static file serving
setup_static_routes(app)

class ConnectionManager:
    """Manages active WebSocket connections."""
    def __init__(self):
        self.active_connections: Set[WebSocket] = set()

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.add(websocket)
        print(f"INFO: New WebSocket connection: {websocket.client}. Total clients: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
             self.active_connections.remove(websocket)
             print(f"INFO: WebSocket disconnected: {websocket.client}. Total clients: {len(self.active_connections)}")

    async def broadcast(self, message: str):
        """Sends a message to all active WebSocket connections."""
        # Create a copy of the set to avoid modification issues during iteration
        active_connections_copy = self.active_connections.copy()
        print(f"DEBUG: Broadcasting message to {len(active_connections_copy)} clients.")
        for connection in active_connections_copy:
            try:
                await connection.send_text(message)
                # print(f"DEBUG: Sent message to {connection.client}") # Optional: Verbose logging
            except WebSocketDisconnect:
                print(f"INFO: Client {connection.client} disconnected during broadcast. Removing.")
                self.disconnect(connection) # Use the manager's disconnect method
            except Exception as e:
                # Catch other potential errors during send (e.g., connection state issues)
                print(f"WARN: Error sending message to {connection.client}: {e}. Removing connection.")
                # Attempt to close the problematic connection gracefully before removing
                try:
                     await connection.close(code=1011) # Internal error code
                except Exception as close_e:
                     print(f"WARN: Error closing connection for {connection.client} after send error: {close_e}")
                finally:
                     self.disconnect(connection) # Ensure removal even if close fails
        # print("DEBUG: Broadcast attempt finished.") # Optional: Verbose logging


manager = ConnectionManager()

async def redis_listener(pubsub: redis.client.PubSub):
    """Listens to Redis Pub/Sub and broadcasts messages."""
    print("INFO: Starting Redis listener task...")
    while True: # Keep running indefinitely
        try:
            await pubsub.subscribe(REDIS_CHANNEL)
            print(f"INFO: Subscribed to Redis channel: {REDIS_CHANNEL}")
            # Loop to listen for messages
            while True:
                 message = await pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0) # Timeout allows checking loop/connection
                 if message and message.get("type") == "message":
                     signal_data = message['data'] # Already decoded if decode_responses=True
                     print(f"DEBUG: Received from Redis: {signal_data}")
                     await manager.broadcast(signal_data) # Broadcast raw data received
                 # Add a small sleep to prevent tight loop if connection drops but no exception raised
                 await asyncio.sleep(0.01)
        except redis.exceptions.ConnectionError as e:
             print(f"ERROR: Redis connection error in listener: {e}. Attempting to reconnect in 5s...")
             await asyncio.sleep(5) # Wait before retrying subscription
             # Close the existing pubsub connection explicitly before retrying? Maybe not needed with async redis handling.
             # If using older redis lib, might need explicit pubsub.close() or similar.
        except asyncio.CancelledError:
             print("INFO: Redis listener task cancelled.")
             break # Exit loop if task is cancelled
        except Exception as e:
            print(f"ERROR: Unexpected error in Redis listener: {e}")
            print(traceback.format_exc())
            await asyncio.sleep(5) # Wait before retrying


@app.on_event("startup")
async def startup_event():
    """Connect to Redis and start listener on app startup."""
    print("INFO: Application startup...")
    redis_connected = False
    while not redis_connected:
        try:
            print(f"INFO: Attempting to connect to Redis at {REDIS_HOST}:{REDIS_PORT}...")
            app.state.redis_connection = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True, health_check_interval=30)
            await app.state.redis_connection.ping() # Verify connection
            print("INFO: Connected to Redis successfully.")
            redis_connected = True

            app.state.pubsub = app.state.redis_connection.pubsub()
            # Start listener as a background task
            app.state.redis_listener_task = asyncio.create_task(redis_listener(app.state.pubsub))
            print("INFO: Redis listener background task started.")

        except redis.exceptions.ConnectionError as e:
             print(f"WARN: Could not connect to Redis on startup at {REDIS_HOST}:{REDIS_PORT}. Retrying in 5 seconds... Error: {e}")
             await asyncio.sleep(5)
        except Exception as e:
             print(f"FATAL: Unexpected error during startup Redis connection: {e}")
             print(traceback.format_exc())
             # Decide how to handle fatal startup error (e.g., exit)
             await asyncio.sleep(5) # Prevent rapid restart loop


@app.on_event("shutdown")
async def shutdown_event():
    """Clean up Redis connection and listener task on app shutdown."""
    print("INFO: Application shutdown...")
    # Cancel the listener task
    if hasattr(app.state, 'redis_listener_task') and app.state.redis_listener_task:
         print("INFO: Cancelling Redis listener task...")
         app.state.redis_listener_task.cancel()
         try:
             await app.state.redis_listener_task # Wait for task to finish cancellation
         except asyncio.CancelledError:
             print("INFO: Redis listener task successfully cancelled.")
         except Exception as e:
              print(f"WARN: Error during listener task cancellation: {e}")

    # Close PubSub (may not be strictly necessary if connection is closed)
    if hasattr(app.state, 'pubsub') and app.state.pubsub:
        try:
             await app.state.pubsub.close()
             print("INFO: Redis PubSub closed.")
        except Exception as e:
             print(f"WARN: Error closing PubSub: {e}")

    # Close main Redis connection
    if hasattr(app.state, 'redis_connection') and app.state.redis_connection:
        try:
             await app.state.redis_connection.close()
             print("INFO: Redis connection closed.")
        except Exception as e:
             print(f"WARN: Error closing Redis connection: {e}")


@app.websocket("/ws/signals") # More specific endpoint path
async def websocket_endpoint(websocket: WebSocket):
    """Handles incoming WebSocket connections for signals."""
    await manager.connect(websocket)
    try:
        # Keep connection alive, could optionally handle messages from client here
        while True:
            # data = await websocket.receive_text() # Example: if client sends data
            # print(f"DEBUG: Received from client {websocket.client}: {data}")
            await asyncio.sleep(60) # Check connection state periodically or just keep alive
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
         print(f"ERROR: Unhandled exception in WebSocket connection {websocket.client}: {e}")
         print(traceback.format_exc())
         manager.disconnect(websocket) # Ensure cleanup on unexpected error


# Simple health check endpoint
@app.get("/health")
async def health_check():
    # Check Redis connection state if possible
    redis_status = "disconnected"
    if hasattr(app.state, 'redis_connection') and app.state.redis_connection:
         try:
             if await app.state.redis_connection.ping():
                 redis_status = "connected"
         except redis.exceptions.ConnectionError:
             redis_status = "connection_error"
         except Exception:
              redis_status = "check_error"

    return {"status": "ok", "redis_status": redis_status, "active_ws_clients": len(manager.active_connections)}

# To run this use: uvicorn src.backend.main:app --reload --port 8000
