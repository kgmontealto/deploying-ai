# services.py
import requests
import chromadb
import os
import sys

# Allow importing from ../utils
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'utils'))
from clients import get_client

def get_weather(city_name: str, city_coords_csv: str = "city_coordinates.csv") -> str:
    """Fetch current weather from Open-Meteo and return a friendly sentence."""
    import pandas as pd
    try:
        df = pd.read_csv(city_coords_csv)
        match = df[df['city'].str.lower() == city_name.lower()]
        if match.empty:
            return f"I couldn't find '{city_name}' in my city list. Please try another city or provide coordinates."
        lat = match.iloc[0]['latitude']
        lon = match.iloc[0]['longitude']
    except Exception as e:
        return "Sorry, I couldn't look up city coordinates. Please ensure the city is in my database."

    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true"
    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code != 200:
            return "The weather service is unavailable right now. Please try later."
        data = resp.json()
        current = data["current_weather"]
        temp = current["temperature"]
        wind = current["windspeed"]
        condition_code = current["weathercode"]
        condition = weather_code_description(condition_code)
        return f"Right now in {city_name} it's {condition.lower()} with a temperature of {temp}°C and wind speed of {wind} km/h. Enjoy your trip! ✈️"
    except Exception as e:
        return "I ran into a problem fetching the weather. Please try again later."

def weather_code_description(code: int) -> str:
    mapping = {
        0: "Clear sky",
        1: "Mainly clear",
        2: "Partly cloudy",
        3: "Overcast",
        45: "Foggy",
        48: "Depositing rime fog",
        51: "Light drizzle",
        53: "Moderate drizzle",
        55: "Dense drizzle",
        61: "Slight rain",
        63: "Moderate rain",
        65: "Heavy rain",
        71: "Slight snowfall",
        73: "Moderate snowfall",
        75: "Heavy snowfall",
        80: "Slight rain showers",
        81: "Moderate rain showers",
        82: "Violent rain showers",
        95: "Thunderstorm",
        96: "Thunderstorm with slight hail",
        99: "Thunderstorm with heavy hail"
    }
    return mapping.get(code, "Unknown weather")

def init_chroma(persist_directory: str = "./chroma_db"):
    """Return a persistent ChromaDB collection (no embedding function needed)."""
    client = chromadb.PersistentClient(path=persist_directory)
    collection = client.get_or_create_collection("travel_cities")
    return collection

def semantic_search_cities(query: str, collection, n_results: int = 3) -> str:
    """
    Embed the query using the OpenAI client (via gateway) and search the ChromaDB
    collection of pre‑computed embeddings.
    """
    try:
        # Get the embedding client (respects USE_GATEWAY)
        client = get_client()
        emb = client.embeddings.create(
            model="text-embedding-3-small",
            input=[query]
        )
        query_embedding = emb.data[0].embedding

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "metadatas"]
        )
        if not results["documents"][0]:
            return "I couldn't find any matching cities for your query. Try asking about a different travel topic!"
        response = "Here are some travel ideas based on your question:\n\n"
        for i, doc in enumerate(results["documents"][0]):
            meta = results["metadatas"][0][i]
            city = meta.get("city", "Unknown")
            country = meta.get("country", "")
            response += f"**{city}, {country}**: {doc}\n\n"
        response += "Ready to pack your bags? 😊"
        return response
    except Exception as e:
        return "Sorry, I had trouble searching my travel notes. Please try again later."