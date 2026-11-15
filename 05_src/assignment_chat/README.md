# GlobeTrotter Travel Assistant

GlobeTrotter is a conversational travel assistant with a cheerful, encouraging
personality. It helps users look up current weather, discover destinations, and
estimate travel distances between cities.

## Features

### 1. Current weather

The assistant looks up a city's coordinates in `city_coordinates.csv`, calls
the Open-Meteo forecast API, and presents the current conditions, temperature,
and wind speed in a sentence.

### 2. Destination recommendations

The assistant searches a persistent ChromaDB collection of city descriptions
to find destinations relevant to a user's travel request. Search results include
the city, country, and a short description.

### 3. Distance and estimated flight time

The assistant can estimate the straight-line distance between two cities and
convert that distance into a rough flight-time estimate. This uses function
calling to invoke the `get_travel_info` function.

The distance is not an actual flight route. The time estimate uses an assumed
average speed of 900 km/h and does not account for airline schedules, stops,
airport time, or routing.

## Chat interface and memory

The app uses Gradio for its chat interface. Its system instructions give the
assistant a friendly, travel-focused personality. Conversation history is
passed to the model, and the app includes a simple memory-management function
that removes older user/assistant exchanges when the estimated history exceeds
a configured limit.

## Guardrails

The app includes checks intended to refuse requests about cats, dogs,
horoscopes, zodiac signs, and Taylor Swift. It also checks for several common
phrases asking to reveal or modify the system prompt. These are lightweight
text-based checks, not a guarantee against every possible rephrasing.

## Data and semantic-search setup

- `city_descriptions.csv` contains the city, country, and destination
  description used for semantic search.
- `city_coordinates.csv` contains the latitude and longitude used by the
  weather and travel-information services.
- `create_embeddings.py` generates an embedding for each destination
  description using the `text-embedding-3-small` model and stores the documents,
  embeddings, and city/country metadata in the persistent ChromaDB collection
  named `travel_cities`.
- The app queries that collection with an embedding of the user's request.

The embedding-generation script reads `city_descriptions.csv` and creates
`./chroma_db` relative to the working directory. The app's default database
path is the `chroma_db` directory beside `app.py`. For the default paths to
refer to the same database, run the commands below from this project directory.

The generated ChromaDB data is needed for semantic search. Include the populated
database with the project if reviewers are not expected to run the
embedding-generation script. Keep the submitted data files within the
assignment's 40 MB limit.

## Running the app

Use the course's standard environment and configured OpenAI client. The app
imports `get_client` from the course utilities, so ensure those utilities and
the required credentials/configuration are available as expected by the course
setup.

From `05_src/assignment_chat`:

1. If you need to generate or regenerate the ChromaDB collection, run:

   python create_embeddings.py

If the collection already exists, the script asks whether to overwrite it.

2. Start the chat app:

   python app.py

The app uses city_coordinates.csv and chroma_db beside app.py by default. Optional environment variables COORDS_CSV_PATH and CHROMA_DB_PATH can be used to set alternate paths.

## Project files

- app.py — Gradio interface, chat logic, guardrails, memory handling, function-calling definitions, and travel-distance calculations.
- services.py — weather API service and semantic-search service.
- create_embeddings.py — creates the persistent ChromaDB collection.
- city_descriptions.csv — destination descriptions for semantic search.
- city_coordinates.csv — coordinates for weather and travel calculations.