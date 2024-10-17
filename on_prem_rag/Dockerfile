# Use the official Python 3.11-slim image as the base image, which is a lightweight version of Python
FROM python:3.11

# Install Poetry version 1.8.4 for dependency management
RUN pip install poetry==1.8.4

# Configure Poetry to not create virtual environments (so it installs dependencies in the system environment)
RUN poetry config virtualenvs.create false


# Set the working directory to /code inside the container
WORKDIR /code

############

############


# Copy the essential project files (pyproject.toml, README.md, and poetry.lock if present) into the container
COPY ./pyproject.toml ./README.md ./poetry.lock* ./

# Copy the "packages" directory from the host to the container (if your code has multiple packages)
COPY ./package[s] ./packages

# Install project dependencies defined in pyproject.toml (excluding the root package)
RUN poetry install  --no-interaction --no-ansi --no-root

# Copy the "app" directory from the host into the container (this is where your application code lives)
COPY ./app ./app

# Install project dependencies, including the root package (your project itself)
RUN poetry install --no-interaction --no-ansi

# Expose port 8080 so the application can be accessed externally
EXPOSE 8080

# The default command to run Uvicorn, serving the app located in app/server.py
CMD ["uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "8080", "--reload"]