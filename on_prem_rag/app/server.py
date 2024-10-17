import os, sys
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from langserve import add_routes

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
from packages.opensource_rag.chain import rag_chain

app = FastAPI()


@app.get("/")
async def redirect_root_to_docs():
    return RedirectResponse("/docs")


# Edit this to add the chain you want to add
add_routes(app, rag_chain)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
